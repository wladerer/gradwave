"""run_phonons and its parallel spoke workers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from gradwave.api._common import _DEFAULT_MIXING_HISTORY, XC_REGISTRY
from gradwave.api.dispersion import _compute_dispersion
from gradwave.api.system import _as_paws, _as_upfs, _is_uspp, _species_upfs, _spin_setup
from gradwave.core.xc.base import XCFunctional
from gradwave.inputs import Input

if TYPE_CHECKING:

    from gradwave.scf.loop import SCFResult
    from gradwave.scf.results import USPPResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SeedPool: forward-only parallel dispatch for the hub-and-spoke campaigns.
#
# Each campaign computes its reference SCF ONCE (serially), checkpoints it, and
# fans the independent spoke SCFs out to worker PROCESSES that warm-start from
# that checkpoint (postscf.seedpool.map_spokes). The specs below are picklable
# NamedTuples; the worker functions are module-level (picklable by reference)
# and reload upfs/xc from `inp` inside the worker (nothing heavy is pickled —
# only the checkpoint PATH and cheap ndarray/scalar spoke parameters cross the
# process boundary). forward-only: the workers return plain arrays; autograd
# graphs are process-local and do NOT survive the pool, so a differentiable
# campaign must use n_workers=1 (the serial path, which keeps the live result).
# ---------------------------------------------------------------------------


# --- phonons (displacement spokes) -----------------------------------------
def _phonon_run_scf(
    inp: Input, scmap: Any, ksuper: Any, upfs: Any, uspp: bool, xc: Any,
    mags: Any, pos_sc: Any, start_from: Any,
) -> SCFResult | USPPResult:
    """One supercell SCF at displaced positions ``pos_sc``. Module-level so the
    serial (run_phonons make_scf) and parallel (worker) paths share ONE SCF
    construction — the two cannot drift."""
    if uspp:
        from gradwave.scf.uspp import scf_uspp, setup_uspp

        usystem = setup_uspp(
            scmap.cell_super, pos_sc, scmap.species_super, _as_paws(upfs),
            ecut=inp.ecut, kmesh=ksuper, ecutrho=inp.ecutrho,
            use_symmetry=False)
        if inp.device != "cpu":
            usystem = usystem.to(inp.device)
        return scf_uspp(
            usystem, xc, nspin=inp.nspin, start_mag=mags,
            smearing=inp.smearing.type, width=inp.smearing.width,
            etol=inp.scf.etol, rhotol=inp.scf.rhotol,
            mixing_alpha=inp.scf.mixing.alpha,
            mixing_history=inp.scf.mixing.history,
            diago_tol=inp.scf.diago_tol, start_from=start_from, verbose=False)
    from gradwave.scf.loop import scf, setup_system

    system = setup_system(
        scmap.cell_super, pos_sc, scmap.species_super, _as_upfs(upfs),
        ecut=inp.ecut, kmesh=ksuper, kshift=inp.kpoints.shift,
        use_symmetry=False)
    if inp.device != "cpu":
        system = system.to(inp.device)
    spin_kw: dict[str, Any] = (
        {"tot_magnetization": inp.tot_magnetization}
        if inp.nspin == 2 and inp.tot_magnetization is not None else {})
    return scf(system, cast("XCFunctional", xc), nspin=inp.nspin,
               start_mag=mags, smearing=inp.smearing.type,
               width=inp.smearing.width, etol=inp.scf.etol, rhotol=inp.scf.rhotol,
               mixing_alpha=inp.scf.mixing.alpha,
               mixing_history=inp.scf.mixing.history or _DEFAULT_MIXING_HISTORY,
               precond=inp.scf.mixing.precond, diago_tol=inp.scf.diago_tol,
               start_from=start_from, verbose=False, **spin_kw)


def _phonon_force(res: Any, xc: Any, uspp: bool) -> Any:
    """Analytic force (N_sc, 3) [eV/Å] for the FD fold — USPP/PAW routes through
    the augmentation-aware forces_uspp, NC through the default force path."""
    if uspp:
        from gradwave.postscf.paw_forces import forces_uspp

        return forces_uspp(res, xc)
    from gradwave.postscf.forces import forces

    return forces(res, xc=xc)


def _phonon_rebuild(inp: Input) -> tuple[Any, bool, Any, Any]:
    """Reconstruct ``(upfs, uspp, xc, mags)`` from ``inp`` inside a worker,
    mirroring run_phonons' setup (path-cached upf load, nspin xc/mags)."""
    _species, upfs, _soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc, mags = XC_REGISTRY[inp.xc](), None
    return upfs, uspp, xc, mags


class _PhononSpoke(NamedTuple):
    inp: Input
    scmap: Any
    ksuper: tuple[int, int, int]
    pos_sc: Any            # (N_sc, 3) displaced positions [Å], numpy
    ckpt_path: str
    tag: tuple[int, int, int]   # (home atom a, axis i, sign)


def _phonon_spoke_worker(spoke: _PhononSpoke) -> tuple[tuple[int, int, int], Any]:
    """Worker: build the displaced supercell, warm-start from the reference
    checkpoint, run the SCF and return ``(tag, force[N_sc,3])`` (numpy)."""
    from gradwave.io.checkpoint import as_start_from, load_checkpoint

    upfs, uspp, xc, mags = _phonon_rebuild(spoke.inp)
    start_from = as_start_from(load_checkpoint(spoke.ckpt_path))
    res = _phonon_run_scf(spoke.inp, spoke.scmap, spoke.ksuper, upfs, uspp,
                          xc, mags, spoke.pos_sc, start_from)
    return spoke.tag, _phonon_force(res, xc, uspp).detach().cpu().numpy()


def _phonons_fc_parallel(
    inp: Input, scmap: Any, ksuper: Any, upfs: Any, uspp: bool, xc: Any,
    mags: Any, *, h: float, n_workers: int, verbose: bool,
) -> Any:
    """Force constants Φ_home via SeedPool: reference SCF serially, then the
    6·N_prim displacement SCFs across worker processes, each warm-started from
    the reference checkpoint. The FD/symmetrize/ASR assembly is shared with the
    serial path (postscf.phonons_supercell.force_constants_from_forces)."""
    import os
    import tempfile

    from gradwave.io.checkpoint import save_checkpoint
    from gradwave.postscf.phonons_supercell import (
        displacement_list,
        force_constants_from_forces,
    )
    from gradwave.postscf.seedpool import map_spokes

    ref = _phonon_run_scf(inp, scmap, ksuper, upfs, uspp, xc, mags,
                          scmap.positions_super.copy(), None)
    with tempfile.TemporaryDirectory(prefix="gw_seedpool_") as td:
        ckpt = os.path.join(td, "ref.ckpt")
        save_checkpoint(ref, ckpt)
        spokes = [_PhononSpoke(inp, scmap, ksuper, pos, ckpt, (a, i, sign))
                  for (a, i, sign, pos) in displacement_list(scmap, h)]
        out = map_spokes(_phonon_spoke_worker, spokes,
                         n_workers=n_workers, verbose=verbose)
    force_map = {tag: f for tag, f in out}
    return force_constants_from_forces(force_map, scmap, h)


def run_phonons(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Supercell finite-displacement phonons: dispersion along a q-path + a
    phonon DOS on a q-mesh.

    Builds the diagonal supercell, displaces only the primitive home-cell atoms
    (the translational reduction — 6·N_prim SCFs regardless of supercell size,
    each warm-started from the undisplaced reference), folds the force constants
    to D(q) and diagonalizes. Norm-conserving OR ultrasoft/PAW, nspin ∈ {1, 2}
    (the analytic forces sum per spin channel — see postscf.forces /
    postscf.paw_forces).

    ``phonons.n_workers > 1`` (SeedPool) fans the 6·N_prim displacement SCFs out
    to that many worker processes, each warm-started from the shared undisplaced
    reference checkpoint (thread budget split ~total/n_workers per worker). This
    is forward-only: worker autograd graphs do not cross processes, so a
    differentiable phonon run must keep ``n_workers=1`` (the serial path). It
    composes with a future irreducible-displacement reduction — the parallel
    dispatch runs over whatever displacement set ``displacement_list`` yields, so
    reducing 6·N_prim to the symmetry-irreducible spokes simply shortens the list
    the pool consumes."""
    import numpy as np

    from gradwave.postscf.phonons_supercell import (
        build_supercell,
        dispersion,
        force_constants_home,
        phonon_dos,
    )

    if inp.noncollinear:
        raise NotImplementedError(
            "supercell phonons do not support noncollinear/spinor runs "
            "(the finite-displacement force fold uses the collinear force path)")
    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)

    if inp.nspin == 2:
        xc, mags = _spin_setup(inp)
    else:
        xc, mags = XC_REGISTRY[inp.xc](), None
    cell = np.asarray(inp.atoms.cell.array, dtype=float)
    positions = inp.atoms.get_positions()
    masses = inp.atoms.get_masses()  # primitive-atom masses [amu]
    n = inp.phonons.supercell
    scmap = build_supercell(cell, positions, species_of_atom, n)
    # fold the primitive k-mesh by the supercell size (equivalent BZ sampling)
    ksuper = tuple(max(1, inp.kpoints.mesh[i] // n[i]) for i in range(3))

    def make_scf(pos_sc: Any, start_from: Any = None) -> SCFResult | USPPResult:
        # thin wrapper over the module-level _phonon_run_scf so the serial
        # (here) and parallel (worker) paths share ONE SCF construction — the
        # (tot_magnetization pin / device move / nspin seed) logic lives there.
        return _phonon_run_scf(inp, scmap, ksuper, upfs, uspp, xc, mags,
                               pos_sc, start_from)

    if verbose:
        sym_note = (" (point-group-reduced set)"
                    if inp.phonons.use_displacement_symmetry else "")
        print(f"phonons: {tuple(n)} supercell ({scmap.n_sc} atoms), displacing "
              f"{scmap.n_prim} home atoms → ≤{6 * scmap.n_prim} SCFs{sym_note}, "
              f"k-mesh {ksuper}", flush=True)
    # xc is needed by the force path only for NLCC species (spin-resolved for
    # nspin=2); it is ignored for valence-only pseudos. USPP/PAW routes the fold
    # through the augmentation-aware paw_forces.forces_uspp; NC uses the default.
    force_fn: Callable[[Any], Any] | None = None
    if uspp:
        from gradwave.postscf.paw_forces import forces_uspp

        def force_fn(res: Any) -> Any:
            return forces_uspp(res, xc)
    if inp.dispersion.enabled:
        # Fold the Grimme D3/D4 dispersion FORCES into every displaced-geometry
        # force. The bare-DFT phonon path omits them, so the interlayer (vdW-bound)
        # branches of a layered material come out unphysically soft; adding the
        # analytic dispersion force at each displaced supercell geometry stiffens
        # them. The correction is recomputed per displacement because it depends on
        # the displaced positions (res.system carries them). Applies on the SERIAL
        # force path below; the SeedPool parallel workers rebuild forces themselves
        # and do not add D3 in v1 (keep n_workers=1 for a dispersion-corrected run).
        import numpy as _np
        import torch as _torch

        from gradwave.postscf.forces import forces as _nc_forces

        _dp = inp.dispersion
        _method = _dp.method.lower()
        _zprim = inp.atoms.get_atomic_numbers()
        _sp2z = {species_of_atom[a]: int(_zprim[a]) for a in range(len(_zprim))}
        _zsuper = [_sp2z[s] for s in scmap.species_super]
        _base_ff = force_fn

        def force_fn(res: Any) -> Any:  # noqa: F811
            f = _base_ff(res) if _base_ff is not None else _nc_forces(res, xc=xc)
            system = res.system
            pos = system.positions.detach().to(_torch.float64)
            cell_sc = _np.asarray(system.grid.cell, dtype=_np.float64)
            fd = _compute_dispersion(
                pos, cell_sc, _zsuper, method=_method,
                functional=_dp.functional or inp.xc, charge=_dp.charge,
                cutoff_ang=_dp.cutoff, cn_cutoff_ang=_dp.cn_cutoff,
                s6=_dp.s6, s8=_dp.s8, a1=_dp.a1, a2=_dp.a2, need_stress=False,
            ).forces
            return f + fd.to(f.dtype).to(f.device)
    # SeedPool: fan the 6·N_prim displacement SCFs across worker processes (each
    # warm-started from the undisplaced reference checkpoint). Forced serial under
    # `distributed`. SeedPool (parallelize the full displacement set) and the
    # point-group displacement reduction (fewer SCFs, serial) are alternative ways
    # to cut the phonon cost and do not compose in v1 — parallel wins if both are
    # requested; the reduction (and the D3 fold above) apply on the serial path.
    n_workers = 1 if inp.distributed else (inp.phonons.n_workers or 1)
    if n_workers > 1:
        phi = _phonons_fc_parallel(inp, scmap, ksuper, upfs, uspp, xc, mags,
                                   h=inp.phonons.displacement,
                                   n_workers=n_workers, verbose=verbose)
    else:
        phi = force_constants_home(make_scf, scmap, h=inp.phonons.displacement,
                                   xc=xc, force_fn=force_fn,
                                   use_displacement_symmetry=(
                                       inp.phonons.use_displacement_symmetry),
                                   verbose=verbose)

    bp = inp.atoms.cell.bandpath(path=inp.phonons.path or None,
                                 npoints=inp.phonons.npoints)
    qpts = np.asarray(bp.kpts)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    freqs = dispersion(phi, scmap, masses, qpts)  # (nq, 3·N_prim) [cm⁻¹]
    block = {
        "supercell": list(n),
        "n_atoms_supercell": scmap.n_sc,
        "displacement_ang": inp.phonons.displacement,
        "kmesh_supercell": list(ksuper),
        "qpts_frac": qpts.tolist(),
        "x": np.asarray(x).tolist(),
        "labels": list(zip(xticks.tolist(), list(xlabels), strict=True)),
        "frequencies_cm1": freqs.tolist(),
        "min_frequency_cm1": float(freqs.min()),
    }
    if min(inp.phonons.dos_mesh) > 0:
        from gradwave.kpoints import monkhorst_pack

        qmesh, weights = monkhorst_pack(inp.phonons.dos_mesh)
        grid, dos = phonon_dos(phi, scmap, masses, qmesh, weights,
                               width=inp.phonons.dos_width)
        block["dos"] = {"frequency_cm1": grid.tolist(), "dos": dos.tolist(),
                        "mesh": list(inp.phonons.dos_mesh)}
        # Harmonic thermodynamics from the DOS we just built (F, U, Cv, S vs T,
        # plus ZPE and θ_D). The thermo integrals drop imaginary branches, so
        # flag them from the raw mode frequencies — the numbers are only
        # meaningful for a dynamically stable (all-real) structure.
        from gradwave.postscf.thermo import thermo_table

        thermo_block = thermo_table(grid, dos, inp.phonons.thermo_temperatures)
        thermo_block["imaginary_modes_present"] = bool(freqs.min() < -1.0)
        block["thermo"] = thermo_block
    if verbose:
        print(f"phonons: min frequency {freqs.min():.1f} cm⁻¹ "
              f"({'has imaginary modes' if freqs.min() < -1.0 else 'all real'})",
              flush=True)
    return block
