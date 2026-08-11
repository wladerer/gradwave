"""run_elastic and its parallel spoke workers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from gradwave.api._common import SPIN_XC_REGISTRY, XC_REGISTRY
from gradwave.api.scf import run_scf
from gradwave.api.system import _as_paws, _as_upfs, _fft_grid, _is_uspp, _species_upfs
from gradwave.inputs import Input

if TYPE_CHECKING:
    from ase import Atoms

    from gradwave.scf.loop import System
    from gradwave.scf.uspp_setup import USPPSystem

logger = logging.getLogger(__name__)


# --- elastic (strain spokes; clamped-ion only) -----------------------------
def _elastic_rebuild(inp: Input) -> tuple[Any, bool, Any, Any, bool]:
    """Reconstruct ``(upfs, uspp, species_of_atom, xc, is_fr)`` from ``inp``
    inside a worker, mirroring run_elastic's setup."""
    _species, upfs, soa = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    is_fr = any(b.j is not None for u in upfs for b in u.betas)
    if inp.noncollinear:
        from gradwave.core.xc.noncollinear import NoncollinearXC

        xc: Any = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    elif inp.nspin == 2:
        xc = SPIN_XC_REGISTRY[inp.xc]()
    else:
        xc = XC_REGISTRY[inp.xc]()
    return upfs, uspp, soa, xc, is_fr


def _elastic_time_reversal(inp: Input) -> bool:
    """k ≡ −k holds unless a magnetic spinor breaks it (mirrors run_elastic)."""
    return not (inp.noncollinear and not inp.nonmagnetic)


def _elastic_build(
    inp: Input, upfs: Any, uspp: bool, soa: Any, cell: Any,
    fixed: Any, time_reversal: bool,
) -> System | USPPSystem:
    """Build the strained system on the pinned FFT grid (mirrors run_elastic's
    ``_build`` closure; fractional coordinates held fixed = clamped-ion)."""
    import numpy as np

    pos = inp.atoms.get_scaled_positions() @ np.asarray(cell, dtype=float)
    if uspp:
        from gradwave.scf.uspp import setup_uspp

        return setup_uspp(
            cell, pos, soa, _as_paws(upfs), ecut=inp.ecut,
            kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
            use_symmetry=inp.symmetry, fft_shape=fixed)
    from gradwave.scf.loop import setup_system

    return setup_system(
        cell=cell, positions=pos, species_of_atom=soa, upfs=_as_upfs(upfs),
        ecut=inp.ecut, kmesh=inp.kpoints.mesh, kshift=inp.kpoints.shift,
        nbands=inp.nbands, use_symmetry=inp.symmetry,
        time_reversal=time_reversal, fft_shape=fixed)


def _elastic_stress(res: Any, xc: Any, uspp: bool) -> Any:
    """Analytic stress (3, 3) [eV/Å³] — USPP/PAW vs norm-conserving path."""
    if uspp:
        from gradwave.postscf.paw_stress import stress_uspp

        return stress_uspp(res, xc)
    from gradwave.postscf.stress import stress

    return stress(res, xc)


def _epskey(eps: Any) -> tuple[float, ...]:
    """Hashable key for a 3×3 strain tensor, so the parallel path can back an
    ``elastic.elastic_tensor`` closure with precomputed stresses (elastic_tensor
    calls it with the exact voigt_strain_tensor(j, ±h) tensors)."""
    import numpy as np

    return tuple(np.round(np.asarray(eps, dtype=float).ravel(), 12).tolist())


class _ElasticSpoke(NamedTuple):
    inp: Input
    cell0: Any
    fixed: Any
    eps: Any               # 3×3 strain tensor
    ckpt_path: str
    key: tuple[float, ...]


def _elastic_spoke_worker(
    spoke: _ElasticSpoke,
) -> tuple[tuple[float, ...], Any, bool]:
    """Worker: strain the cell, warm-start from the reference checkpoint, run
    the SCF and return ``(key, stress[3,3], converged)`` (numpy)."""
    import numpy as np

    from gradwave.checkpoint import as_start_from, load_checkpoint

    upfs, uspp, soa, xc, _is_fr = _elastic_rebuild(spoke.inp)
    tr = _elastic_time_reversal(spoke.inp)
    cell = spoke.cell0 @ (np.eye(3) + spoke.eps).T
    system = _elastic_build(spoke.inp, upfs, uspp, soa, cell, spoke.fixed, tr)
    start_from = as_start_from(load_checkpoint(spoke.ckpt_path))
    res = run_scf(spoke.inp, system=system, verbose=False, start_from=start_from)
    sigma = _elastic_stress(res, xc, uspp).detach().cpu().numpy()
    return spoke.key, sigma, bool(getattr(res, "converged", True))


def run_elastic(inp: Input, verbose: bool = True) -> dict[str, Any]:
    """Elastic constants: FD of the analytic stress over the six Voigt strains
    → the 6×6 stiffness C and Voigt–Reuss–Hill moduli.

    ``elastic.mode`` picks the tensor. ``"clamped"`` (default) deforms the cell
    with fractional coordinates fixed and re-relaxes only the electrons.
    ``"relaxed"`` additionally relaxes the internal coordinates at every
    strained cell (fixed-cell BFGS on the analytic forces, gated by
    ``elastic.fmax``) before reading the stress, giving the relaxed-ion tensor
    C_ij = dσ_i/dε_j along the relaxed path — the physical constants for
    crystals whose compliance is dominated by symmetry-allowed internal
    displacement (quartz, diamond/zincblende shear). The input geometry is
    assumed to be the equilibrium one; the residual reference fmax is reported
    so a non-equilibrium start is visible in the output.

    Each strain runs on one FFT grid pinned across the scan; every strained SCF
    warm-starts from the unstrained reference (in relaxed mode, each ionic step
    warm-starts from the previous one). Norm-conserving (``postscf.stress``) and
    USPP/PAW (``postscf.paw_stress``) are both handled, for nspin=1 and
    collinear nspin=2 (the fixed-basis stress sums per spin channel — see
    postscf.stress).

    Fully-relativistic (spin-orbit) spinor runs (``noncollinear: true`` with
    j-resolved pseudos) are handled on the norm-conserving path:
    ``postscf.stress.stress`` differentiates the spinor strained energy
    (``_energy_strained_fr``), so the same FD-of-analytic-stress driver folds it
    into C. DFT+U on that path is a feature boundary (#142), and the USPP/PAW
    spinor stress has no path yet, so both are rejected below.

    ``elastic.n_workers > 1`` (SeedPool) fans the 12 CLAMPED-ion strain SCFs out
    to that many worker processes, each warm-started from the shared unstrained
    reference checkpoint. Relaxed-ion stays serial (its per-strain BFGS is a
    nested optimization, not one forward SCF). Forward-only: a differentiable
    elastic run must keep ``n_workers=1``. It composes with a future
    irreducible-strain reduction — the pool runs over whatever strain spoke set
    is generated."""
    import numpy as np

    from gradwave.postscf.elastic import (
        elastic_tensor,
        is_mechanically_stable,
        moduli_from_cij,
    )

    _species, upfs, species_of_atom = _species_upfs(inp)
    uspp = _is_uspp(upfs)
    is_fr = any(b.j is not None for u in upfs for b in u.betas)
    relaxed_ion = inp.elastic.mode == "relaxed"
    if relaxed_ion and inp.noncollinear:
        raise NotImplementedError(
            "relaxed-ion elastic constants need the collinear force path — "
            "noncollinear/spinor forces are not implemented (use "
            "elastic.mode: clamped)")
    if inp.noncollinear:
        if uspp:
            raise NotImplementedError(
                "elastic constants for noncollinear USPP/PAW runs are not "
                "supported (no spinor USPP/PAW stress path)")
        if not is_fr:
            raise NotImplementedError(
                "elastic constants for a magnetic non-collinear run without "
                "spin-orbit need fully-relativistic (j-resolved) pseudos — the "
                "scalar stress path has no magnetic-spinor route")
        if inp.hubbard.enabled:
            raise NotImplementedError(
                "DFT+U elastic constants on the spin-orbit path (feature "
                "boundary, #142)")

    if inp.noncollinear:
        from gradwave.core.xc.noncollinear import NoncollinearXC
        xc = NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]())
    elif inp.nspin == 2:
        xc = SPIN_XC_REGISTRY[inp.xc]()
    else:
        xc = XC_REGISTRY[inp.xc]()
    if uspp:
        from gradwave.postscf.paw_stress import stress_uspp as _stress
    else:
        from gradwave.postscf.stress import stress as _stress

    cell0 = np.asarray(inp.atoms.cell.array, dtype=float)
    frac = inp.atoms.get_scaled_positions()
    natoms = len(inp.atoms)
    h = inp.elastic.strain

    # a magnetic spinor breaks k ≡ −k (TR flips m⃗); a nonmagnetic spinor
    # (SOC only) keeps Kramers, matching build_system's time-reversal logic.
    time_reversal = not (inp.noncollinear and not inp.nonmagnetic)

    def _build(
        cell: Any, fft_shape: tuple[int, ...] | None, pos: Any = None
    ) -> System | USPPSystem:
        if pos is None:
            pos = frac @ cell
        if uspp:
            from gradwave.scf.uspp import setup_uspp

            return setup_uspp(
                cell, pos, species_of_atom, _as_paws(upfs), ecut=inp.ecut,
                kmesh=inp.kpoints.mesh, ecutrho=inp.ecutrho, nbands=inp.nbands,
                use_symmetry=inp.symmetry, fft_shape=fft_shape)
        from gradwave.scf.loop import setup_system

        return setup_system(
            cell=cell, positions=pos, species_of_atom=species_of_atom,
            upfs=_as_upfs(upfs), ecut=inp.ecut, kmesh=inp.kpoints.mesh,
            kshift=inp.kpoints.shift, nbands=inp.nbands,
            use_symmetry=inp.symmetry, time_reversal=time_reversal,
            fft_shape=fft_shape)

    # pin one FFT grid: the +h strains give the largest cells / finest grids
    from gradwave.postscf.elastic import voigt_strain_tensor

    probe = [cell0] + [cell0 @ (np.eye(3) + voigt_strain_tensor(j, h)).T
                       for j in range(6)]
    fixed = tuple(max(int(_fft_grid(_build(c, None)).shape[i]) for c in probe)
                  for i in range(3))

    # opt-in Laue-point-group strain reduction: run only the irreducible Voigt
    # strains and reconstruct C by the group action (mirrors
    # phonons.use_displacement_symmetry). The pinned FFT box is handed in so the
    # reducer can refuse an anisotropic grid and fall back to the full 6 strains.
    strain_sym = None
    if inp.elastic.use_strain_symmetry:
        from gradwave.postscf.elastic import ElasticStrainSymmetry

        strain_sym = ElasticStrainSymmetry(cell0, frac, species_of_atom,
                                           fft_shape=fixed)
        if not strain_sym.can_reduce:
            strain_sym = None
    if verbose:
        sym_note = ""
        if inp.elastic.use_strain_symmetry:
            sym_note = (f", strain symmetry {len(strain_sym.strains)}/6 "
                        f"irreducible ({strain_sym.sg.international})"
                        if strain_sym is not None
                        else ", strain symmetry off (unsafe → full set)")
        print(f"elastic: {inp.elastic.mode}-ion, strain h={h}, "
              f"fixed FFT grid {fixed}{sym_note}", flush=True)

    # reference SCF once — warm-start seed and residual-stress readout
    ref = run_scf(inp, system=_build(cell0, fixed), verbose=False)
    converged = [bool(getattr(ref, "converged", True))]
    # _stress is stress_uspp (res: USPPResult, xc: XCFunctional | SpinXC) or
    # stress (res: SCFResult | NCResult, xc: XCFunctional | SpinXC |
    # NoncollinearXC) depending on the branch above; ty can't correlate which
    # member of that function union is bound with which result/xc value this
    # call actually has (the uspp/noncollinear guards above are runtime, not
    # visible to the type checker across the import-time branch), so the seam
    # needs Any on both arguments, not a specific cast.
    sigma_ref = _stress(cast(Any, ref), cast(Any, xc)).detach().cpu().numpy()

    # relaxed-ion machinery: analytic forces for the per-strain fixed-cell BFGS
    ref_fmax: float | None = None
    relax_steps: list[int] = []
    relax_conv: list[bool] = []
    if relaxed_ion:
        if uspp:
            from gradwave.postscf.paw_forces import forces_uspp

            def _forces(res: Any) -> Any:
                return forces_uspp(res, cast(Any, xc)).detach().cpu().numpy()
        else:
            from gradwave.postscf.forces import forces as _forces_nc

            def _forces(res: Any) -> Any:
                return _forces_nc(res, xc=cast(Any, xc)).detach().cpu().numpy()

        ref_fmax = float(np.linalg.norm(_forces(ref), axis=1).max())
        if verbose and ref_fmax > inp.elastic.fmax:
            print(f"elastic: reference fmax {ref_fmax:.4f} eV/Å exceeds the "
                  f"relax gate {inp.elastic.fmax} — input geometry is not at "
                  "equilibrium, C is about the current state", flush=True)

    def _relax_ions(cell: Any) -> Any:
        """Fixed-cell BFGS on the analytic forces at a strained cell; returns
        the SCF result at the relaxed geometry (each ionic step warm-starts
        from the previous one, the first from the unstrained reference)."""
        from ase import Atoms as _AseAtoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.optimize import BFGS
        from typing_extensions import override

        state: dict[str, Any] = {"prev": ref, "res": None}

        class _FixedCellCalc(Calculator):
            implemented_properties = ["energy", "forces"]

            # ASE's signature (atoms/properties/system_changes defaults) — the
            # base class drives this; only the SCF+forces body is ours.
            @override
            def calculate(self, atoms: Any = None,
                          properties: Any = ("energy",),
                          system_changes: Any = all_changes) -> None:
                super().calculate(atoms, properties, system_changes)
                pos = cast("Atoms", self.atoms).get_positions()
                res = run_scf(
                    inp, system=_build(cell, fixed, pos=pos),
                    verbose=False, start_from=state["prev"])
                state["prev"] = state["res"] = res
                converged.append(bool(getattr(res, "converged", True)))
                self.results = {
                    "energy": float(res.energies.total),
                    "forces": _forces(res),
                }

        atoms = _AseAtoms(inp.atoms.get_chemical_symbols(), scaled_positions=frac,
                          cell=np.asarray(cell), pbc=True)
        atoms.calc = _FixedCellCalc()
        opt = BFGS(atoms, logfile=None)
        ok = bool(opt.run(fmax=inp.elastic.fmax, steps=inp.elastic.max_steps))
        relax_steps.append(int(opt.nsteps))
        relax_conv.append(ok)
        if verbose:
            fnow = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            print(f"elastic: strain {len(relax_steps):>2d}/12 ions relaxed in "
                  f"{opt.nsteps} steps (fmax {fnow:.4f} eV/Å"
                  f"{'' if ok else ', NOT converged'})", flush=True)
        return state["res"]

    def _stress_at(eps: Any) -> Any:
        cell = cell0 @ (np.eye(3) + eps).T
        if relaxed_ion:
            res = _relax_ions(cell)
        else:
            res = run_scf(inp, system=_build(cell, fixed), verbose=False,
                          start_from=ref)
            converged.append(bool(getattr(res, "converged", True)))
        return _stress(cast(Any, res), cast(Any, xc)).detach().cpu().numpy()

    # SeedPool: the 12 clamped-ion strain SCFs are independent forward runs, so
    # fan them across worker processes (each warm-started from the unstrained
    # reference checkpoint). Relaxed-ion stays serial — its per-strain BFGS is a
    # nested optimization, not one forward SCF — and `distributed` forces serial
    # (that path shards k across ranks; the two parallelisms don't compose in v1).
    # StrainStar symmetry (`strain_sym`) still applies either way: elastic_tensor
    # only evaluates the Laue-irreducible strains and rebuilds the rest by symmetry
    # (the parallel path precomputes all 12; the reduction just uses fewer of them).
    n_workers = 1 if inp.distributed else (inp.elastic.n_workers or 1)
    if n_workers > 1 and not relaxed_ion:
        import os
        import tempfile

        from gradwave.checkpoint import save_checkpoint
        from gradwave.postscf.seedpool import map_spokes

        with tempfile.TemporaryDirectory(prefix="gw_seedpool_") as td:
            ckpt = os.path.join(td, "ref.ckpt")
            save_checkpoint(ref, ckpt)
            spokes = []
            for j in range(6):
                for sgn in (+1, -1):
                    eps = voigt_strain_tensor(j, sgn * h)
                    spokes.append(
                        _ElasticSpoke(inp, cell0, fixed, eps, ckpt, _epskey(eps)))
            out = map_spokes(_elastic_spoke_worker, spokes,
                             n_workers=n_workers, verbose=verbose)
        stress_map = {key: sigma for key, sigma, _conv in out}
        converged.extend(bool(conv) for _key, _sigma, conv in out)

        # back elastic_tensor with the precomputed stresses — it calls the
        # closure with the exact voigt_strain_tensor(j, ±h) tensors we keyed.
        def _stress_at_precomputed(eps: Any) -> Any:
            return stress_map[_epskey(eps)]

        c = elastic_tensor(_stress_at_precomputed, h=h, symmetry=strain_sym)
    else:
        c = elastic_tensor(_stress_at, h=h, symmetry=strain_sym)
    mod = moduli_from_cij(c)
    resid_gpa = float(np.abs(sigma_ref).max()) * 160.2176634
    block: dict[str, Any] = {
        "strain": h,
        "mode": inp.elastic.mode,
        "n_atoms": natoms,
        "formalism": "uspp/paw" if uspp else "nc",
        "c_GPa": c.tolist(),
        "bulk_modulus_GPa": {"voigt": mod.bulk_voigt, "reuss": mod.bulk_reuss,
                             "hill": mod.bulk_hill},
        "shear_modulus_GPa": {"voigt": mod.shear_voigt, "reuss": mod.shear_reuss,
                              "hill": mod.shear_hill},
        "young_modulus_GPa": mod.young,
        "poisson_ratio": mod.poisson,
        "mechanically_stable": is_mechanically_stable(c),
        "residual_stress_GPa": resid_gpa,
        "all_converged": all(converged),
        "n_strains": len(strain_sym.strains) if strain_sym is not None else 6,
    }
    if relaxed_ion:
        block["relax_fmax"] = inp.elastic.fmax
        block["ref_fmax_eV_ang"] = ref_fmax
        # 12 entries in FD order: strain j = 0..5, +h then −h for each
        block["relax_steps"] = relax_steps
        block["relax_all_converged"] = all(relax_conv)
    if verbose:
        print(f"elastic: K={mod.bulk_hill:.1f}  G={mod.shear_hill:.1f}  "
              f"E={mod.young:.1f} GPa  ν={mod.poisson:.3f}  "
              f"stable={block['mechanically_stable']}", flush=True)
    return block
