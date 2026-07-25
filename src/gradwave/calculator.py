"""ASE Calculator interface (Layer C).

Supports energy, forces, and stress (fixed-basis, via the differentiable
radial transforms in pseudo/radial_torch.py), so variable-cell relaxation
through ase.filters.FrechetCellFilter works. The usual plane-wave caveat
applies: relaxing the cell at fixed ecut carries Pulay (basis-incompleteness)
pressure — converge ecut or re-relax at the new cell.

Geometry setup (grids, form-factor tables) is cached and reused when only
positions change, which is the common case during a relaxation; any cell
change triggers a full re-setup.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.dtypes import RDTYPE
from gradwave.postscf.forces import forces as hf_forces
from gradwave.scf.loop import scf, setup_system

_XC = {"lda": LDA_PW92, "pbe": PBE, "r2scan": R2SCAN}
# Spin-polarized (nspin=2) counterparts, keyed identically — same registry the
# api's _spin_setup uses, so the calculator's collinear path matches task: scf.
_SPIN_XC = {"lda": LSDA_PW92, "pbe": SpinPBE, "r2scan": SpinR2SCAN}


class GradWave(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "magmom"]

    def __init__(
        self,
        *,
        ecut: float,
        pseudopotentials: dict[str, str],  # element → UPF path
        xc: str = "pbe",
        ecutrho: float | None = None,  # density cutoff (USPP/PAW); default 4×ecut
        kpts=(1, 1, 1),
        kshift=(0, 0, 0),
        smearing: str = "none",
        width: float = 0.1,
        nbands: int | None = None,
        use_symmetry: bool = True,
        nspin: int = 1,  # 1 (restricted) or 2 (collinear); auto-bumps to 2 when
        # the atoms carry nonzero initial magnetic moments (see _resolve_spin)
        tot_magnetization: float | None = None,  # fix M=N↑−N↓ for a no-smearing
        # nspin=2 run; None → derive from the initial moments (smearing="none")
        # or let the shared Fermi level find it when a smearing is set
        max_iter: int = 100,
        etol: float = 1e-8,
        rhotol: float = 1e-7,
        diago_tol: float = 1e-9,
        mixing_scheme: str = "pulay",  # USPP/PAW path only (NC scf is Pulay)
        mixing_alpha: float = 0.7,
        mixing_history: int | None = None,  # None → solver's per-scheme default
        mixing_kerker=None,  # None → auto (on iff smeared)
        device: str = "cpu",
        compile_xc: bool = False,
        eigensolver: str = "davidson",  # davidson | chebyshev (NC path only)
        precond: str = "kerker",  # kerker | local_tf (NC and USPP/PAW paths)
        dispersion=None,  # opt-in D3(BJ): True/False, or a dict of overrides
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if nspin not in (1, 2):
            # collinear spin (nspin=2) threads through the norm-conserving SCF,
            # forces, and stress below; noncollinear/SOC has no calculator path
            raise ValueError(
                "GradWave supports nspin=1 (spin-restricted) and nspin=2 "
                "(collinear spin); noncollinear/spin-orbit has no calculator "
                "path yet (use task: magnetism via the api)")
        self.parameters.update(
            dict(ecut=ecut, ecutrho=ecutrho, xc=xc, kpts=tuple(kpts),
                 kshift=tuple(kshift), smearing=smearing, width=width,
                 nbands=nbands, use_symmetry=use_symmetry, nspin=nspin,
                 tot_magnetization=tot_magnetization,
                 max_iter=max_iter, etol=etol, rhotol=rhotol,
                 diago_tol=diago_tol, mixing_scheme=mixing_scheme,
                 mixing_alpha=mixing_alpha, mixing_history=mixing_history,
                 mixing_kerker=mixing_kerker, eigensolver=eigensolver,
                 precond=precond)
        )
        # Opt-in D3(BJ) dispersion, mirroring inputs.DispersionParams: True →
        # enabled with defaults (functional = the SCF xc); a dict overrides any
        # of functional/cutoff/cn_cutoff/s6/s8/a1/a2; None/False → off. Stored on
        # self.parameters so toggling it invalidates ASE's cached results.
        if dispersion in (None, False):
            self._dispersion = None
        elif dispersion is True:
            self._dispersion = {}
        elif isinstance(dispersion, dict):
            self._dispersion = dict(dispersion)
        else:
            raise ValueError(
                "dispersion must be True/False or a dict of D3(BJ)/D4(BJ) "
                "overrides (method, functional, cutoff, cn_cutoff, s6, s8, a1, "
                "a2; D4 also accepts charge)")
        if self._dispersion is not None:
            method = str(self._dispersion.get("method", "d3")).lower()
            if method not in ("d3", "d4"):
                raise ValueError(
                    f"dispersion method must be 'd3' or 'd4', got {method!r}")
        self.parameters["dispersion"] = self._dispersion
        self._pseudo_paths = dict(pseudopotentials)
        self._upf_cache: dict[str, object] = {}
        self._system = None
        self._system_key = None
        self._device = device
        self._compile_xc = compile_xc
        self._verbose = verbose
        self.last_result = None

    def _make_xc(self, nspin: int = 1):
        """Instantiate the XC functional, opting into the compiled real-valued
        energy_density path when compile_xc is set. nspin=2 selects the
        spin-polarized (LSDA / SpinPBE / SpinR2SCAN) variant so the collinear
        SCF, forces, and stress all see (ρ↑, ρ↓).
        The functional degrades to eager on any toolchain gap, so this is safe
        to leave on. It pays only for XC-heavy, CPU-bound work (PAW one-center
        loop, response HVPs, learned-XC training), not a plain FFT-bound SCF."""
        xc = (_SPIN_XC if nspin == 2 else _XC)[self.parameters["xc"]]()
        if self._compile_xc:
            xc.enable_compile()
        return xc

    def _resolve_spin(self, atoms, system):
        """Effective (nspin, start_mag, tot_magnetization) from ASE's initial
        magnetic moments. nspin=2 whenever the user asked for it (nspin=2) or an
        atom carries a nonzero initial moment; the per-atom μ_B moments become
        the polarization fractions scf's start_mag expects (m = μ / Z_valence),
        and their sum fixes the spin moment for a no-smearing (fixed-spin-moment)
        run — with a smearing the shared Fermi level finds the moment instead."""
        magmoms = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        if self.parameters["nspin"] != 2 and not np.any(magmoms != 0.0):
            return 1, None, None
        z_val = system.charges.detach().cpu().numpy().astype(float)
        start_mag = [m / z if z > 0 else 0.0
                     for m, z in zip(magmoms, z_val, strict=True)]
        tot_mag = self.parameters["tot_magnetization"]
        if tot_mag is None and self.parameters["smearing"] == "none":
            tot_mag = float(magmoms.sum())
        return 2, start_mag, None if tot_mag is None else float(tot_mag)

    def _upf(self, symbol):
        # shared loader (NC / USPP-PAW detection) lives in api; keep the
        # per-instance cache so a relaxation parses each pseudo once
        if symbol not in self._upf_cache:
            from gradwave.api import _load_upf

            self._upf_cache[symbol] = _load_upf(self._pseudo_paths[symbol])
        return self._upf_cache[symbol]

    def _is_uspp(self, species):
        from gradwave.api import _is_uspp

        return _is_uspp([self._upf(s) for s in species])

    def _apply_dispersion(self, system, properties):
        """Fold the opt-in D3(BJ)/D4(BJ) correction into the reported energy,
        forces, and stress — the calculator analog of ``api._apply_dispersion``.

        The energy is folded into ``res.energies.dispersion`` so ``free_energy``
        (which the calculator reports) carries it; the dispersion forces/stress
        are added on top of the SCF Hellmann–Feynman terms. Both are geometric,
        autograd contributions from ``postscf/dispersion.py`` (D3) or
        ``postscf/dispersion_d4.py`` (D4, charge-dependent EEQ C6 with a periodic
        Ewald charge model); no reimplementation here. ``method`` (``'d3'`` by
        default, or ``'d4'``) selects the model; D4 also reads a ``charge`` knob.
        Degrades to a no-op when the element set is uncovered or no BJ preset
        exists for the functional, matching the api's ``{'available': False}``."""
        if self._dispersion is None:
            return
        d = self._dispersion
        method = str(d.get("method", "d3")).lower()
        functional = d.get("functional") or self.parameters["xc"]
        positions = system.positions.detach().to(RDTYPE)
        cell = np.asarray(system.grid.cell, dtype=np.float64)
        z = [int(v) for v in self.atoms.get_atomic_numbers()]
        try:
            if method == "d4":
                from gradwave.postscf.dispersion_d4 import (
                    D4Config,
                    dispersion_energy,
                    dispersion_forces,
                    dispersion_stress,
                )
                cfg = D4Config.resolve(
                    functional, charge=float(d.get("charge", 0.0)),
                    cutoff_ang=d.get("cutoff", 21.2),
                    cn_cutoff_ang=d.get("cn_cutoff", 10.6),
                    s6=d.get("s6"), s8=d.get("s8"), a1=d.get("a1"), a2=d.get("a2"),
                )
            else:
                from gradwave.postscf.dispersion import (
                    D3Config,
                    dispersion_energy,
                    dispersion_forces,
                    dispersion_stress,
                )
                cfg = D3Config.resolve(
                    functional,
                    cutoff_ang=d.get("cutoff", 21.2),
                    cn_cutoff_ang=d.get("cn_cutoff", 10.6),
                    s6=d.get("s6"), s8=d.get("s8"), a1=d.get("a1"), a2=d.get("a2"),
                )
            cell_t = torch.as_tensor(cell, dtype=RDTYPE, device=positions.device)
            e = dispersion_energy(positions, cell_t, z, cfg)
            f = dispersion_forces(positions, cell, z, cfg)
            sig = (dispersion_stress(positions, cell, z, cfg)
                   if "stress" in properties else None)
        except (ValueError, NotImplementedError):
            return

        res = self.last_result
        res.energies.dispersion = (
            res.energies.dispersion + e.detach().to(positions.device))
        self.results["energy"] = float(res.energies.free_energy)
        self.results["free_energy"] = float(res.energies.free_energy)
        self.results["forces"] = self.results["forces"] + f.cpu().numpy()
        if sig is not None and "stress" in self.results:
            s = sig.cpu().numpy()
            self.results["stress"] = self.results["stress"] + np.array([
                s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

    def _get_system(self, atoms):
        symbols = atoms.get_chemical_symbols()
        species = sorted(set(symbols))
        key = (tuple(np.round(atoms.cell.array, 12).ravel()), tuple(symbols))
        if self._system is not None and key == self._system_key:
            return dataclasses.replace(
                self._system,
                positions=torch.as_tensor(atoms.get_positions(), dtype=RDTYPE).to(
                    self._system.positions.device),
            )
        system = setup_system(
            cell=atoms.cell.array,
            positions=atoms.get_positions(),
            species_of_atom=[species.index(s) for s in symbols],
            upfs=[self._upf(s) for s in species],
            ecut=self.parameters["ecut"],
            kmesh=self.parameters["kpts"],
            kshift=self.parameters["kshift"],
            nbands=self.parameters["nbands"],
            use_symmetry=self.parameters["use_symmetry"],
        ).to(self._device)
        self._system, self._system_key = system, key
        return system

    def _warm_start(self, system, nspin=1):
        """Seed the next solve from the previous converged state (same
        FFT grid — positions-only moves during a relaxation/MD qualify),
        with QE-style atomic extrapolation when the atoms moved: the
        superposition-of-atoms part of ρ travels with the atoms, the
        bonding remainder is reused. Plain reuse is nearly worthless
        under motion (measured: 8 vs 9 iterations for a 6 mÅ move —
        the seed error is first-order in displacement) while the
        extrapolated seed keeps the ~2-iteration warm restart.

        Restricted to nspin=1: the extrapolated seed below builds a single
        (total-density) channel, so a collinear-spin run cold-starts rather
        than reuse a mismatched seed."""
        from gradwave.scf.guess import sad_density

        prev = self.last_result
        if prev is None or nspin != 1 or getattr(prev, "nspin", 1) != 1:
            return None
        is_uspp = prev.formalism == "uspp"
        prev_sys = prev.system
        if tuple(prev_sys.grid.shape) != tuple(system.grid.shape):
            return None
        pos_new = system.positions
        pos_old = prev_sys.positions.to(pos_new.device)
        if float((pos_new - pos_old).abs().max()) < 1e-12:
            return prev
        tabs = prev_sys.paws if is_uspp else prev_sys.upfs
        soa = prev_sys.species_of_atom
        ne = prev_sys.n_electrons
        delta = (sad_density(system.grid, pos_new, soa, tabs, ne)
                 - sad_density(system.grid, pos_old, soa, tabs, ne))
        rho = prev.rho.detach() + delta
        if is_uspp:
            # becsum is per-atom and rides along as-is
            return dataclasses.replace(prev, rho=rho)
        return {"system": prev_sys, "nspin": 1, "rho": rho,
                "rho_spin": None, "coeffs": prev.coeffs}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p = self.parameters
        symbols = self.atoms.get_chemical_symbols()
        if self._is_uspp(sorted(set(symbols))):
            self._calculate_uspp(properties)
            return
        system = self._get_system(self.atoms)
        # collinear-spin channel from the atoms' initial moments (nspin=1 when
        # unmagnetized); start_mag/tot_magnetization seed and (optionally) pin M
        nspin, start_mag, tot_mag = self._resolve_spin(self.atoms, system)
        xc = self._make_xc(nspin)
        # NC scf takes an int mixing_history (None isn't accepted); omit it so
        # the solver's own default stands when the user left it unset
        mix_kw = ({} if p["mixing_history"] is None
                  else {"mixing_history": p["mixing_history"]})
        res = scf(
            system, xc,
            smearing=p["smearing"], width=p["width"],
            max_iter=p["max_iter"], etol=p["etol"], rhotol=p["rhotol"],
            mixing_alpha=p["mixing_alpha"], kerker=p["mixing_kerker"],
            diago_tol=p["diago_tol"], verbose=self._verbose,
            eigensolver=p["eigensolver"], precond=p["precond"],
            nspin=nspin, start_mag=start_mag, tot_magnetization=tot_mag,
            start_from=self._warm_start(system, nspin), **mix_kw,
        )
        if not res.converged:
            raise RuntimeError("gradwave SCF did not converge")
        self.last_result = res
        self.results["energy"] = float(res.energies.free_energy)  # consistent forces
        self.results["free_energy"] = float(res.energies.free_energy)
        if nspin == 2:
            # total spin moment M = N↑−N↓ (μ_B); cheap scalar off the result
            self.results["magmom"] = float(res.mag_total)
        # xc is used only when the system carries an NLCC core charge (the
        # core-correction force term, spin-resolved for nspin=2); ignored for
        # valence-only species. Reuse the (spin-matched) functional above.
        self.results["forces"] = hf_forces(res, xc=xc).cpu().numpy()
        if "stress" in properties:
            from gradwave.postscf.stress import stress as hf_stress

            sig = hf_stress(res, xc).cpu().numpy()
            # ASE Voigt order (xx, yy, zz, yz, xz, xy); ASE's convention is
            # +(1/Ω)∂E/∂ε, same as ours
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])
        self._apply_dispersion(system, properties)

    def _get_uspp_system(self, atoms):
        """With use_symmetry off, positions-only updates reuse the cached
        USPPSystem (its tables are phase-free; positions enter through
        structure factors built per solve). With use_symmetry on the density
        symmetrizer and the IBZ k-mesh are position-dependent, so the system
        is rebuilt every call — spglib then finds the current configuration's
        group (dropping to time-reversal-only when a move breaks it)."""
        from gradwave.scf.uspp import setup_uspp

        p = self.parameters
        symbols = atoms.get_chemical_symbols()
        species = sorted(set(symbols))
        key = (tuple(np.round(atoms.cell.array, 12).ravel()), tuple(symbols))
        if (not p["use_symmetry"] and self._system is not None
                and key == self._system_key):
            return dataclasses.replace(
                self._system,
                positions=torch.as_tensor(atoms.get_positions(), dtype=RDTYPE).to(
                    self._system.positions.device),
            )
        system = setup_uspp(
            atoms.cell.array, atoms.get_positions(),
            [species.index(s) for s in symbols],
            [self._upf(s) for s in species],
            ecut=p["ecut"], kmesh=p["kpts"], nbands=p["nbands"],
            ecutrho=p.get("ecutrho"), use_symmetry=p["use_symmetry"],
        ).to(self._device)
        self._system, self._system_key = system, key
        return system

    def _calculate_uspp(self, properties):
        """USPP/PAW route (nspin=1)."""
        from gradwave.scf.uspp import scf_uspp

        p = self.parameters
        if p["nspin"] == 2 or np.any(
                self.atoms.get_initial_magnetic_moments() != 0.0):
            # collinear spin through the calculator is norm-conserving only for
            # now; the USPP/PAW spin SCF + PAW spin forces/stress exist (api
            # task: scf) but are not wired here or covered by a same-commit
            # oracle, so keep this narrowly gated rather than silently wrong.
            raise NotImplementedError(
                "nspin=2 through the GradWave calculator is norm-conserving "
                "only; USPP/PAW collinear spin is not wired to the calculator "
                "yet (run task: scf via the api for a single-point nspin=2 "
                "USPP/PAW energy)")
        if p["eigensolver"] != "davidson":
            raise ValueError(
                "eigensolver='chebyshev' is norm-conserving only; the USPP/PAW "
                "generalized S-metric problem is not supported yet")
        system = self._get_uspp_system(self.atoms)
        # scf_uspp takes mixing_history=None natively (per-scheme default)
        res = scf_uspp(system, self._make_xc(), smearing=p["smearing"],
                       width=p["width"], max_iter=p["max_iter"],
                       etol=p["etol"], rhotol=p["rhotol"],
                       diago_tol=p["diago_tol"], mixing_scheme=p["mixing_scheme"],
                       mixing_alpha=p["mixing_alpha"],
                       mixing_history=p["mixing_history"],
                       mixing_kerker=p["mixing_kerker"], precond=p["precond"],
                       verbose=self._verbose,
                       start_from=self._warm_start(system))
        if not res.converged:
            raise RuntimeError("gradwave USPP SCF did not converge")
        self.last_result = res
        xc = self._make_xc()
        self.results["energy"] = float(res.energies.free_energy)
        self.results["free_energy"] = float(res.energies.free_energy)
        from gradwave.postscf.paw_forces import forces_uspp

        self.results["forces"] = forces_uspp(res, xc).cpu().numpy()
        if "stress" in properties:
            from gradwave.postscf.paw_stress import stress_uspp

            sig = stress_uspp(res, xc).cpu().numpy()
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])
        self._apply_dispersion(system, properties)
