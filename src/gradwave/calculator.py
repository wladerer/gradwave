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
from gradwave.dtypes import RDTYPE
from gradwave.postscf.forces import forces as hf_forces
from gradwave.scf.loop import scf, setup_system

_XC = {"lda": LDA_PW92, "pbe": PBE}


class GradWave(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

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
        nspin: int = 1,  # 1 only: nspin=2 forces are not implemented (see below)
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
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if nspin != 1:
            # forces/stress (the reason this calculator exists — relaxation)
            # have no collinear-spin implementation yet, so honoring nspin=2
            # here would silently fall over inside hf_forces
            raise ValueError(
                "GradWave calculator supports nspin=1 only; collinear-spin "
                "forces are not implemented (run task: scf via the api for a "
                "single-point nspin=2 energy)")
        self.parameters.update(
            dict(ecut=ecut, ecutrho=ecutrho, xc=xc, kpts=tuple(kpts),
                 kshift=tuple(kshift), smearing=smearing, width=width,
                 nbands=nbands, use_symmetry=use_symmetry, nspin=nspin,
                 max_iter=max_iter, etol=etol, rhotol=rhotol,
                 diago_tol=diago_tol, mixing_scheme=mixing_scheme,
                 mixing_alpha=mixing_alpha, mixing_history=mixing_history,
                 mixing_kerker=mixing_kerker, eigensolver=eigensolver,
                 precond=precond)
        )
        self._pseudo_paths = dict(pseudopotentials)
        self._upf_cache: dict[str, object] = {}
        self._system = None
        self._system_key = None
        self._device = device
        self._compile_xc = compile_xc
        self._verbose = verbose
        self.last_result = None

    def _make_xc(self):
        """Instantiate the XC functional, opting into the compiled real-valued
        energy_density path when compile_xc is set.
        The functional degrades to eager on any toolchain gap, so this is safe
        to leave on. It pays only for XC-heavy, CPU-bound work (PAW one-center
        loop, response HVPs, learned-XC training), not a plain FFT-bound SCF."""
        xc = _XC[self.parameters["xc"]]()
        if self._compile_xc:
            xc.enable_compile()
        return xc

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

    def _warm_start(self, system):
        """Seed the next solve from the previous converged state (same
        FFT grid — positions-only moves during a relaxation/MD qualify),
        with QE-style atomic extrapolation when the atoms moved: the
        superposition-of-atoms part of ρ travels with the atoms, the
        bonding remainder is reused. Plain reuse is nearly worthless
        under motion (measured: 8 vs 9 iterations for a 6 mÅ move —
        the seed error is first-order in displacement) while the
        extrapolated seed keeps the ~2-iteration warm restart."""
        from gradwave.scf.guess import sad_density

        prev = self.last_result
        if prev is None:
            return None
        is_dict = isinstance(prev, dict)
        prev_sys = prev["system"] if is_dict else prev.system
        if tuple(prev_sys.grid.shape) != tuple(system.grid.shape):
            return None
        pos_new = system.positions
        pos_old = prev_sys.positions.to(pos_new.device)
        if float((pos_new - pos_old).abs().max()) < 1e-12:
            return prev
        tabs = prev_sys.paws if is_dict else prev_sys.upfs
        soa = prev_sys.species_of_atom
        ne = prev_sys.n_electrons
        delta = (sad_density(system.grid, pos_new, soa, tabs, ne)
                 - sad_density(system.grid, pos_old, soa, tabs, ne))
        rho = (prev["rho"] if is_dict else prev.rho).detach() + delta
        if is_dict:
            out = dict(prev)
            out["rho"] = rho  # becsum is per-atom and rides along as-is
            return out
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
        # NC scf takes an int mixing_history (None isn't accepted); omit it so
        # the solver's own default stands when the user left it unset
        mix_kw = ({} if p["mixing_history"] is None
                  else {"mixing_history": p["mixing_history"]})
        res = scf(
            system, self._make_xc(),
            smearing=p["smearing"], width=p["width"],
            max_iter=p["max_iter"], etol=p["etol"], rhotol=p["rhotol"],
            mixing_alpha=p["mixing_alpha"], kerker=p["mixing_kerker"],
            diago_tol=p["diago_tol"], verbose=self._verbose,
            eigensolver=p["eigensolver"], precond=p["precond"],
            start_from=self._warm_start(system), **mix_kw,
        )
        if not res.converged:
            raise RuntimeError("gradwave SCF did not converge")
        self.last_result = res
        self.results["energy"] = float(res.energies.free_energy)  # consistent forces
        self.results["free_energy"] = float(res.energies.free_energy)
        self.results["forces"] = hf_forces(res).cpu().numpy()
        if "stress" in properties:
            from gradwave.postscf.stress import stress as hf_stress

            sig = hf_stress(res, self._make_xc()).cpu().numpy()
            # ASE Voigt order (xx, yy, zz, yz, xz, xy); ASE's convention is
            # +(1/Ω)∂E/∂ε, same as ours
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])

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
        if not res["converged"]:
            raise RuntimeError("gradwave USPP SCF did not converge")
        self.last_result = res
        xc = self._make_xc()
        self.results["energy"] = float(res["energies"].free_energy)
        self.results["free_energy"] = float(res["energies"].free_energy)
        from gradwave.postscf.paw_forces import forces_uspp

        self.results["forces"] = forces_uspp(res, xc).cpu().numpy()
        if "stress" in properties:
            from gradwave.postscf.paw_stress import stress_uspp

            sig = stress_uspp(res, xc).cpu().numpy()
            self.results["stress"] = np.array([
                sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1],
            ])
