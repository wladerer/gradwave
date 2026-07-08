"""ASE Calculator interface (Layer C).

Fixed cell only: no stress, no variable-cell relaxation (do not wrap in
ExpCellFilter) — the radial→G tables are not differentiable in the cell.
Geometry setup (grids, form-factor tables) is cached and reused when only
positions change, which is the common case during a relaxation.
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
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

_XC = {"lda": LDA_PW92, "pbe": PBE}


class GradWave(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        ecut: float,
        pseudopotentials: dict[str, str],  # element → UPF path
        xc: str = "pbe",
        kpts=(1, 1, 1),
        kshift=(0, 0, 0),
        smearing: str = "none",
        width: float = 0.1,
        nbands: int | None = None,
        use_symmetry: bool = True,
        etol: float = 1e-8,
        rhotol: float = 1e-7,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parameters.update(
            dict(ecut=ecut, xc=xc, kpts=tuple(kpts), kshift=tuple(kshift),
                 smearing=smearing, width=width, nbands=nbands,
                 use_symmetry=use_symmetry, etol=etol, rhotol=rhotol)
        )
        self._pseudo_paths = dict(pseudopotentials)
        self._upf_cache: dict[str, object] = {}
        self._system = None
        self._system_key = None
        self._verbose = verbose
        self.last_result = None

    def _upf(self, symbol):
        if symbol not in self._upf_cache:
            self._upf_cache[symbol] = parse_upf(self._pseudo_paths[symbol])
        return self._upf_cache[symbol]

    def _get_system(self, atoms):
        symbols = atoms.get_chemical_symbols()
        species = sorted(set(symbols))
        key = (tuple(np.round(atoms.cell.array, 12).ravel()), tuple(symbols))
        if self._system is not None and key == self._system_key:
            return dataclasses.replace(
                self._system,
                positions=torch.as_tensor(atoms.get_positions(), dtype=RDTYPE),
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
        )
        self._system, self._system_key = system, key
        return system

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p = self.parameters
        system = self._get_system(self.atoms)
        res = scf(
            system, _XC[p["xc"]](),
            smearing=p["smearing"], width=p["width"],
            etol=p["etol"], rhotol=p["rhotol"], verbose=self._verbose,
        )
        if not res.converged:
            raise RuntimeError("gradwave SCF did not converge")
        self.last_result = res
        self.results["energy"] = float(res.energies.free_energy)  # consistent forces
        self.results["free_energy"] = float(res.energies.free_energy)
        self.results["forces"] = hf_forces(res).cpu().numpy()
