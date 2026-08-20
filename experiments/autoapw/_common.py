"""Shared fixtures for the TiO2 fullpot experiments: the rutile geometry, the
env-overridable ``crystal_scf_multi`` config dict, and EFG eigenvalue sorting.

Every probe in this campaign runs the same rutile TiO2 cell and builds the same
fullpot config dict, differing only in its env-var prefix (NP_/MA_/KA_/TK_) and
defaults — that boilerplate lives here exactly once.
"""
import os

import numpy as np

# Rutile TiO2 (P4_2/mnm), experimental geometry. Internal coordinate u.
U = 0.3048
A_BOHR = [8.68083, 8.68083, 5.59096]
ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((U, U, 0.0), "O"), ((1 - U, 1 - U, 0.0), "O"),
         ((0.5 + U, 0.5 - U, 0.5), "O"), ((0.5 - U, 0.5 + U, 0.5), "O")]
RADII = {"Ti": 1.098, "O": 0.824}


def env_int(prefix: str, name: str, default: int) -> int:
    """int(os.environ["<PREFIX>_<NAME>"]) with a default."""
    return int(os.environ.get(f"{prefix}_{name}", str(default)))


def env_float(prefix: str, name: str, default: float) -> float:
    """float(os.environ["<PREFIX>_<NAME>"]) with a default."""
    return float(os.environ.get(f"{prefix}_{name}", str(default)))


def fullpot_cfg(prefix: str, *, ecut: float = 150.0, lmax: int = 2, k: int = 1,
                fullpot_lmax: int = 2, kworkers: int = 1) -> dict:
    """The campaign's shared fullpot ``crystal_scf_multi`` kwargs.

    Each knob is env-overridable as ``<prefix>_{ECUT,LMAX,K,FPLMAX,KWORKERS}``;
    the fixed entries (no smearing, symmetry on, exact solves only) are the
    determinism-preserving choices every probe in this campaign shares.
    """
    return dict(ecut=env_float(prefix, "ECUT", ecut),
                lmax=env_int(prefix, "LMAX", lmax),
                kmesh=(env_int(prefix, "K", k),) * 3,
                smearing=0.0, efg=False, fullpot=True,
                fullpot_lmax=env_int(prefix, "FPLMAX", fullpot_lmax),
                use_symmetry=True,
                kworkers=env_int(prefix, "KWORKERS", kworkers),
                subspace_reuse=False)


def efg_eigs(tensor) -> np.ndarray:
    """EFG principal values sorted by descending |V_ii| (the NQR convention:
    V_zz first)."""
    w = np.linalg.eigvalsh(tensor)
    return w[np.argsort(-np.abs(w))]
