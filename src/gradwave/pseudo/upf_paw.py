"""UPF v2 parser for PAW (and ultrasoft) datasets.

PAW = ultrasoft plane-wave machinery + one-center corrections. This module
parses everything both stages need:

ultrasoft (stage 1):
- PP_BETA/PP_DIJ as in NC (same unit conventions as upf.py)
- PP_AUGMENTATION/PP_Q: q_ij = ∫ Q_ij(r) d³r (the S-operator/charge weights)
- PP_QIJL.i.j.l: radial augmentation functions q^l_ij(r), r² included, so
  Q_ij(G) form factors come from the same spherical Bessel transform as
  everything else. Only q_with_l="true", nqf=0 datasets are supported
  (all psl 1.0 PAW files; old RRKJ USPPs with polynomial interior refits
  are rejected).

one-center PAW (stage 2):
- PP_FULL_WFC: AE and PS partial waves r·φ_i(r)
- PP_PAW: occupations, AE core density, AE local potential, core energy

Unit conventions match upf.py exactly (everything eV/Å at parse time; r·φ and
r·β carry BOHR^{-1/2}; densities e/Å³; q^l_ij like PP_RHOATOM scale 1/BOHR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gradwave.constants import BOHR_ANG, RY_EV
from gradwave.pseudo.upf import (
    BetaProjector,
    _parse_floats,
    _qe_msh,
    _read_root,
)


@dataclass(frozen=True)
class PartialWave:
    l: int
    label: str
    rphi: np.ndarray  # r·φ(r) [Å^{-1/2}], full mesh


@dataclass(frozen=True)
class PAWData:
    element: str
    z_valence: float
    l_max: int
    l_max_aug: int  # angular momenta of the augmentation (2·l_max typically)
    is_paw: bool  # False → bare ultrasoft (no one-center data)
    r: np.ndarray  # [Å]
    rab: np.ndarray  # [Å]
    msh: int  # QE 10-bohr local-channel truncation (see upf.py)
    vloc: np.ndarray  # [eV]
    betas: tuple[BetaProjector, ...]
    dij: np.ndarray  # (nproj, nproj) [eV]
    q: np.ndarray  # (nproj, nproj) augmentation charges ∫Q_ij d³r [e]
    qijl: dict  # (i, j, l) → radial q^l_ij(r) [Å⁻¹], truncated at aug cutoff
    aug_cutoff_idx: int
    rhoatom: np.ndarray  # 4πr²ρ_atom [Å⁻¹]
    core_rho: np.ndarray | None  # NLCC ρ̃_core(r) [e/Å³] (smooth part)
    rho_cutoff: float  # suggested ecutrho [eV]
    wfc_cutoff: float  # suggested ecutwfc [eV]
    # one-center (PAW only; empty/None for bare ultrasoft)
    aewfc: tuple[PartialWave, ...] = ()
    pswfc: tuple[PartialWave, ...] = ()
    paw_occ: np.ndarray | None = None  # (nproj,) reference occupations
    ae_core_rho: np.ndarray | None = None  # AE ρ_core(r) [e/Å³]
    ae_vloc: np.ndarray | None = None  # AE local (Hartree-screened) potential [eV]
    core_energy: float = 0.0  # frozen-core energy [eV]

    @property
    def n_proj(self) -> int:
        return len(self.betas)


def parse_upf_paw(path) -> PAWData:
    from pathlib import Path

    path = Path(path)
    root = _read_root(path)
    h = root.find("PP_HEADER").attrib

    def flag(name: str) -> bool:
        return h.get(name, "F").strip().upper() in ("T", "TRUE")

    is_paw = flag("is_paw")
    if not flag("is_ultrasoft") and not is_paw:
        raise ValueError(f"{path}: not an ultrasoft/PAW dataset — use pseudo/upf.py")
    if flag("has_so"):
        raise ValueError(f"{path}: fully-relativistic PAW not supported")

    mesh = root.find("PP_MESH")
    r = _parse_floats(mesh.find("PP_R").text) * BOHR_ANG
    rab = _parse_floats(mesh.find("PP_RAB").text) * BOHR_ANG

    vloc = _parse_floats(root.find("PP_LOCAL").text) * RY_EV

    nonlocal_ = root.find("PP_NONLOCAL")
    betas = []
    for child in sorted(
        (c for c in nonlocal_ if c.tag.startswith("PP_BETA.")),
        key=lambda c: int(c.tag.split(".")[1]),
    ):
        l = int(child.attrib["angular_momentum"])
        kkbeta = int(child.attrib.get("cutoff_radius_index", len(r)))
        vals = _parse_floats(child.text) * BOHR_ANG ** (-0.5)
        betas.append(BetaProjector(l=l, rbeta=vals[:kkbeta], cutoff_idx=kkbeta))
    nproj = len(betas)
    dij = _parse_floats(nonlocal_.find("PP_DIJ").text).reshape(nproj, nproj) * RY_EV

    aug = nonlocal_.find("PP_AUGMENTATION")
    a = aug.attrib
    if a.get("q_with_l", "F").strip().upper() not in ("T", "TRUE"):
        raise ValueError(f"{path}: only q_with_l datasets supported (psl PAW)")
    if int(a.get("nqf", "0")) != 0:
        raise ValueError(f"{path}: polynomial augmentation refit (nqf>0) not supported")
    # bare USPP files carry nqlc = l_max_aug + 1 instead of l_max_aug
    if "l_max_aug" in a:
        l_max_aug = int(float(a["l_max_aug"]))
    else:
        l_max_aug = int(a["nqlc"]) - 1
    cutoff_idx = int(a.get("cutoff_r_index", len(r)))
    q = _parse_floats(aug.find("PP_Q").text).reshape(nproj, nproj)
    qijl = {}
    for child in aug:
        if not child.tag.startswith("PP_QIJL."):
            continue
        _, i, j, l = child.tag.split(".")
        vals = _parse_floats(child.text) / BOHR_ANG  # like PP_RHOATOM
        key = (int(i) - 1, int(j) - 1, int(l))
        qijl[key] = vals[:cutoff_idx]

    rhoatom = _parse_floats(root.find("PP_RHOATOM").text) / BOHR_ANG
    core_rho = None
    nlcc = root.find("PP_NLCC")
    if nlcc is not None:
        core_rho = _parse_floats(nlcc.text) / BOHR_ANG**3

    aewfc, pswfc = [], []
    paw_occ = ae_core = ae_vloc = None
    core_energy = 0.0
    if is_paw:
        full = root.find("PP_FULL_WFC")
        for child in full:
            pw = PartialWave(
                l=int(child.attrib["l"]),
                label=child.attrib.get("label", "").strip(),
                rphi=_parse_floats(child.text) * BOHR_ANG ** (-0.5),
            )
            (aewfc if child.tag.startswith("PP_AEWFC") else pswfc).append(pw)
        paw = root.find("PP_PAW")
        core_energy = float(paw.attrib.get("core_energy", "0")) * RY_EV
        paw_occ = _parse_floats(paw.find("PP_OCCUPATIONS").text)
        ae_core = _parse_floats(paw.find("PP_AE_NLCC").text) / BOHR_ANG**3
        ae_vloc = _parse_floats(paw.find("PP_AE_VLOC").text) * RY_EV

    return PAWData(
        element=h["element"].strip(),
        z_valence=float(h["z_valence"]),
        l_max=int(h["l_max"]),
        l_max_aug=l_max_aug,
        is_paw=is_paw,
        r=r,
        rab=rab,
        msh=_qe_msh(r),
        vloc=vloc,
        betas=tuple(betas),
        dij=dij,
        q=q,
        qijl=qijl,
        aug_cutoff_idx=cutoff_idx,
        rhoatom=rhoatom,
        core_rho=core_rho,
        rho_cutoff=float(h.get("rho_cutoff", "0")) * RY_EV,
        wfc_cutoff=float(h.get("wfc_cutoff", "0")) * RY_EV,
        aewfc=tuple(aewfc),
        pswfc=tuple(pswfc),
        paw_occ=paw_occ,
        ae_core_rho=ae_core,
        ae_vloc=ae_vloc,
        core_energy=core_energy,
    )
