"""UPF v2 pseudopotential parser (norm-conserving only).

Unit conversions happen HERE, at the boundary, once:

- PP_R, PP_RAB:      Bohr → Å
- PP_LOCAL:          Ry → eV
- PP_BETA (r·β(r)):  multiplied by BOHR_ANG**-0.5, so that the standard
                     Kleinman–Bylander contraction with PP_DIJ scaled by
                     RY_EV reproduces Quantum ESPRESSO's Ry formula in eV
                     with all radial integrals done in Å.  (β carries
                     Bohr^{-3/2}-like normalization; r·β therefore scales
                     by BOHR^{-1/2} when r moves to Å.)
- PP_DIJ:            multiplied by RY_EV (see above; only the combination
                     D_ij · ⟨β_i|ψ⟩⟨ψ|β_j⟩ is physical).
- PP_RHOATOM (4πr²ρ): Bohr⁻¹ → Å⁻¹ (divide by BOHR_ANG), so that
                     ∫ rhoatom dr = Z_val with dr in Å.

Nothing downstream of this module may touch Ry or Bohr.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gradwave.constants import BOHR_ANG, RY_EV


@dataclass(frozen=True)
class BetaProjector:
    l: int
    rbeta: np.ndarray  # r·β(r), converted (see module docstring), truncated at cutoff index
    cutoff_idx: int  # number of valid mesh points (kkbeta)
    j: float | None = None  # total angular momentum (fully-relativistic UPFs)


@dataclass(frozen=True)
class UPFData:
    element: str
    z_valence: float
    l_max: int
    functional: str
    core_correction: bool
    r: np.ndarray  # radial mesh [Å]
    rab: np.ndarray  # dr/di integration weights [Å]
    vloc: np.ndarray  # local potential on mesh [eV]
    betas: tuple[BetaProjector, ...]
    dij: np.ndarray  # (nproj, nproj) [eV-scaled, see module docstring]
    rhoatom: np.ndarray  # 4πr²ρ_atom(r) [Å⁻¹] — integrates to ~Z_val

    @property
    def n_proj(self) -> int:
        return len(self.betas)


def _parse_floats(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=np.float64)


def _read_root(path: Path) -> ET.Element:
    text = path.read_text()
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        # Some UPF generators emit stray '&' or '<'-adjacent junk inside
        # PP_INFO free text. Drop PP_INFO (metadata only) and retry.
        cleaned = re.sub(r"<PP_INFO>.*?</PP_INFO>", "<PP_INFO></PP_INFO>", text, flags=re.DOTALL)
        cleaned = cleaned.replace("&", "&amp;")
        return ET.fromstring(cleaned)


def parse_upf(path: str | Path) -> UPFData:
    path = Path(path)
    root = _read_root(path)
    if root.tag != "UPF":
        raise ValueError(f"{path}: not a UPF file (root tag {root.tag!r})")
    version = root.attrib.get("version", "")
    if not version.startswith("2."):
        raise ValueError(f"{path}: only UPF v2 is supported, got version {version!r}")

    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"{path}: missing PP_HEADER")
    h = header.attrib

    def flag(name: str) -> bool:
        return h.get(name, "F").strip().upper() in ("T", "TRUE")

    pseudo_type = h.get("pseudo_type", "").strip()
    if pseudo_type != "NC" or flag("is_ultrasoft") or flag("is_paw"):
        raise ValueError(
            f"{path}: gradwave supports norm-conserving pseudopotentials only "
            f"(got pseudo_type={pseudo_type!r}). Use PseudoDojo or SG15 ONCV UPF files."
        )
    has_so = flag("has_so")

    mesh = root.find("PP_MESH")
    r = _parse_floats(mesh.find("PP_R").text) * BOHR_ANG
    rab = _parse_floats(mesh.find("PP_RAB").text) * BOHR_ANG

    vloc = _parse_floats(root.find("PP_LOCAL").text) * RY_EV

    nonlocal_ = root.find("PP_NONLOCAL")
    jjj = {}
    if has_so:
        so = root.find("PP_SPIN_ORB")
        if so is None:
            raise ValueError(f"{path}: has_so but no PP_SPIN_ORB block")
        for child in so:
            if child.tag.startswith("PP_RELBETA"):
                jjj[int(child.attrib["index"])] = float(child.attrib["jjj"])
    betas = []
    for child in sorted(
        (c for c in nonlocal_ if c.tag.startswith("PP_BETA.")),
        key=lambda c: int(c.tag.split(".")[1]),
    ):
        l = int(child.attrib["angular_momentum"])
        kkbeta = int(child.attrib.get("cutoff_radius_index", len(r)))
        vals = _parse_floats(child.text) * BOHR_ANG ** (-0.5)
        idx = int(child.tag.split(".")[1])  # index attr can be "*" in SG15 FR
        # Respect the hard truncation: SG15 β are exactly zero-padded/noisy
        # beyond kkbeta; integrating past it adds noise.
        betas.append(BetaProjector(l=l, rbeta=vals[:kkbeta], cutoff_idx=kkbeta,
                                   j=jjj.get(idx)))

    nproj = len(betas)
    dij = np.zeros((nproj, nproj))
    if nproj:
        dij = _parse_floats(nonlocal_.find("PP_DIJ").text).reshape(nproj, nproj) * RY_EV

    rhoatom = _parse_floats(root.find("PP_RHOATOM").text) / BOHR_ANG

    if flag("core_correction"):
        raise ValueError(
            f"{path}: NLCC (core_correction) UPF files are not supported yet — "
            "pick a pseudopotential without nonlinear core correction."
        )

    n = len(r)
    for name, arr in (("PP_RAB", rab), ("PP_LOCAL", vloc), ("PP_RHOATOM", rhoatom)):
        if len(arr) != n:
            raise ValueError(f"{path}: {name} length {len(arr)} != mesh size {n}")

    return UPFData(
        element=h["element"].strip(),
        z_valence=float(h["z_valence"]),
        l_max=int(h["l_max"]),
        functional=h.get("functional", "").strip(),
        core_correction=False,
        r=r,
        rab=rab,
        vloc=vloc,
        betas=tuple(betas),
        dij=dij,
        rhoatom=rhoatom,
    )
