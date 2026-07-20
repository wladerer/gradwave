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
class AtomicOrbital:
    """A pseudo-atomic wavefunction from PP_PSWFC — the localized orbital used
    as the DFT+U projector (and for LCAO/PDOS). Stored r·R_nl(r) scaled by
    BOHR^{-1/2} exactly like r·β, so the radial form factor reuses the same
    spherical Bessel transform. Normalized ∫(r·R)² dr = 1 with dr in Å."""

    l: int
    label: str  # e.g. "3D", "4S"
    occupation: float  # reference atomic occupation of this orbital
    rchi: np.ndarray  # r·R_nl(r) [Å^{-1/2}], full mesh
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
    core_rho: np.ndarray | None = None  # NLCC ρ_core(r) [e/Å³] (added to XC only)
    pswfc: tuple[AtomicOrbital, ...] = ()  # PP_PSWFC atomic orbitals (empty if none)
    # QE truncates LOCAL-channel radial integrals (v_loc, alpha-Z, ρ_core,
    # ρ_atom) at r = 10 bohr with an odd point count (readpp.f90's msh) — for
    # meshes reaching past 10 bohr the UPF tail deviates from −Z/r by ~1e-5 eV
    # and integrating it shifts v_loc(G=0) by ~0.1 eV·Å³ (a rigid ~8 meV
    # eigenvalue offset for Ni). 0 means "unset" (synthetic data): full mesh.
    msh: int = 0

    @property
    def n_proj(self) -> int:
        return len(self.betas)

    def hubbard_orbitals(self, l: int) -> list[AtomicOrbital]:
        """PP_PSWFC orbitals with angular momentum `l` (the +U manifold).
        Returns j-split channels for fully-relativistic pseudos; the caller
        combines or picks per its projection scheme."""
        return [w for w in self.pswfc if w.l == l]


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
        # Escape only bare '&' — leave already-valid entities (&amp; &lt; &gt;
        # &quot; &apos; &#nn;) untouched so we don't double-escape.
        cleaned = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", cleaned)
        return ET.fromstring(cleaned)


def _validate_root(root: ET.Element, path: Path) -> None:
    """Up-front UPF sanity checks shared by parse_upf and parse_upf_paw:
    root tag is UPF and the version is v2."""
    if root.tag != "UPF":
        raise ValueError(f"{path}: not a UPF file (root tag {root.tag!r})")
    version = root.attrib.get("version", "")
    if not version.startswith("2."):
        raise ValueError(f"{path}: only UPF v2 is supported, got version {version!r}")


def _check_mesh_lengths(path: Path, n: int, arrays: dict[str, np.ndarray]) -> None:
    """Assert each named radial array matches the PP_MESH point count n."""
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"{path}: {name} length {len(arr)} != mesh size {n}")


def _header_flag(h: dict, name: str) -> bool:
    """UPF boolean header attribute: 'T'/'TRUE' (case/space-insensitive) → True,
    everything else (including a missing attribute) → False. Shared by the NC
    and PAW/USPP header checks."""
    return h.get(name, "F").strip().upper() in ("T", "TRUE")


def _parse_mesh_vloc(root: ET.Element) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, rab, vloc) from PP_MESH/PP_LOCAL: r, rab Bohr→Å and vloc Ry→eV.
    Identical for the NC and PAW/USPP parsers."""
    mesh = root.find("PP_MESH")
    r = _parse_floats(mesh.find("PP_R").text) * BOHR_ANG
    rab = _parse_floats(mesh.find("PP_RAB").text) * BOHR_ANG
    vloc = _parse_floats(root.find("PP_LOCAL").text) * RY_EV
    return r, rab, vloc


def _parse_betas(
    nonlocal_: ET.Element, n_r: int, jjj: dict | None = None
) -> list[BetaProjector]:
    """PP_BETA.i projectors in index order: r·β scaled by BOHR^{-1/2} and hard-
    truncated at cutoff_radius_index (SG15 β are noisy/zero-padded past it).
    `jjj` maps the 1-based beta index to its total j for fully-relativistic
    UPFs; None/empty leaves j unset (the ultrasoft/PAW and scalar-relativistic
    NC cases)."""
    jjj = jjj or {}
    betas = []
    for child in sorted(
        (c for c in nonlocal_ if c.tag.startswith("PP_BETA.")),
        key=lambda c: int(c.tag.split(".")[1]),
    ):
        l = int(child.attrib["angular_momentum"])
        kkbeta = int(child.attrib.get("cutoff_radius_index", n_r))
        vals = _parse_floats(child.text) * BOHR_ANG ** (-0.5)
        idx = int(child.tag.split(".")[1])  # index attr can be "*" in SG15 FR
        betas.append(BetaProjector(l=l, rbeta=vals[:kkbeta], cutoff_idx=kkbeta,
                                   j=jjj.get(idx)))
    return betas


def _parse_pswfc_chi(root: ET.Element, jchi: dict | None = None) -> list[AtomicOrbital]:
    """PP_PSWFC PP_CHI.i atomic orbitals (the +U/LCAO manifold), r·R_nl scaled
    by BOHR^{-1/2} so ∫(r·R)² dr = 1 with dr in Å and the r·β SBT form factor is
    reused. Empty when the dataset carries no PP_PSWFC block. `jchi` maps the
    1-based chi index to its total j (fully-relativistic UPFs)."""
    jchi = jchi or {}
    chi = []
    pswfc_block = root.find("PP_PSWFC")
    if pswfc_block is not None:
        for child in sorted(
            (c for c in pswfc_block if c.tag.startswith("PP_CHI.")),
            key=lambda c: int(c.tag.split(".")[1]),
        ):
            idx = int(child.tag.split(".")[1])
            chi.append(AtomicOrbital(
                l=int(child.attrib["l"]),
                label=child.attrib.get("label", "").strip(),
                occupation=float(child.attrib.get("occupation", "0")),
                rchi=_parse_floats(child.text) * BOHR_ANG ** (-0.5),
                j=jchi.get(idx),
            ))
    return chi


def parse_upf(path: str | Path) -> UPFData:
    path = Path(path)
    root = _read_root(path)
    _validate_root(root, path)

    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"{path}: missing PP_HEADER")
    h = header.attrib

    pseudo_type = h.get("pseudo_type", "").strip()
    if pseudo_type != "NC" or _header_flag(h, "is_ultrasoft") or _header_flag(h, "is_paw"):
        raise ValueError(
            f"{path}: gradwave supports norm-conserving pseudopotentials only "
            f"(got pseudo_type={pseudo_type!r}). Use PseudoDojo or SG15 ONCV UPF files."
        )
    has_so = _header_flag(h, "has_so")

    r, rab, vloc = _parse_mesh_vloc(root)

    nonlocal_ = root.find("PP_NONLOCAL")
    # Fully-relativistic UPFs carry PP_RELBETA (β total-j) and PP_RELWFC (χ
    # total-j) side by side in one PP_SPIN_ORB block — read both in one pass.
    jjj = {}
    jchi = {}
    if has_so:
        so = root.find("PP_SPIN_ORB")
        if so is None:
            raise ValueError(f"{path}: has_so but no PP_SPIN_ORB block")
        for child in so:
            if child.tag.startswith("PP_RELBETA"):
                jjj[int(child.attrib["index"])] = float(child.attrib["jjj"])
            elif child.tag.startswith("PP_RELWFC"):
                jchi[int(child.attrib["index"])] = float(child.attrib["jchi"])
    betas = _parse_betas(nonlocal_, len(r), jjj)

    nproj = len(betas)
    dij = np.zeros((nproj, nproj))
    if nproj:
        dij = _parse_floats(nonlocal_.find("PP_DIJ").text).reshape(nproj, nproj) * RY_EV

    rhoatom = _parse_floats(root.find("PP_RHOATOM").text) / BOHR_ANG

    # PP_PSWFC atomic orbitals (present in PseudoDojo, absent/empty in SG15).
    pswfc = _parse_pswfc_chi(root, jchi)

    core_rho = None
    if _header_flag(h, "core_correction"):
        # PP_NLCC stores ρ_core(r) directly (NOT 4πr²ρ), in bohr⁻³
        core_rho = _parse_floats(root.find("PP_NLCC").text) / BOHR_ANG**3

    _check_mesh_lengths(path, len(r), {"PP_RAB": rab, "PP_LOCAL": vloc, "PP_RHOATOM": rhoatom})

    return UPFData(
        element=h["element"].strip(),
        z_valence=float(h["z_valence"]),
        l_max=int(h["l_max"]),
        functional=h.get("functional", "").strip(),
        core_correction=core_rho is not None,
        r=r,
        rab=rab,
        vloc=vloc,
        betas=tuple(betas),
        dij=dij,
        rhoatom=rhoatom,
        core_rho=core_rho,
        pswfc=tuple(pswfc),
        msh=_qe_msh(r),
    )


def _qe_msh(r_ang: np.ndarray) -> int:
    """QE's readpp.f90 rule: 1-based index of the first mesh point beyond
    10 bohr (or the mesh size), rounded DOWN to odd for Simpson."""
    rmax = 10.0 * BOHR_ANG
    n_le = int(np.searchsorted(r_ang, rmax, side="right"))
    ir = n_le + 1 if n_le < len(r_ang) else len(r_ang)
    return 2 * ((ir + 1) // 2) - 1
