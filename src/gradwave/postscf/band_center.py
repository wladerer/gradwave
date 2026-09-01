"""d-band center and band width — moments of a projected DOS (Hammer-Nørskov).

The d-band model of transition-metal catalysis summarizes the coupling strength
of an adsorbate to a metal surface by two moments of the metal's d-projected DOS
ρ_d(ε), referenced to the Fermi level E_F:

    ε_d   = ∫ (ε − E_F) ρ_d(ε) dε / ∫ ρ_d(ε) dε          (1st moment, "center")
    W_d   = sqrt( ∫ (ε − <ε>)² ρ_d(ε) dε / ∫ ρ_d(ε) dε ) (2nd central moment, width)

An upshifted ε_d (closer to E_F) empties more antibonding adsorbate states above
E_F and binds intermediates more strongly — the central activity descriptor of
Hammer & Nørskov (Nature 376, 238, 1995; Surf. Sci. 343, 211, 1995). The width
W_d complements it (a broader band lowers the local DOS at E_F for the same
filling).

Two entry points:

* ``band_moments`` is the differentiable primitive: it takes per-sample energies
  and non-negative weights as **torch tensors** and returns the moments as 0-d
  tensors, so a moment computed from projection weights that still carry an
  autograd graph (before any numpy hop) is differentiable — e.g. dε_d/dstrain or
  dε_d/dU for a strain- or potential-tunable descriptor. The weights are whatever
  makes ∫·dε a weighted sum: either per-state Löwdin populations times k-weights
  (the exact discrete moment) or ρ_d(ε) sampled on a uniform energy grid (the
  broadened moment; the grid spacing cancels in the ratio).

* ``band_center`` is the convenience over an already-computed
  :class:`~gradwave.postscf.pdos.ProjectedDOS`: it selects the requested angular
  channel (and atoms), references to E_F, and returns the requested moment. This
  path runs through the numpy DOS grid, so it is a plain float (use
  ``band_moments`` directly for the differentiable route).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from gradwave.dtypes import RDTYPE

if TYPE_CHECKING:
    from gradwave.postscf.pdos import ProjectedDOS

_L_LETTER = {0: "S", 1: "P", 2: "D", 3: "F"}


@dataclass(frozen=True)
class BandMoments:
    """Moments of a (projected) DOS. The scalars are 0-d torch tensors so a
    moment built from grad-carrying weights stays differentiable; call ``.item()``
    for a python float. ``center`` is referenced (ε − ref); ``abs_center`` is the
    unreferenced mean; ``width`` = sqrt(``variance``) is reference-independent."""

    center: torch.Tensor        # 1st moment about `ref` (eV) — the d-band center
    abs_center: torch.Tensor    # 1st moment about 0 (eV) — the raw mean energy
    width: torch.Tensor         # sqrt(2nd central moment) (eV) — the band width
    variance: torch.Tensor      # 2nd central moment (eV²)
    weight: torch.Tensor        # total ∫ρ dε (the DOS area / population norm)


def band_moments(
    energies: torch.Tensor, weights: torch.Tensor, *, ref: float | torch.Tensor = 0.0,
) -> BandMoments:
    """First and second moments of a DOS given per-sample ``energies`` and
    non-negative ``weights`` (torch tensors, same shape). Fully differentiable in
    both inputs and in ``ref``.

    ``weights`` may be per-state populations (the exact discrete moment) or
    ρ(ε) sampled on a **uniform** energy grid (the grid spacing cancels in every
    ratio). The returned ``center`` is referenced by ``ref`` (pass the Fermi level
    for a d-band center); ``width``/``variance`` are reference-independent.
    """
    e = energies.to(RDTYPE)
    w = weights.to(RDTYPE)
    if e.shape != w.shape:
        raise ValueError("energies and weights must have the same shape")
    total = w.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError(
            "total weight is non-positive — the selected projection is empty "
            "(check the l/atoms selection or that the DOS window covers the band)")
    abs_center = (w * e).sum() / total
    variance = (w * (e - abs_center) ** 2).sum() / total
    width = variance.clamp_min(0.0).sqrt()
    ref_t = ref if isinstance(ref, torch.Tensor) else torch.as_tensor(ref, dtype=RDTYPE)
    return BandMoments(
        center=abs_center - ref_t.to(RDTYPE),
        abs_center=abs_center,
        width=width,
        variance=variance,
        weight=total,
    )


def _l_letter(l: int | str) -> str:
    if isinstance(l, str):
        s = l.strip().upper()
        if s not in _L_LETTER.values():
            raise ValueError(f"unknown angular-momentum label {l!r}; use s/p/d/f")
        return s
    if l not in _L_LETTER:
        raise ValueError(f"angular momentum {l} out of range; use 0..3 or s/p/d/f")
    return _L_LETTER[l]


def _orbital_letter(group_key: str) -> str | None:
    """The l-letter (S/P/D/F) of a pdos group key like 'atom1:3D' or
    'atom1:3D_z2', or None for an l-less grouping ('total', 'atomN')."""
    if ":" not in group_key:
        return None
    label = group_key.split(":", 1)[1].split("_", 1)[0]  # e.g. '3D'
    letters = [c for c in label if c.isalpha()]
    return letters[-1].upper() if letters else None


def _atom_index(group_key: str) -> int | None:
    """The 0-based atom index of a pdos group key 'atomN:...' (labels are 1-based)."""
    head = group_key.split(":", 1)[0]
    if head.startswith("atom") and head[4:].isdigit():
        return int(head[4:]) - 1
    return None


def _select_dos(
    pdos: ProjectedDOS, l: int | str, atoms: Iterable[int] | None, spin: int | None,
) -> np.ndarray:
    """Sum the pdos group arrays matching angular momentum ``l`` (and ``atoms``),
    returning a 1D DOS over the energy grid. For nspin=2, ``spin`` selects a
    channel or (None) sums both."""
    want = _l_letter(l)
    atom_set = None if atoms is None else {int(a) for a in atoms}
    picked = []
    for key, arr in pdos.groups.items():
        if _orbital_letter(key) != want:
            continue
        if atom_set is not None and _atom_index(key) not in atom_set:
            continue
        a = np.asarray(arr, dtype=float)
        if a.ndim == 2:  # (nspin, npoints)
            a = a[spin] if spin is not None else a.sum(axis=0)
        picked.append(a)
    if not picked:
        raise ValueError(
            f"no '{want}' groups in the projected DOS (group_by={pdos.group_by!r}); "
            "rebuild projected_dos with group_by='l' or 'lm', and check the "
            "atoms selection")
    return np.sum(picked, axis=0)


def band_center(
    pdos: ProjectedDOS,
    *,
    l: int | str = "d",
    atoms: Iterable[int] | None = None,
    ref: str | float = "fermi",
    moment: int = 1,
    spin: int | None = None,
) -> float:
    """d-band center (or width) from a :class:`ProjectedDOS`.

    Selects the ``l``-projected DOS (default the d band), optionally restricted to
    ``atoms`` (0-based indices, matching pdos' 1-based ``atomN`` labels), and
    returns its 1st moment about the reference (``moment=1``, the band center) or
    its width (``moment=2``, sqrt of the 2nd central moment), in eV.

    ``ref`` is ``"fermi"`` (subtract ``pdos.fermi_eV``; requires it be set),
    ``"none"``/``None``/0 for the absolute mean, or an explicit float. For
    nspin=2, ``spin`` selects a channel (0/1) or, if None, sums both spins.

    This is the numpy-grid convenience returning a float; for the differentiable
    descriptor call :func:`band_moments` on the torch projection weights directly.
    """
    if moment not in (1, 2):
        raise ValueError("moment must be 1 (center) or 2 (width)")
    if ref in ("fermi", "ef", "E_F"):
        if pdos.fermi_eV is None:
            raise ValueError("ref='fermi' but the ProjectedDOS carries no fermi_eV")
        ref_val = float(pdos.fermi_eV)
    elif ref in (None, "none", "abs", "absolute"):
        ref_val = 0.0
    else:
        ref_val = float(ref)

    dos = _select_dos(pdos, l, atoms, spin)
    e = torch.as_tensor(np.asarray(pdos.energy_eV, dtype=float), dtype=RDTYPE)
    w = torch.as_tensor(dos, dtype=RDTYPE)
    m = band_moments(e, w, ref=ref_val)
    return float(m.center if moment == 1 else m.width)
