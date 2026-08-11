"""Work function and electrode potential from an open-boundary (ESM) SCF.

For a slab with a vacuum/plate region the electrostatic potential there is a
well-defined reference (unlike a periodic cell), so the work function

    Φ = E_vac − E_F                                            [eV]

is directly available, and with it the electrode potential on the absolute
(vacuum) scale U_abs = Φ/e [V] and, via the standard hydrogen electrode's
absolute potential (≈ 4.44 V, Trasatti; ~0.1 V uncertainty),

    U(vs SHE) = Φ/e − U_SHE_abs.

E_vac is read as the plane-averaged effective potential in the low-density
region along the open axis (deep in the vacuum / at the grounded plate, where
v_xc → 0 so v_eff is the electrostatic potential). This is exact for the ESM
open-boundary modes (``open_z`` / ``open_z_metal``), where that level is
box-independent; for a bare periodic slab it is only as good as the vacuum is
converged. Asymmetric slabs have two different face potentials — see
``work_function(..., both_faces=True)``.

The convention-free reference is the **potential of zero charge** (the neutral
Fermi level): ``U − U_PZC = Φ − Φ_PZC`` needs no external constant and is what
capacitance / relative-potential studies use. Only placing a number on the SHE
axis needs ``U_SHE_abs``, which is external in every constant-potential code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Absolute potential of the standard hydrogen electrode [V] (Trasatti's
# IUPAC-recommended value; literature spans ~4.2–4.85 depending on the water
# surface-potential convention, so absolute-vs-SHE carries ~0.1 V uncertainty).
U_SHE_ABS = 4.44


def _plane_profile(field: torch.Tensor, open_axis: int) -> torch.Tensor:
    """Plane-average a (n1,n2,n3) or (nspin,n1,n2,n3) field to a 1D profile along
    the open axis. Spins are averaged (v_xc → 0 in the vacuum region anyway)."""
    if field.ndim == 4:
        field = field.mean(0)
    axes = tuple(a for a in range(3) if a != open_axis)
    return field.mean(dim=axes)


def vacuum_level(res, open_axis: int = 2, frac: float = 0.1,
                 both_faces: bool = False):
    """E_vac [eV]: the plane-averaged effective potential in the vacuum region.

    Averages ``v_eff`` over the ``frac`` fraction of open-axis slabs with the
    lowest plane-averaged density (the deep vacuum / plate, where v_xc ≈ 0). With
    ``both_faces`` returns ``(E_lo, E_hi)`` — the vacuum levels beyond the two
    ends of the slab, which differ for a dipolar (asymmetric) surface.
    """
    v_z = _plane_profile(res.v_eff, open_axis)
    rho_z = _plane_profile(res.rho, open_axis)
    nz = rho_z.numel()
    if both_faces:
        # split the box at the density maximum (inside the slab); take the
        # lowest-density points on each side as that face's vacuum.
        mid = int(torch.argmax(rho_z))
        # If the slab straddles the periodic seam (vacuum in the cell middle,
        # slab wrapping z=0/z=nz), argmax lands near an edge and one [0:mid] /
        # [mid:nz] segment holds no vacuum — an empty slice makes topk raise, a
        # tiny one returns an in-slab plane as a bogus "face". Recenter the slab
        # on the open axis first; a slab already in the interior (argmax well
        # away from both edges) triggers no roll and is left byte-identical.
        k_vac = max(1, int(frac * nz))
        if min(mid, nz - mid) < k_vac:
            shift = nz // 2 - mid
            rho_z = torch.roll(rho_z, shift)
            v_z = torch.roll(v_z, shift)
            mid = int(torch.argmax(rho_z))
        out = []
        for lo, hi in ((0, mid), (mid, nz)):
            seg_rho, seg_v = rho_z[lo:hi], v_z[lo:hi]
            k = max(1, int(frac * seg_rho.numel()))
            idx = torch.topk(seg_rho, k, largest=False).indices
            out.append(float(seg_v[idx].mean()))
        return tuple(out)
    k = max(1, int(frac * nz))
    idx = torch.topk(rho_z, k, largest=False).indices
    return float(v_z[idx].mean())


def work_function(res, open_axis: int = 2, both_faces: bool = False):
    """Work function Φ = E_vac − E_F [eV] (a tuple per face if ``both_faces``)."""
    ev = vacuum_level(res, open_axis, both_faces=both_faces)
    if both_faces:
        return tuple(e - res.fermi for e in ev)
    return ev - res.fermi


@dataclass
class ElectrodePotential:
    work_function: float          # Φ [eV]
    potential_vs_vacuum: float    # U_abs = Φ/e [V]
    potential_vs_she: float       # U − U_SHE [V]
    potential_vs_pzc: float | None = None  # U − U_PZC [V] (if a PZC ref was given)


def electrode_potential(res, *, u_she_abs: float = U_SHE_ABS,
                        phi_pzc: float | None = None,
                        open_axis: int = 2) -> ElectrodePotential:
    """Electrode potential of an ESM slab, on the vacuum, SHE, and (optional) PZC
    scales.

    ``u_she_abs`` is the absolute SHE potential [V] (external, ~4.44). ``phi_pzc``
    is the work function [eV] of the potential-of-zero-charge reference (e.g. from
    a neutral / fixed-N run at the same geometry); when given, the convention-free
    ``U − U_PZC`` is included.
    """
    phi = work_function(res, open_axis)
    return ElectrodePotential(
        work_function=phi,
        potential_vs_vacuum=phi,
        potential_vs_she=phi - u_she_abs,
        potential_vs_pzc=None if phi_pzc is None else phi - phi_pzc,
    )
