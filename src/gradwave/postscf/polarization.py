"""Berry-phase electronic polarization (King-Smith--Vanderbilt string formula).

The macroscopic electronic polarization of an insulator is a bulk Berry phase
of the occupied Bloch manifold, not an expectation value of the (ill-defined,
unbounded) position operator. King-Smith--Vanderbilt discretize it: along each
primitive reciprocal direction ``d`` the cell-periodic states are transported
around a closed k-string and the accumulated Berry phase is

    phi_d = -Im ln  prod_{j} det M^{(k_j, k_{j+1})},
    M^{(k,k')}_{mn} = <u_{m k} | u_{n k'}>   (occupied bands m, n),

averaged over the perpendicular k-strings. In a plane-wave basis the
cell-periodic overlap is the inner product of the coefficient vectors over the
SHARED reciprocal-lattice (Miller) vectors of the two k-point G-spheres; the
end-of-string link closes the loop across one reciprocal-lattice vector G_d, a
periodic-gauge condition |u_{k+G_d}> = e^{-i G_d.r} |u_k> that becomes a pure
Miller-index shift of one coefficient vector (no e^{-i G_d.r} to apply on a
grid). The reduced polarization is defined so the polarization QUANTUM is 1 per
direction, i.e. e a_d / V for the vector; the branch is documented on
:class:`Polarization`.

Scope: insulators only (integer occupations, a clear gap) — the string formula
assumes an isolated occupied manifold. nspin in {1, 2} (spin channels summed;
factor 2 for the spin-degenerate nspin=1 case). The SCF must have been run on
the FULL (symmetry-unreduced) Monkhorst--Pack mesh whose shape is passed here,
so every string is present: build the System with ``use_symmetry=False`` and an
unshifted mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from gradwave.dtypes import CDTYPE, RDTYPE

if TYPE_CHECKING:
    from gradwave.scf.loop import SCFResult


# --------------------------------------------------------------------------- #
# Low-level string machinery (pure linear algebra — no SCF/physics imports).   #
# --------------------------------------------------------------------------- #


def _match_indices(
    mill_a: Tensor, mill_b: Tensor, gshift: tuple[int, int, int]
) -> tuple[Tensor, Tensor]:
    """Row indices ``(ia, ib)`` of the plane waves shared by two G-spheres.

    Returns the gather indices such that ``mill_a[ia] == mill_b[ib] + gshift``
    (integer Miller 3-vectors). ``gshift`` is the reciprocal-lattice shift that
    closes an end-of-string link (``(0, 0, 0)`` for interior links). The overlap
    then pairs coefficient ``c_a(G)`` with ``c_b(G - gshift)``.
    """
    ma = mill_a.to(torch.int64)
    mb = mill_b.to(torch.int64) + torch.tensor(gshift, dtype=torch.int64, device=mill_b.device)
    # Encode each Miller triple as a single integer key on a common bounding box.
    both = torch.cat([ma, mb], dim=0)
    lo = both.min(dim=0).values
    span = (both.max(dim=0).values - lo + 1).tolist()
    s1, s2 = int(span[1]), int(span[2])

    def _key(m: Tensor) -> Tensor:
        r = m - lo
        return (r[:, 0] * s1 + r[:, 1]) * s2 + r[:, 2]

    key_a, key_b = _key(ma), _key(mb)
    # Build b's key -> row map, then look each a-key up (npw ~ 10^2-10^4; a dict
    # keyed on python ints is simplest and the cost is negligible vs the SCF).
    b_of: dict[int, int] = {int(k): j for j, k in enumerate(key_b.tolist())}
    ia_list: list[int] = []
    ib_list: list[int] = []
    for i, k in enumerate(key_a.tolist()):
        j = b_of.get(int(k))
        if j is not None:
            ia_list.append(i)
            ib_list.append(j)
    dev = mill_a.device
    return (
        torch.tensor(ia_list, dtype=torch.int64, device=dev),
        torch.tensor(ib_list, dtype=torch.int64, device=dev),
    )


def _overlap_matrix(
    c_a: Tensor, c_b: Tensor, ia: Tensor, ib: Tensor
) -> Tensor:
    """Occupied-band overlap ``M[m, n] = sum_G conj(c_a[m, G]) c_b[n, G']``.

    ``c_a`` (nocc, npw_a), ``c_b`` (nocc, npw_b) are the occupied plane-wave
    coefficients; ``ia``/``ib`` select the shared plane waves (from
    :func:`_match_indices`). Returns an (nocc, nocc) complex matrix.
    """
    return c_a[:, ia].conj() @ c_b[:, ib].transpose(0, 1)


def string_berry_phase(
    coeffs: list[Tensor], millers: list[Tensor], e_dir: tuple[int, int, int]
) -> Tensor:
    """Berry phase (radians) of ONE ordered closed k-string.

    ``coeffs[j]`` (nocc, npw_j) and ``millers[j]`` (npw_j, 3) are the occupied
    coefficients and Miller indices at the j-th k-point along the string, ordered
    so consecutive points differ by the string step b = G_d / N. ``e_dir`` is the
    Miller unit vector of the reciprocal direction (e.g. ``(1, 0, 0)``); the
    wrap-around link from the last point back to the first closes across the
    reciprocal-lattice vector ``e_dir``.

    Returns ``-Im ln prod_j det M`` as a real 0-dim tensor (gauge invariant under
    any per-k-point U(N) rotation of the occupied bands, up to the 2*pi branch).
    """
    n = len(coeffs)
    wrap: tuple[int, int, int] = (-e_dir[0], -e_dir[1], -e_dir[2])
    phase = coeffs[0].new_zeros((), dtype=RDTYPE)
    for j in range(n):
        a, b = j, (j + 1) % n
        gshift: tuple[int, int, int] = (0, 0, 0) if b != 0 else wrap
        ia, ib = _match_indices(millers[a], millers[b], gshift)
        m = _overlap_matrix(coeffs[a], coeffs[b], ia, ib)
        sign, _ = torch.linalg.slogdet(m)
        phase = phase + torch.angle(sign)
    return -phase


# --------------------------------------------------------------------------- #
# Result container.                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Polarization:
    """Reduced polarization of an insulator, split into ionic and electronic
    Berry-phase parts, plus the geometry needed to form the physical vector.

    All quantities are per direction of the primitive reciprocal lattice. The
    reduced polarization ``p_d = p_ion_d + p_el_d`` is defined so that the
    polarization QUANTUM is 1 per direction: adding 1 to ``p_d`` shifts the
    physical polarization vector by ``e a_d / V`` (one electron transported by
    the lattice vector ``a_d``). ``p_ion`` and the total are therefore only
    defined modulo 1; :meth:`vector` returns the branch as computed (the
    electronic Berry phase folded into (-1/2, 1/2] per string before averaging is
    NOT done here — see :func:`berry_phase_polarization`, which sums raw phases so
    the value is continuous across small displacements for finite-difference Born
    charges). Units: :attr:`cell` in Angstrom, :attr:`volume` in Angstrom^3;
    :meth:`vector` returns P / e in e-free units of 1 / Angstrom^2.
    """

    reduced_ionic: Tensor      # (3,) p_ion_d = sum_kappa Z_kappa s_{kappa,d}
    reduced_electronic: Tensor  # (3,) p_el_d = -(spin) phi_d / (2 pi)
    berry_phases: Tensor       # (3,) string-averaged phi_d [rad]
    cell: Tensor               # (3, 3) rows = lattice vectors [Angstrom]
    volume: float
    spin_factor: float         # 2 for nspin=1, 1 for a summed nspin=2 run

    @property
    def reduced_total(self) -> Tensor:
        """Total reduced polarization ``p_d`` (3,), modulo the quantum 1."""
        return self.reduced_ionic + self.reduced_electronic

    def vector(self) -> Tensor:
        """Physical polarization vector P / e (3,) [1 / Angstrom^2], Cartesian.

        ``P / e = (1 / V) sum_d p_d a_d``. Defined modulo the quantum vectors
        :meth:`quantum_vectors`.
        """
        return (self.reduced_total @ self.cell) / self.volume

    def quantum_vectors(self) -> Tensor:
        """The three polarization-quantum vectors ``a_d / V`` (3, 3) [1/Ang^2].

        Adding any of these (or an integer combination) to :meth:`vector` gives a
        physically equivalent polarization.
        """
        return self.cell / self.volume


def reduced_ionic_polarization(
    cell: Tensor, positions: Tensor, charges: Tensor
) -> Tensor:
    """Ionic reduced polarization ``p_ion_d = sum_kappa Z_kappa s_{kappa,d}`` (3,).

    ``cell`` (3, 3) rows are lattice vectors [Angstrom]; ``positions`` (na, 3) are
    Cartesian [Angstrom]; ``charges`` (na,) are the ionic (pseudo-valence) charges
    in units of e. ``s_kappa`` are the fractional coordinates. A rigid lattice
    translation shifts ``p_ion`` by ``sum_kappa Z_kappa`` (an integer for a
    neutral cell) — one polarization quantum per electron transported.
    """
    inv = torch.linalg.inv(cell.to(RDTYPE))
    s_frac = positions.to(RDTYPE) @ inv
    return torch.einsum("a,ad->d", charges.to(RDTYPE), s_frac)


# --------------------------------------------------------------------------- #
# Driver: reduced polarization from a converged full-mesh SCF.                 #
# --------------------------------------------------------------------------- #


def _grid_index_map(
    k_frac: list[np.ndarray], mesh: tuple[int, int, int]
) -> dict[tuple[int, int, int], int]:
    """Map integer MP grid coordinates ``(i, j, l)`` -> k-point index."""
    n = np.asarray(mesh, dtype=np.int64)
    out: dict[tuple[int, int, int], int] = {}
    for ik, kf in enumerate(k_frac):
        idx = np.rint(np.asarray(kf, dtype=float) * n).astype(np.int64) % n
        out[(int(idx[0]), int(idx[1]), int(idx[2]))] = ik
    return out


def _occupied_bands(occupations: Tensor, nspin: int) -> int:
    """Number of occupied bands (insulator); asserts integer occupations.

    ``occupations`` is (nk, nb) for nspin=1 or (nspin, nk, nb) for nspin=2.
    """
    occ = occupations.detach().to(RDTYPE)
    full = 2.0 if nspin == 1 else 1.0
    # Per-band occupation averaged over k; an insulator is ~full or ~0.
    if occ.dim() == 3:
        occ = occ.reshape(-1, occ.shape[-1])
    per_band = occ.mean(dim=0)
    occupied = per_band > 0.5 * full
    nocc = int(occupied.sum().item())
    # gap check: lowest "occupied" mean and highest "empty" mean must separate
    if nocc == 0 or nocc == per_band.shape[0]:
        raise ValueError(
            "Berry-phase polarization needs an insulating band structure with "
            "empty conduction bands in the SCF; found none (add nbands or check "
            "the gap)."
        )
    lo_occ = per_band[occupied].min().item()
    hi_emp = per_band[~occupied].max().item()
    if lo_occ - hi_emp < 0.5 * full:
        raise ValueError(
            "Berry-phase polarization requires integer occupations (an "
            f"insulator); occupation of the frontier bands is ambiguous "
            f"(min occupied {lo_occ:.3f}, max empty {hi_emp:.3f} of {full})."
        )
    return nocc


def _spin_coeffs(
    res: SCFResult, spin: int, nocc: int
) -> list[Tensor]:
    """Occupied coefficients per k-point for one spin channel, (nocc, npw_k)."""
    coeffs = res.coeffs
    if res.nspin == 2:
        chan = cast("list[list[Tensor]]", coeffs)[spin]
    else:
        chan = cast("list[Tensor]", coeffs)
    out: list[Tensor] = []
    for ck in chan:
        c = ck.detach().to(CDTYPE)
        out.append(c[:nocc])
    return out


def berry_phase_polarization(
    res: SCFResult,
    mesh: tuple[int, int, int],
    *,
    unwrap_reference: Tensor | None = None,
) -> Polarization:
    """Reduced polarization of ``res`` by the King-Smith--Vanderbilt formula.

    ``res`` must be a converged insulating SCF on the FULL Monkhorst--Pack mesh
    of shape ``mesh`` (``use_symmetry=False``, unshifted). For each of the three
    reciprocal directions the occupied-band Berry phase is averaged over the
    perpendicular k-strings; the electronic reduced polarization is
    ``p_el_d = -(spin) phi_d / (2 pi)`` (spin = 2 for nspin=1, else the two
    channels are summed). The ionic part is ``p_ion_d = sum_kappa Z_kappa
    s_{kappa,d}`` with ``Z_kappa`` the pseudopotential valence charge and
    ``s_kappa`` the fractional atomic coordinates.

    The raw (un-folded) Berry phase is returned so the value moves CONTINUOUSLY
    with a small atomic displacement — the requirement for finite-difference Born
    charges (:mod:`gradwave.postscf.born`). Pass ``unwrap_reference`` (a previous
    run's ``berry_phases``) to unwrap each direction onto the branch nearest the
    reference (differences kept in (-pi, pi]).
    """
    system = res.system
    spheres = system.spheres
    n1, n2, n3 = (int(x) for x in mesh)
    if len(spheres) != n1 * n2 * n3:
        raise ValueError(
            f"berry_phase_polarization: SCF has {len(spheres)} k-points but mesh "
            f"{mesh} needs {n1 * n2 * n3}; run the SCF with use_symmetry=False AND "
            "time_reversal=False on the full unshifted mesh (time reversal still "
            "folds k → -k otherwise)."
        )
    k_frac = [np.asarray(s.k_frac, dtype=float) for s in spheres]
    gmap = _grid_index_map(k_frac, (n1, n2, n3))
    if len(gmap) != n1 * n2 * n3:
        raise ValueError(
            "berry_phase_polarization: k-points do not form a regular unshifted "
            f"{mesh} mesh (got {len(gmap)} distinct grid nodes)."
        )
    millers = [s.miller for s in spheres]

    nspin = res.nspin
    nocc = _occupied_bands(res.occupations, nspin)
    spin_channels = [0, 1] if nspin == 2 else [0]

    dims = (n1, n2, n3)
    phi = torch.zeros(3, dtype=RDTYPE)
    for d in range(3):
        e_dir = [0, 0, 0]
        e_dir[d] = 1
        nd = dims[d]
        perp = [(dims[a] if a != d else 1) for a in range(3)]
        acc = torch.zeros((), dtype=RDTYPE)
        n_strings = 0
        for p0 in range(perp[0]):
            for p1 in range(perp[1]):
                for p2 in range(perp[2]):
                    base = [p0, p1, p2]
                    order: list[int] = []
                    for j in range(nd):
                        coord = list(base)
                        coord[d] = j
                        order.append(gmap[(coord[0], coord[1], coord[2])])
                    mill_str = [millers[k] for k in order]
                    string_phase = acc.new_zeros(())
                    for spin in spin_channels:
                        cs = _spin_coeffs(res, spin, nocc)
                        c_str = [cs[k] for k in order]
                        string_phase = string_phase + string_berry_phase(
                            c_str, mill_str, (e_dir[0], e_dir[1], e_dir[2])
                        )
                    acc = acc + string_phase
                    n_strings += 1
        phi[d] = acc / n_strings
    # spin_factor folds the spin-degeneracy weight into the electronic reduced
    # polarization; for nspin=2 the two channels were already summed above, so
    # each carries weight 1.
    spin_factor = 2.0 if nspin == 1 else 1.0

    if unwrap_reference is not None:
        ref = unwrap_reference.to(RDTYPE)
        two_pi = 2.0 * torch.pi
        phi = ref + torch.remainder(phi - ref + torch.pi, two_pi) - torch.pi

    cell = torch.as_tensor(np.asarray(system.grid.cell, dtype=float), dtype=RDTYPE)
    volume = float(system.grid.volume)
    pos = system.positions.detach().to(RDTYPE)
    z_val = system.charges.detach().to(RDTYPE)  # (na,)
    p_ion = reduced_ionic_polarization(cell, pos, z_val)
    p_el = -(spin_factor / (2.0 * torch.pi)) * phi

    return Polarization(
        reduced_ionic=p_ion,
        reduced_electronic=p_el,
        berry_phases=phi,
        cell=cell,
        volume=volume,
        spin_factor=spin_factor,
    )
