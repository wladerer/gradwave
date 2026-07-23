"""Magnetocrystalline anisotropy by the magnetic force theorem.

The expensive route to an anisotropy energy is one full self-consistent SOC
SCF per magnetization direction (examples/fept_mae.py). The force theorem
replaces all but the first: converge (rho, m) once along a reference axis,
then for each direction n rotate the magnetization rigidly, rebuild the
frozen-potential spinor Hamiltonian, and diagonalize ONCE. To second order in
the density change the total-energy difference equals the occupied
band-energy difference,

    E(n) - E(ref) ~ F_band(n) - F_band(ref),
    F_band = sum_k w_k sum_b f_kb eps_kb - sigma*S,

because the double-counting terms are evaluated on the SAME frozen fields for
every direction and cancel exactly in the difference. Each direction refits
its own Fermi level at fixed electron count.

Why the rotation is exact for the potential: the locally-collinear XC gives
B_xc parallel to m with magnitude a function of (rho, |m|) only, so rotating m
rigidly rotates B_xc rigidly and leaves v_xc untouched. The anisotropy enters
solely through the SOC projector block, which is fixed in the lattice frame
and does NOT co-rotate. Without it (a scalar-relativistic pseudo) the band
sum is exactly direction-independent, which is the rotation-invariance gate in
tests/integration/test_mae_force_theorem.py.

Each one-shot solve is seeded with the SU(2)-rotated reference spinors: for
the SOC-free part of H the rotation is the exact eigenbasis, so the Davidson
polishes only the SOC-induced change and converges in a few rounds. The cost
per direction is roughly one SCF iteration, against a full SCF per direction
for the self-consistent route.

Scope: norm-conserving spinor path (scf_noncollinear on an is_fr system).
The reference must be converged on the FULL k-mesh (use_symmetry=False,
time_reversal=False): a k-mesh folded by the magnetic group of the reference
axis is not a valid quadrature for a rotated moment, whose magnetic group is
different. With ``magmoms=`` each one-shot solve instead folds into its OWN
direction's magnetic (Shubnikov) IBZ: the folded representatives are points
of the full mesh, so the solve runs on a subset of the reference spheres with
the folded weights, and the SU(2)-rotated seeds gather straight from the
reference coefficients. The fold is exact for the collinear part of the
frozen magnetization (rho and |m| carry the crystal symmetry, the uniform
rotated direction transforms as an axial vector). The small SOC-induced
transverse textures in m(r) break it at a level the folded-vs-full gate in
tests/integration/test_mae_force_theorem.py measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.batch import build_batched, projectors_b
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import r_to_g
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import NoncollinearXC, vxc_and_bxc
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.noncollinear import NCResult, SpinorHamiltonian
from gradwave.solvers.davidson import davidson_batched


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v)


def _rotation_between(a: torch.Tensor, b: torch.Tensor):
    """Rodrigues rotation taking unit vector a onto unit vector b.

    Returns (R, axis, theta) with R (3,3) float64, axis a unit 3-vector and
    theta in [0, pi]. The antiparallel case picks an arbitrary axis
    perpendicular to a (any is valid: the rotations differ by a rotation
    about a, which is a symmetry of the starting texture)."""
    a, b = _unit(a.to(RDTYPE)), _unit(b.to(RDTYPE))
    cos = float(torch.dot(a, b))
    eye = torch.eye(3, dtype=RDTYPE)
    if cos > 1.0 - 1e-14:
        return eye, torch.tensor([0.0, 0.0, 1.0], dtype=RDTYPE), 0.0
    if cos < -1.0 + 1e-14:
        seed = torch.tensor([1.0, 0.0, 0.0], dtype=RDTYPE)
        if abs(float(torch.dot(a, seed))) > 0.9:
            seed = torch.tensor([0.0, 1.0, 0.0], dtype=RDTYPE)
        axis = _unit(seed - torch.dot(seed, a) * a)
        theta = torch.pi
    else:
        axis = _unit(torch.linalg.cross(a, b))
        theta = float(torch.arccos(torch.clamp(torch.dot(a, b), -1.0, 1.0)))
    kx, ky, kz = (float(x) for x in axis)
    k_cross = torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
                           dtype=RDTYPE)
    r_mat = eye + math.sin(theta) * k_cross \
        + (1.0 - math.cos(theta)) * (k_cross @ k_cross)
    return r_mat, axis, theta


def _spin_rotate(coeffs: torch.Tensor, m_pw: int, axis, theta: float) -> torch.Tensor:
    """Apply the SU(2) rotation U = exp(-i theta/2 sigma.axis) to spinor
    coefficients (nk, nb, 2*m_pw)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    kx, ky, kz = (float(x) for x in axis)
    uu = complex(c, -s * kz)
    ud = complex(-s * ky, -s * kx)      # -i s (kx - i ky)
    du = complex(s * ky, -s * kx)       # -i s (kx + i ky)
    dd = complex(c, s * kz)
    cu, cd = coeffs[..., :m_pw], coeffs[..., m_pw:]
    return torch.cat([uu * cu + ud * cd, du * cu + dd * cd], dim=-1)


def _mesh_key(kfrac, mesh):
    """Integer mesh coordinates of a fractional k-point, as a dict key."""
    return tuple(int(x) for x in np.round(np.asarray(kfrac) * mesh).astype(np.int64) % mesh)


def _fold_setup(eval_system):
    """Paramagnetic group and full-mesh index map for per-direction folding.

    Returns (sg, cell, mesh, full_index) where mesh is the (3,) MP division
    recovered from the stored sphere k-points and full_index maps the integer
    mesh coordinates of every k-point to its index in eval_system.spheres.
    Raises if the stored k-set does not form a full Γ-centered MP mesh —
    only then is every folded representative guaranteed to be a stored point.
    """
    from gradwave.symmetry import find_spacegroup

    cell = np.asarray(eval_system.grid.cell, dtype=np.float64)
    kf = np.array([s.k_frac for s in eval_system.spheres])
    mesh = np.array([len(np.unique(np.round(kf[:, i], 9))) for i in range(3)],
                    dtype=np.int64)
    on_mesh = np.abs(kf * mesh - np.round(kf * mesh)).max() < 1e-8
    keys = {_mesh_key(k, mesh) for k in kf}
    if len(kf) != int(mesh.prod()) or not on_mesh or len(keys) != len(kf):
        raise ValueError(
            "magmoms= folding needs the reference on a full Γ-centered "
            f"Monkhorst-Pack mesh; the stored {len(kf)} k-points do not form "
            f"one (inferred divisions {tuple(int(n) for n in mesh)})")
    full_index = {_mesh_key(k, mesh): i for i, k in enumerate(kf)}
    pos = eval_system.positions.detach().cpu().numpy()
    sg = find_spacegroup(cell, pos @ np.linalg.inv(cell),
                         eval_system.species_of_atom)
    return sg, cell, mesh, full_index


@dataclass
class MAEResult:
    """Force-theorem band free energies over magnetization directions.

    ``band_free_energies[i]`` is F_band for ``directions[i]`` [eV];
    ``mae`` is F_band - F_band[0], the anisotropy relative to the first
    (reference) direction. ``eigenvalues[i]`` is the (nk, nb) spectrum and
    ``fermi[i]`` the direction's own Fermi level. ``nk[i]`` is the number of
    k-points the direction was evaluated on (its magnetic-IBZ fold when
    ``magmoms=`` was passed, the full mesh otherwise). ``meta`` carries
    caller-supplied provenance (mesh, ecut, machine, ...) through
    ``save``/``load``."""

    directions: list
    band_free_energies: torch.Tensor  # (ndir,) [eV]
    mae: torch.Tensor                 # (ndir,) F - F[0] [eV]
    fermi: list
    eigenvalues: list                 # per direction (nk, nb)
    nk: list                          # per direction k-count
    meta: dict | None = None          # provenance, survives save/load

    def save(self, path, meta: dict | None = None) -> None:
        """Write the full result (spectra included) to ``path`` via
        torch.save. A ``meta`` dict passed here replaces the stored one."""
        torch.save({
            "format": "gradwave-mae-1",
            "directions": [[float(x) for x in d] for d in self.directions],
            "band_free_energies": self.band_free_energies.cpu(),
            "mae": self.mae.cpu(),
            "fermi": [float(f) for f in self.fermi],
            "eigenvalues": [e.cpu() for e in self.eigenvalues],
            "nk": [int(n) for n in self.nk],
            "meta": self.meta if meta is None else meta,
        }, path)

    @classmethod
    def load(cls, path) -> MAEResult:
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d.get("format") != "gradwave-mae-1":
            raise ValueError(f"{path}: not a gradwave MAE result file")
        return cls(directions=d["directions"],
                   band_free_energies=d["band_free_energies"],
                   mae=d["mae"], fermi=d["fermi"],
                   eigenvalues=d["eigenvalues"], nk=d["nk"],
                   meta=d.get("meta"))


@torch.no_grad()
def force_theorem_mae(
    res: NCResult,
    xc: NoncollinearXC,
    directions,
    ref_dir=None,          # reference axis of res; default: its net moment
    smearing: str = "gaussian",
    width: float = 0.1,
    diago_tol: float = 1e-10,
    system=None,           # optional full-mesh evaluation system (same box)
    magmoms=None,          # (na,3) reference per-atom moments → per-direction fold
    verbose: bool = True,
) -> MAEResult:
    """Anisotropy energies from one converged SOC SCF plus one frozen-potential
    diagonalization per direction.

    ``res`` is a converged ``scf_noncollinear`` result on a fully-relativistic
    system with the full k-mesh. ``directions`` is a list of magnetization
    axes; the first is the reference for the returned ``mae`` differences (it
    is re-evaluated through the same one-shot machinery, so the force-theorem
    residual cancels in the difference rather than contaminating it).

    ``magmoms`` (the per-atom moments of the reference texture, e.g. the
    ``mag_vec_init`` the reference SCF was seeded with) folds each one-shot
    solve into its own direction's magnetic (Shubnikov) IBZ: the moments are
    rotated with the direction, the magnetic group of the rotated texture
    folds the mesh, and the solve runs on the surviving subset of the stored
    k-points with the folded weights. The reference SCF still needs the full
    mesh. Only the evaluations fold."""
    eval_system = res.system if system is None else system
    if eval_system.rho_symmetrizer is not None:
        raise ValueError(
            "force_theorem_mae needs the full k-mesh: a mesh folded by the "
            "reference magnetic group is not a valid quadrature for a rotated "
            "moment. Converge the reference with use_symmetry=False, "
            "time_reversal=False (or pass system= built that way)")
    if not eval_system.is_fr:
        if verbose:
            print("force_theorem_mae: no SOC (scalar-relativistic pseudos) — "
                  "band sums will be direction-independent", flush=True)
    if res.coeffs is None:
        raise ValueError("res carries no spinor coefficients")
    if system is not None and tuple(system.grid.shape) != tuple(res.system.grid.shape):
        raise ValueError(
            f"evaluation system FFT box {tuple(system.grid.shape)} differs from "
            f"the reference box {tuple(res.system.grid.shape)} — build it with "
            "the same cell and ecut")

    grid, bk = eval_system.grid, eval_system.batch
    device = eval_system.positions.device
    vol = grid.volume
    m_pw = bk.npw_max
    scheme = SCHEMES[smearing]

    ref = torch.as_tensor(
        res.mag_vec if ref_dir is None else ref_dir, dtype=RDTYPE)
    if float(torch.linalg.norm(ref)) < 1e-8:
        raise ValueError("reference direction is undefined (zero net moment) — "
                         "pass ref_dir explicitly")
    ref = _unit(ref)

    # frozen direction-independent pieces: v_H + v_loc from rho, projectors, SOC
    rho, m = res.rho.to(device), res.m.to(device)
    rho_g_box = r_to_g(rho.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    vloc_g = local_potential_g(eval_system.positions, eval_system.species_index,
                               eval_system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    projs_b = projectors_b(bk, eval_system.positions)
    q_so = dij_so = None
    if eval_system.is_fr:
        from gradwave.core.spinor_proj import build_so_projectors

        q_so, dij_so = build_so_projectors(bk, eval_system)
    t2 = torch.cat([bk.t, bk.t], dim=-1)
    mask2 = torch.cat([bk.mask, bk.mask], dim=-1)
    c_all = res.coeffs.to(device)

    fold = magmoms is not None
    if fold:
        from gradwave.symmetry import magnetic_spacegroup, reduce_mesh_magnetic

        magmoms_np = np.atleast_2d(np.asarray(magmoms, dtype=np.float64))
        sg0, cell, mesh, full_index = _fold_setup(eval_system)

    f_band, fermis, spectra, nks = [], [], [], []
    for n_dir in directions:
        n_t = _unit(torch.as_tensor(n_dir, dtype=RDTYPE))
        r_mat, axis, theta = _rotation_between(ref, n_t)
        m_rot = torch.einsum("ij,jxyz->ixyz", r_mat.to(device), m)
        # v_xc depends on (rho, |m|) only and B_xc co-rotates with m, so one
        # call on the rotated field is the exactly-rotated frozen potential
        v_xc, b_xc, _ = vxc_and_bxc(xc, rho, m_rot, grid,
                                    rho_core=eval_system.rho_core)

        if fold:
            # this direction's Shubnikov group folds the mesh; every
            # representative is a stored full-mesh point, so the solve runs
            # on a subset of the reference spheres and seeds
            mg = magnetic_spacegroup(sg0, magmoms_np @ r_mat.numpy().T, cell)
            kfrac_d, kw_np = reduce_mesh_magnetic(
                tuple(int(x) for x in mesh), (0, 0, 0), mg)
            idx = [full_index[_mesh_key(k, mesh)] for k in kfrac_d]
            bk_d = build_batched([eval_system.spheres[i] for i in idx],
                                 [eval_system.proj_data[i] for i in idx],
                                 device=device)
            m_d = bk_d.npw_max
            t2_d = torch.cat([bk_d.t, bk_d.t], dim=-1)
            mask2_d = torch.cat([bk_d.mask, bk_d.mask], dim=-1)
            projs_d = projectors_b(bk_d, eval_system.positions)
            q_d = dij_d = None
            if eval_system.is_fr:
                from gradwave.core.spinor_proj import build_so_projectors

                tabs = [t[idx][:, :, :m_d] for t in eval_system.so_beta_tables]
                q_d, dij_d = build_so_projectors(bk_d, eval_system,
                                                 so_tables=tabs)
            kw_d = torch.as_tensor(kw_np, dtype=RDTYPE, device=device)
            seed0 = torch.cat([c_all[idx, :, :m_d],
                               c_all[idx, :, m_pw:m_pw + m_d]], dim=-1)
        else:
            bk_d, m_d, t2_d, mask2_d = bk, m_pw, t2, mask2
            projs_d, q_d, dij_d = projs_b, q_so, dij_so
            kw_d, seed0 = eval_system.kweights, c_all

        h = SpinorHamiltonian(bk_d, grid.shape, v_h + v_xc + vloc_r, b_xc,
                              projs_d, q=q_d, dij_so=dij_d)
        seed = _spin_rotate(seed0, m_d, axis, theta)
        dav = davidson_batched(h.apply, seed, t2_d, mask2_d, tol=diago_tol)
        eigs = dav.eigenvalues.to(RDTYPE)

        mu = float(find_fermi(eigs, kw_d, scheme, width,
                              eval_system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=eigs.device)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width,
                                             degeneracy=1.0)
        e_band = float((kw_d[:, None] * occ * eigs).sum())
        entropy_term = float(-width * (kw_d[:, None] * s_ent).sum())
        f_band.append(e_band + entropy_term)
        fermis.append(mu)
        spectra.append(eigs)
        nks.append(int(eigs.shape[0]))
        if verbose:
            d_mev = (f_band[-1] - f_band[0]) * 1000.0
            print(f"  FT-MAE n=({float(n_t[0]):+.3f},{float(n_t[1]):+.3f},"
                  f"{float(n_t[2]):+.3f})  nk={nks[-1]}  "
                  f"F_band = {f_band[-1]:+.8f} eV  dF = {d_mev:+.4f} meV",
                  flush=True)

    f_t = torch.tensor(f_band, dtype=RDTYPE)
    return MAEResult(directions=list(directions), band_free_energies=f_t,
                     mae=f_t - f_t[0], fermi=fermis, eigenvalues=spectra,
                     nk=nks)
