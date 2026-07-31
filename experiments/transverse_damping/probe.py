"""Per-iteration convergence probe for the non-collinear / SOC SCF.

Base probe reused from the noncollinear_convergence campaign: a ``mixer_hook``
callable, called each iteration with the RAW packed (vin, vout) BEFORE the mixer
(and, in this study, before the transverse-damping hook) touches them. Records
the per-channel decomposition of the raw residual so the transverse
amplification is measured on the true (undamped) map.

Packing (magnetic run): vin/vout are the density-sphere G-space coefficients of
[rho, m_x, m_y, m_z] concatenated, each channel ``ng`` complex entries. The G=0
coefficient of a channel is its real-space mean, so the integrated total moment
is ``vout[c*ng+g0].real * vol``.

This study adds an optional per-ATOM moment tracker (nearest-atom Voronoi
partition of the real-space box) so the canted 2-atom kill-criterion can measure
the angle between the two atomic moments per iteration, which the total-moment
G=0 coefficient alone does not resolve.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.scf.spinor_common import unpack_grid_channels


def _angle_deg(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return float("nan")
    c = float((a @ b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


class NCConvergenceProbe:
    """A ``mixer_hook`` callable that records the per-iteration channel
    decomposition of the raw (pre-mix) residual. Read ``.records`` afterward.

    If ``track_per_atom`` is set the probe also reconstructs the real-space
    magnetization each iteration and integrates it over a nearest-atom Voronoi
    partition, recording per-atom moment vectors and the pairwise angle between
    the first two atoms (the canted-cell alignment metric)."""

    def __init__(self, system, nonmagnetic: bool, n_shells: int = 12,
                 track_per_atom: bool = False):
        grid = system.grid
        self.grid = grid
        self.system = system
        self.mask_flat = grid.dens_mask.reshape(-1)
        g2 = grid.g2.reshape(-1)[self.mask_flat]
        self.ng = int(self.mask_flat.sum())
        self.vol = float(grid.volume)
        self.nonmagnetic = nonmagnetic
        gmag = torch.sqrt(g2.clamp_min(0.0))
        self.g0 = int(torch.argmin(g2).item())  # |G| == 0 index
        gmax = float(gmag.max()) or 1.0
        edges = torch.linspace(0.0, gmax * (1 + 1e-9), n_shells + 1)
        self.shell_idx = torch.bucketize(gmag, edges[1:-1].contiguous())
        self.n_shells = n_shells
        self.shell_edges = [float(e) for e in edges]
        self.records: list[dict] = []
        self._mvec_prev = None
        self.track_per_atom = track_per_atom
        if track_per_atom:
            self._build_voronoi()

    def _build_voronoi(self):
        """Nearest-atom (minimum-image) assignment of every real-space box point
        to an atom; the per-atom cell weights integrate m over each atom."""
        grid = self.grid
        n1, n2, n3 = grid.shape
        cell = np.asarray(grid.cell)  # (3,3) rows a_i [Å]
        pos = self.system.positions.detach().cpu().numpy()  # (na,3) Å
        self.na = pos.shape[0]
        # fractional coords of every box point
        fi = (np.arange(n1) / n1)[:, None, None]
        fj = (np.arange(n2) / n2)[None, :, None]
        fk = (np.arange(n3) / n3)[None, None, :]
        frac = np.stack(np.broadcast_arrays(fi, fj, fk), axis=-1)  # (n1,n2,n3,3)
        cart = frac @ cell  # (n1,n2,n3,3) Å
        inv = np.linalg.inv(cell)
        assign = np.zeros((n1, n2, n3), dtype=np.int64)
        best = np.full((n1, n2, n3), np.inf)
        for a in range(self.na):
            d = cart - pos[a]  # (...,3)
            df = d @ inv  # fractional
            df -= np.round(df)  # minimum image
            dmin = df @ cell
            r2 = (dmin ** 2).sum(-1)
            m = r2 < best
            best[m] = r2[m]
            assign[m] = a
        self._atom_mask = [torch.as_tensor(assign == a) for a in range(self.na)]
        self._dvol = self.vol / (n1 * n2 * n3)

    def _per_atom_moments(self, vout):
        """Reconstruct real-space m and integrate over each atom's Voronoi cell.
        Returns list of (mx,my,mz) per atom in mu_B."""
        fields = unpack_grid_channels(
            vout, 4, self.ng, self.mask_flat, self.grid.shape,
            self.grid.n_points, vout.device)
        m_r = fields[1:4]  # [m_x, m_y, m_z], each (n1,n2,n3) real
        out = []
        for a in range(self.na):
            mask = self._atom_mask[a].to(m_r[0].device)
            mv = [float((m_r[c] * mask).sum()) * self._dvol for c in range(3)]
            out.append(mv)
        return out

    def _shell_fracs(self, resid_block):
        p = resid_block.abs() ** 2
        if p.dim() == 2:
            p = p.sum(dim=0)
        tot = float(p.sum())
        out = [0.0] * self.n_shells
        if tot <= 0.0:
            return out
        for s in range(self.n_shells):
            out[s] = float(p[self.shell_idx == s].sum()) / tot
        return out

    def __call__(self, it: int, vin: torch.Tensor, vout: torch.Tensor):
        ng = self.ng
        r = vout - vin
        rec: dict = {"it": it}
        rec["dn"] = float(r[:ng].norm()) * self.vol
        rec["shell_charge"] = self._shell_fracs(r[:ng])
        if not self.nonmagnetic:
            rm = r[ng:4 * ng].reshape(3, ng)
            rec["dm"] = float(rm.norm()) * self.vol
            rec["dm_x"] = float(rm[0].norm()) * self.vol
            rec["dm_y"] = float(rm[1].norm()) * self.vol
            rec["dm_z"] = float(rm[2].norm()) * self.vol
            # G=0 vs G!=0 split of each m-component residual: locates the
            # amplified transverse mode (uniform moment rotation vs finite-q)
            for i, c in enumerate("xyz"):
                a2 = float(rm[i, self.g0].abs() ** 2)
                tot2 = float((rm[i].abs() ** 2).sum())
                rec[f"dm_{c}_g0"] = (a2 ** 0.5) * self.vol
                rec[f"dm_{c}_ne0"] = (max(tot2 - a2, 0.0) ** 0.5) * self.vol
            rec["shell_mag"] = self._shell_fracs(rm)
            mvec = torch.tensor(
                [float(vout[c * ng + self.g0].real) * self.vol for c in (1, 2, 3)])
            rec["mvec"] = [round(float(x), 6) for x in mvec]
            rec["m_abs_g0"] = round(float(mvec.norm()), 6)
            rec["dtheta_out"] = (round(_angle_deg(mvec, self._mvec_prev), 4)
                                 if self._mvec_prev is not None else float("nan"))
            self._mvec_prev = mvec
            if self.track_per_atom:
                pam = self._per_atom_moments(vout)
                rec["per_atom_m"] = [[round(x, 5) for x in v] for v in pam]
                if self.na >= 2:
                    a0 = torch.tensor(pam[0])
                    a1 = torch.tensor(pam[1])
                    rec["pair_angle"] = round(_angle_deg(a0, a1), 4)
                    rec["m0_abs"] = round(float(a0.norm()), 5)
                    rec["m1_abs"] = round(float(a1.norm()), 5)
        self.records.append(rec)
