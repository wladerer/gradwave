"""Experiment-side convergence probes for the non-collinear / SOC SCF.

The SCF flight recorder (scf/recorder.py) does not support the spinor path (it
was deferred in PR #203 because a vector magnetization breaks the moment / shell
heuristics the collinear recorder leans on). So this module builds the probe
out of the one hook the driver already exposes, ``scf_noncollinear(...,
mixer_hook=...)``, which is called each iteration with the RAW packed residual
(vin, vout) BEFORE the mixer touches it. No src changes, no monkeypatching.

Packing (magnetic run): vin/vout are the density-sphere G-space coefficients of
[rho, m_x, m_y, m_z] concatenated, each channel ``ng`` complex entries (see
scf/spinor_common.pack_grid_channels). The G=0 coefficient of a channel is its
real-space mean (r_to_g = fftn/N), so the integrated moment is
``vout[c*ng+g0].real * vol`` -- the exact quantity NCResult.mag_vec reports.

Recorded per iteration:
  - dn:   charge-channel raw residual norm ||rho_out - rho_in||
  - dm:   magnetization raw residual norm as a vector-field norm over m_x,m_y,m_z
  - dm_x/dm_y/dm_z: per-Cartesian-component magnetization residual norms
  - mvec: integrated moment vector (mu_B) at this iteration's OUTPUT density
  - dtheta_out: angle (deg) between this iter's moment direction and the previous
  - shell_charge / shell_mag: fraction of squared residual power per |G| shell
"""

from __future__ import annotations

import math

import torch


def _angle_deg(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return float("nan")
    c = float((a @ b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


class NCConvergenceProbe:
    """A ``mixer_hook`` callable that records the per-iteration channel
    decomposition of the raw (pre-mix) residual. Pass an instance as
    ``scf_noncollinear(..., mixer_hook=probe)`` and read ``probe.records``
    afterward (a list of per-iteration dicts, JSON-serializable)."""

    def __init__(self, system, nonmagnetic: bool, n_shells: int = 12):
        grid = system.grid
        mask_flat = grid.dens_mask.reshape(-1)
        g2 = grid.g2.reshape(-1)[mask_flat]
        self.ng = int(mask_flat.sum())
        self.vol = float(grid.volume)
        self.nonmagnetic = nonmagnetic
        gmag = torch.sqrt(g2.clamp_min(0.0))
        self.g0 = int(torch.argmin(g2).item())  # |G| == 0 index
        gmax = float(gmag.max()) or 1.0
        # linear |G| shell edges, same scheme as scf/recorder.py
        edges = torch.linspace(0.0, gmax * (1 + 1e-9), n_shells + 1)
        self.shell_idx = torch.bucketize(gmag, edges[1:-1].contiguous())
        self.n_shells = n_shells
        self.shell_edges = [float(e) for e in edges]
        self.records: list[dict] = []
        self._mvec_prev = None

    def _shell_fracs(self, resid_block):
        # resid_block: (ng,) or (3, ng) complex raw residual; power per |G| shell
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
            rec["shell_mag"] = self._shell_fracs(rm)
            mvec = torch.tensor(
                [float(vout[c * ng + self.g0].real) * self.vol for c in (1, 2, 3)])
            rec["mvec"] = [round(float(x), 6) for x in mvec]
            rec["m_abs_g0"] = round(float(mvec.norm()), 6)
            rec["dtheta_out"] = (round(_angle_deg(mvec, self._mvec_prev), 4)
                                 if self._mvec_prev is not None else float("nan"))
            self._mvec_prev = mvec
        self.records.append(rec)
