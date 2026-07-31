"""Reverse-Kerker damping of the transverse magnetization channels.

Executes recommendation 4 of the noncollinear_convergence campaign: the spinor
residual floor is dominated by long-wavelength (low-|G|) TRANSVERSE magnetization
modes, the magnon-soft sector, amplified ~3x per iteration by the mixing map
because their restoring force is near zero at q -> 0. The cure is the mirror of
Kerker: where Kerker suppresses low-G CHARGE modes (the Hartree kernel amplifies
them), here we suppress low-G TRANSVERSE modes (the exchange has near-zero gain
there, so the mixer must take small steps, not large ones).

Implementation. The driver exposes ``scf_noncollinear(..., mixer_hook=...)``,
called with the raw packed (vin, vout) BEFORE ``mixer.step``. The packed layout
is [rho, m_x, m_y, m_z], each ``ng`` complex G-space coefficients. The mixer
takes a step proportional to the residual r = vout - vin, so scaling down a
residual component scales down the step there. We rewrite vout in place so the
residual the mixer sees is damped on the transverse-perpendicular low-G modes:

    r_m -> r_par + D(G) * r_perp          (kerker form)
    r_m -> r_par + alpha_perp * r_perp    (flat form, null hypothesis)

with r_par the component of the m-residual along the current total-moment
direction and r_perp the perpendicular remainder. D(G) = G^2 / (G^2 + q0^2) is
the Kerker mirror: D -> 0 as G -> 0 (strong damping of the soft sector), D -> 1
at high G (no damping). CRITICAL: the G=0 coefficient is the uniform component
of m -- the total moment, which must stay free to evolve (rotate, grow, select a
branch). So the G=0 residual is NEVER damped, in any direction. This is exactly
the campaign's "reverse-Kerker on the magnon-soft sector, G != 0 only".

The hook touches ONLY vout's m blocks. The charge block and the longitudinal
(parallel) m component are passed through untouched. No src edits: everything is
done through the existing mixer_hook argument.

Frames (frame=):
  "moment": transverse is perpendicular to the current instantaneous TOTAL
            moment direction (the G=0 m vector of vin). Falls back to lab z if
            the moment is ~0.
  "lab":    transverse is the m_x, m_y channels; parallel is m_z. Correct only
            for a z-aligned collinear-limit state.
"""

from __future__ import annotations

import torch


class TransverseDampingHook:
    def __init__(self, system, q0: float = 1.0, frame: str = "moment",
                 form: str = "kerker", alpha_perp: float = 0.3,
                 inner_probe=None, damp_g0: bool = False):
        grid = system.grid
        mask_flat = grid.dens_mask.reshape(-1)
        g2 = grid.g2.reshape(-1)[mask_flat]
        self.ng = int(mask_flat.sum())
        self.g0 = int(torch.argmin(g2).item())
        self.q0 = float(q0)
        self.frame = frame
        self.form = form
        self.alpha_perp = float(alpha_perp)
        self.inner_probe = inner_probe
        self.damp_g0 = damp_g0
        # Kerker mirror kernel D(G) = G^2/(G^2+q0^2): 0 at G->0, 1 at high G.
        self.D = (g2 / (g2 + self.q0 ** 2)).to(torch.float64)
        # flat kernel: constant alpha_perp on every G.
        self.flat = torch.full((self.ng,), self.alpha_perp, dtype=torch.float64)
        if not damp_g0:
            # never touch the uniform component (the total moment).
            self.D[self.g0] = 1.0
            self.flat[self.g0] = 1.0
        self.applied: list[dict] = []

    def _factor(self):
        return self.D if self.form == "kerker" else self.flat

    def __call__(self, it: int, vin: torch.Tensor, vout: torch.Tensor):
        if self.inner_probe is not None:
            # record the RAW (undamped) residual first, so the trace measures the
            # true map's transverse amplification.
            self.inner_probe(it, vin, vout)
        if self.frame == "none":
            return
        ng = self.ng
        vin_m = vin[ng:4 * ng].reshape(3, ng)
        vout_m = vout[ng:4 * ng].reshape(3, ng)
        r_m = vout_m - vin_m  # (3, ng) complex raw m-residual
        # frame direction n_hat (real 3-vector)
        if self.frame == "lab":
            nhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                                device=vin.device)
        else:  # "moment"
            m0 = torch.stack([vin_m[c, self.g0].real for c in range(3)]).double()
            nrm = float(m0.norm())
            nhat = (m0 / nrm if nrm > 1e-8
                    else torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                                      device=vin.device))
        nhat_c = nhat.to(r_m.dtype)
        # decompose the m-residual per G into parallel / perpendicular to n_hat
        proj = (nhat_c[:, None] * r_m).sum(dim=0)  # (ng,) complex
        r_par = nhat_c[:, None] * proj  # (3, ng)
        r_perp = r_m - r_par
        factor = self._factor().to(r_perp.real.dtype)
        r_m_new = r_par + r_perp * factor[None, :]
        vout[ng:4 * ng] = (vin_m + r_m_new).reshape(-1)
        # cheap diagnostic: how much transverse residual power was removed
        p_before = float((r_perp.abs() ** 2).sum())
        p_after = float(((r_perp * factor[None, :]).abs() ** 2).sum())
        self.applied.append({
            "it": it,
            "nhat": [round(float(x), 4) for x in nhat],
            "perp_power_kept": round(p_after / p_before, 4) if p_before > 0 else 1.0,
        })
