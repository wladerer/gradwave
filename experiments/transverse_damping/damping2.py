"""Variant B: transverse low-q damping of the mixer's TOTAL update.

The residual-only hook (damping.py) leaves the transverse amplification
unchanged, measured bit-for-bit through iteration 20 on SOC-free Ni. The Pulay
update has two parts, the extrapolated state sum_i c_i vin_i and the
preconditioned step alpha P sum_i c_i r_i. The hook damps only the step. The
DIIS coefficients c_i are set by the dominant channels (charge and longitudinal
m, orders of magnitude above the transverse noise early on), so the state
recombination freely amplifies the transverse content it was never asked to
contract. That term is the amplifier, and the hook cannot reach it.

Variant B wraps ``mixer.step`` instead. The full update u = mixed - vin,
recombination included, is decomposed per G into components parallel and
perpendicular to the current total moment direction, and the perpendicular
part is scaled by D(G) = G^2/(G^2 + q0^2) at G != 0 (G=0 free, exactly as in
damping.py). At a fixed point the update is zero, so the fixed point is
unchanged. The wrap is installed by monkeypatching
``gradwave.scf.noncollinear._build_nc_mixer`` from the experiment side, no src
edits (a production implementation would put this inside the mixer as a
locally-rotated per-block step scale, see the README).

Usage:
    patch = StepWrapPatch(q0=4.0, frame="moment", form="kerker")
    with patch:
        res = scf_noncollinear(...)
    # patch.applied has per-iteration diagnostics
"""

from __future__ import annotations

import torch

import gradwave.scf.noncollinear as _ncmod


class StepWrapPatch:
    """Context manager that monkeypatches _build_nc_mixer so the built mixer's
    step() damps the transverse low-G part of the total update."""

    def __init__(self, q0: float = 4.0, frame: str = "moment",
                 form: str = "kerker", alpha_perp: float = 0.3):
        self.q0 = float(q0)
        self.frame = frame
        self.form = form
        self.alpha_perp = float(alpha_perp)
        self.applied: list[dict] = []
        self._orig = None

    def __enter__(self):
        self._orig = _ncmod._build_nc_mixer
        patch = self

        def patched(g2_vec, ng, nonmagnetic, mixing_alpha, mag_mixing_alpha,
                    mixing_history, precond_op, m, device, mag_mixer="pulay"):
            mixer, base_step_scale, m = patch._orig(
                g2_vec, ng, nonmagnetic, mixing_alpha, mag_mixing_alpha,
                mixing_history, precond_op, m, device, mag_mixer)
            if nonmagnetic:
                return mixer, base_step_scale, m
            g0 = int(torch.argmin(g2_vec).item())
            if patch.form == "kerker":
                D = (g2_vec / (g2_vec + patch.q0 ** 2)).to(torch.float64)
            else:
                D = torch.full((ng,), patch.alpha_perp, dtype=torch.float64)
            D = D.clone()
            D[g0] = 1.0  # the uniform component (total moment) stays free
            orig_step = mixer.step

            def step(vin, vout):
                mixed = orig_step(vin, vout)
                u_m = (mixed[ng:4 * ng] - vin[ng:4 * ng]).reshape(3, ng)
                if patch.frame == "lab":
                    nhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                                        device=vin.device)
                else:
                    m0 = torch.stack(
                        [vin[c * ng + g0].real for c in (1, 2, 3)]).double()
                    nrm = float(m0.norm())
                    nhat = (m0 / nrm if nrm > 1e-8 else
                            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64,
                                         device=vin.device))
                nhat_c = nhat.to(u_m.dtype)
                proj = (nhat_c[:, None] * u_m).sum(dim=0)
                u_par = nhat_c[:, None] * proj
                u_perp = u_m - u_par
                fac = D.to(u_perp.real.dtype)
                u_new = u_par + u_perp * fac[None, :]
                out = mixed.clone()
                out[ng:4 * ng] = (vin[ng:4 * ng].reshape(3, ng)
                                  + u_new).reshape(-1)
                p_before = float((u_perp.abs() ** 2).sum())
                p_after = float(((u_perp * fac[None, :]).abs() ** 2).sum())
                patch.applied.append({
                    "nhat": [round(float(x), 4) for x in nhat],
                    "perp_power_kept": (round(p_after / p_before, 4)
                                        if p_before > 0 else 1.0)})
                return out

            mixer.step = step
            return mixer, base_step_scale, m

        _ncmod._build_nc_mixer = patched
        return self

    def __exit__(self, *exc):
        _ncmod._build_nc_mixer = self._orig
        return False
