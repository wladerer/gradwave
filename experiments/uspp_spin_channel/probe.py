"""Magnetization-channel residual probe (monkeypatch, no src changes).

The main flight recorder decomposes only the TOTAL-density residual onto
|G|-shells. To see the magnetization channel we wrap the mixer's ``step``,
which receives the composite mixing vector in G-space on the density sphere
with layout [rho_tot (ng), rho_mag (ng), becsum...]. Before the mix we read
the raw residual r = rho_out - rho_in, split its total and magnetization grid
blocks, and bin |r_mag(G)|^2 onto linear |G|-shells (same scheme as
scf/recorder.py). Every quantity is detached; the probe never touches the
differentiable path (the SCF runs under no_grad regardless).

install(system) records ng and the sphere |G| shell bins from that system's
grid, then patches PulayMixer/JohnsonMixer/BroydenMixer.step. Read LOG after
the run (one dict per outer iteration): tot/mag residual L2 norms and the mag
shell fractions. restore() undoes the patch.
"""

from __future__ import annotations

import torch

import gradwave.scf.mixing as _mix

LOG: list[dict] = []
_saved: dict = {}
_state: dict = {}


def _shell_bins(g2_sphere: torch.Tensor, n_shells: int = 12):
    gmag = g2_sphere.sqrt()
    gmax = float(gmag.max())
    edges = torch.linspace(0.0, gmax if gmax > 0 else 1.0, n_shells + 1,
                           device=gmag.device)
    idx = torch.bucketize(gmag, edges) - 1
    return idx.clamp_(0, n_shells - 1).to(torch.long), n_shells


def _frac(r_block: torch.Tensor):
    idx, n_shells = _state["bins"]
    power = (r_block.conj() * r_block).real
    shell = torch.zeros(n_shells, dtype=torch.float64, device=power.device)
    shell.index_add_(0, idx, power.to(torch.float64))
    tot = float(shell.sum())
    if tot == 0.0:
        return [0.0] * n_shells
    return [float(x) / tot for x in shell.cpu().tolist()]


def install(system, n_shells: int = 12) -> None:
    LOG.clear()
    grid = system.grid
    mask = grid.dens_mask.reshape(-1)
    g2_sphere = grid.g2.reshape(-1)[mask]
    ng = int(mask.sum())
    _state["ng"] = ng
    _state["bins"] = _shell_bins(g2_sphere, n_shells)
    _state["it"] = 0

    def make_wrapper(cls):
        orig = cls.step

        def step(self, rho_in, rho_out):
            ng_ = _state["ng"]
            r = (rho_out - rho_in).detach()
            r_tot = r[:ng_]
            r_mag = r[ng_:2 * ng_]
            _state["it"] += 1
            LOG.append({
                "it": _state["it"],
                "tot_norm": float(torch.linalg.norm(r_tot)),
                "mag_norm": float(torch.linalg.norm(r_mag)),
                "mag_shell_frac": _frac(r_mag),
                "tot_shell_frac": _frac(r_tot),
            })
            return orig(self, rho_in, rho_out)

        return orig, step

    for name in ("PulayMixer", "JohnsonMixer", "BroydenMixer"):
        cls = getattr(_mix, name)
        orig, wrapped = make_wrapper(cls)
        _saved[name] = orig
        cls.step = wrapped


def restore() -> None:
    for name, orig in _saved.items():
        getattr(_mix, name).step = orig
    _saved.clear()
