"""MixLayout — the composite mixing vector's single source of truth.

The SCF's mixed variable is [ρ channels on the density sphere, flattened
becsum per spin], with nspin=2 grid channels in the (total, magnetization)
basis. Before this class the packing was assembled inline in the SCF loop
and re-derived independently by the mixer rig, the Newton finisher, and
ad-hoc scripts; every copy was one normalization bug waiting to happen
(docs/manual/wisdom.md, Conventions). The layout also owns the derived mixer
vectors, Kerker mask (ρ-total block only), becsum step scale, and adaptive
block ids.
"""

from __future__ import annotations

import torch

from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE


class MixLayout:
    def __init__(self, grid, nspin: int, atom_slices, device=None,
                 bec_step_scale: float = 0.4):
        self.shape = tuple(grid.shape)
        self.n_pts = grid.n_points
        self.mask = grid.dens_mask.reshape(-1)
        self.nspin = nspin
        self.atom_slices = list(atom_slices)
        self.dev = device if device is not None else self.mask.device
        self.g2_sphere = grid.g2.reshape(-1)[self.mask]
        self.ng = int(self.g2_sphere.shape[0])
        self.nbec = sum((s1 - s0) ** 2 for (s0, s1) in self.atom_slices)
        n_grid = self.ng * nspin
        self.g2_full = torch.cat(
            [self.g2_sphere] * nspin
            + [torch.zeros(self.nbec, device=self.dev)] * nspin)
        self.kerker_mask = torch.cat([
            torch.ones(self.ng, dtype=torch.bool, device=self.dev),
            torch.zeros(n_grid - self.ng + self.nbec * nspin,
                        dtype=torch.bool, device=self.dev),
        ])
        self.step_scale = torch.cat([
            torch.ones(n_grid, dtype=torch.float64, device=self.dev),
            torch.full((self.nbec * nspin,), bec_step_scale,
                       dtype=torch.float64, device=self.dev),
        ])
        # adaptive-damping blocks: ρ_tot grid, m grid, becsum
        self.block_ids = torch.cat([
            torch.zeros(self.ng, dtype=torch.int64, device=self.dev),
            torch.ones(self.ng * (nspin - 1), dtype=torch.int64,
                       device=self.dev),
            torch.full((self.nbec * nspin,), 2, dtype=torch.int64,
                       device=self.dev),
        ])

    @property
    def size(self) -> int:
        return self.ng * self.nspin + self.nbec * self.nspin

    def pack(self, rho_spin, becs) -> torch.Tensor:
        """(per-spin r-space densities, per-spin becsum lists) → flat
        complex mixing vector, grid channels in the (total, mag) basis."""
        vecs = [r_to_g(c.to(CDTYPE)).reshape(-1)[self.mask]
                for c in rho_spin]
        if self.nspin == 2:
            vecs = [vecs[0] + vecs[1], vecs[0] - vecs[1]]
        bec_flat = [torch.cat([m.reshape(-1) for m in becs[isp]])
                    for isp in range(self.nspin)]
        return torch.cat(vecs + bec_flat)

    def unpack(self, v: torch.Tensor):
        """Inverse of pack: → (per-spin r-space real densities, per-spin
        becsum lists)."""
        ng = self.ng
        if self.nspin == 2:
            tot, mag = v[:ng], v[ng:2 * ng]
            chans = [(tot + mag) / 2.0, (tot - mag) / 2.0]
        else:
            chans = [v[:ng]]
        rho_spin = []
        for c in chans:
            box = torch.zeros(self.n_pts, dtype=CDTYPE, device=v.device)
            box[self.mask] = c
            rho_spin.append(torch.fft.ifftn(
                box.reshape(self.shape) * self.n_pts,
                dim=(-3, -2, -1)).real)
        becs = [[] for _ in range(self.nspin)]
        off = ng * self.nspin
        for isp in range(self.nspin):
            for (s0, s1) in self.atom_slices:
                n = s1 - s0
                becs[isp].append(v[off:off + n * n].reshape(n, n).clone())
                off += n * n
        return rho_spin, becs

    def unpack_grid_channels(self, v: torch.Tensor):
        """The raw grid channel slices [(tot), (mag)] without FFTs."""
        return [v[i * self.ng:(i + 1) * self.ng]
                for i in range(self.nspin)]
