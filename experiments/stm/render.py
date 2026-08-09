"""Render a Tersoff-Hamann STM map on the hexagonal grid (Cartesian, tiled)."""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

img = np.load("experiments/stm/out/g_occ.npy")
cell = np.load("experiments/stm/out/g_cell.npy")
pos = np.load("experiments/stm/out/g_pos.npy")
a1, a2 = cell[0, :2], cell[1, :2]
t = 4
fi = np.arange(t * img.shape[0]) / img.shape[0]
fj = np.arange(t * img.shape[1]) / img.shape[1]
fim, fjm = np.meshgrid(fi, fj, indexing="ij")
x = fim * a1[0] + fjm * a2[0]
y = fim * a1[1] + fjm * a2[1]
z = np.tile(img, (t, t))
fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=170)
ax.pcolormesh(x, y, z, cmap="afmhot", shading="gouraud")
for i in range(t + 1):
    for j in range(t + 1):
        for p in pos:
            r = p[:2] + i * a1 + j * a2
            ax.plot(r[0], r[1], "o", ms=6, mfc="none", mec="#33ccff", mew=1.3)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Graphene, Tersoff-Hamann STM (occupied $\\pi$, tip 2 A)\ngradwave  postscf.stm",
             fontsize=10)
cx, cy = 1.5 * a1 + 1.5 * a2
w = 2.1 * np.linalg.norm(a1)
ax.set_xlim(cx - w / 2, cx + w / 2)
ax.set_ylim(cy - w / 2, cy + w / 2)
plt.tight_layout()
plt.savefig("experiments/stm/out/graphene_stm.png", bbox_inches="tight")
print("rendered graphene_stm.png")
