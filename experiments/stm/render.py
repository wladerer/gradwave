import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
img = np.load("experiments/stm/out/graphene_stm.npy")
cell = np.load("experiments/stm/out/graphene_cell.npy")
pos = np.load("experiments/stm/out/graphene_pos.npy")
n1, n2 = img.shape
a1, a2 = cell[0, :2], cell[1, :2]
T = 4
# point-centred coords (gouraud smooth), fractional -> cartesian, tiled & periodic-wrapped
fi = np.arange(T*n1) / n1
fj = np.arange(T*n2) / n2
FI, FJ = np.meshgrid(fi, fj, indexing="ij")
X = FI*a1[0] + FJ*a2[0]
Y = FI*a1[1] + FJ*a2[1]
Z = np.tile(img, (T, T))
fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=170)
ax.pcolormesh(X, Y, Z, cmap="afmhot", shading="gouraud")
for I in range(T+1):
    for J in range(T+1):
        for p in pos:
            r = p[:2] + I*a1 + J*a2
            ax.plot(r[0], r[1], "o", ms=6, mfc="none", mec="#33ccff", mew=1.3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Graphene · Tersoff-Hamann STM (LDOS @ E$_F$, tip 2 Å)\ngradwave  postscf.stm", fontsize=10)
# crop to a clean 2.2 x 2.2 cell window in the interior
cx, cy = 1.6*a1 + 1.6*a2
w = 2.1*np.linalg.norm(a1)
ax.set_xlim(cx - w/2, cx + w/2); ax.set_ylim(cy - w/2, cy + w/2)
plt.tight_layout()
plt.savefig("experiments/stm/out/graphene_stm.png", bbox_inches="tight")
print("re-rendered (gouraud)")
