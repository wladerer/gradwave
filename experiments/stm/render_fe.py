import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
up = np.load("experiments/stm/out/fe_stm_up.npy")
dn = np.load("experiments/stm/out/fe_stm_dn.npy")
asym = (up - dn) / (up + dn + 1e-30)
cell = np.load("experiments/stm/out/fe_cell.npy")
T = 4
def tile(a): return np.tile(a, (T, T))
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150)
for ax, dat, ttl, cm in [(axes[0], tile(up), "spin ↑ LDOS", "afmhot"),
                         (axes[1], tile(dn), "spin ↓ LDOS", "afmhot"),
                         (axes[2], tile(asym), "spin asymmetry (↑−↓)/(↑+↓)", "RdBu_r")]:
    im = ax.imshow(dat.T, origin="lower", cmap=cm, interpolation="bicubic", aspect="equal")
    ax.set_title(ttl, fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Fe(100) spin-polarized STM (SP-STM) @ E$_F$, tip 2.5 Å  ·  gradwave postscf.stm  ·  M=8.0 µB",
             fontsize=11)
plt.tight_layout()
plt.savefig("experiments/stm/out/fe_sp_stm.png", bbox_inches="tight")
print(f"rendered; asym range [{asym.min():+.3f},{asym.max():+.3f}], "
      f"up/dn peak ratio {up.max()/dn.max():.2f}")
