"""SoftModeDeflate — publication figure (3 panels) from this session's real data.

(a) the rescue: BiFeO3 PAW+U response solve at its AFM instability — baseline
    Anderson stalls, deflation converges.
(b) the regime: fcc Ni — deflation is neutral until the soft mode crosses 1,
    then it holds the iteration count while the baseline blows up.
(c) the scaling: the composite response apply is ~24x faster on an H100.
"""
import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "sans-serif",
})
BLUE = "#0072B2"    # deflated (the win)
VERM = "#D55E00"    # baseline
INK = "#222222"
MUTE = "#9aa0a6"

# ---- (a) BiFeO3 fxc=1.5 convergence trajectories -------------------------- #
base = [2.226e-2,1.654e-2,1.127e-2,7.728e-3,6.853e-3,6.595e-3,6.604e-3,6.549e-3,
        6.312e-3,5.910e-3,4.808e-3,4.291e-3,3.685e-3,3.445e-3,3.415e-3,3.413e-3,
        3.409e-3,3.397e-3,3.286e-3,3.135e-3,2.542e-3,1.912e-3,1.387e-3,1.006e-3,
        5.413e-4,4.963e-4,4.566e-4,4.474e-4,4.336e-4,4.325e-4,4.297e-4,4.279e-4,
        4.260e-4,4.236e-4,4.227e-4,4.192e-4,4.184e-4,4.159e-4,4.166e-4,4.108e-4,
        3.899e-4,3.589e-4,3.258e-4,2.796e-4,2.196e-4,1.549e-4,1.142e-4,9.637e-5,
        4.873e-5,3.734e-5,2.254e-5,1.336e-5,1.119e-5,9.191e-6,8.661e-6,8.133e-6,
        7.749e-6,7.596e-6,7.529e-6,7.503e-6]
defl = [1.000e0,8.003e-1,1.559e-2,1.135e-2,5.433e-3,3.894e-3,2.999e-3,2.804e-3,
        2.751e-3,2.750e-3,2.751e-3,2.707e-3,2.321e-3,2.122e-3,1.610e-3,1.243e-3,
        1.043e-3,5.983e-4,4.777e-4,3.147e-4,1.604e-4,9.305e-5,6.734e-5,5.735e-5,
        3.580e-5,2.189e-5,1.037e-5,5.694e-6,4.631e-6,3.144e-6,2.762e-6,1.826e-6,
        1.101e-6,6.477e-7]

# ---- (b) fcc Ni sweep ----------------------------------------------------- #
fxc = np.array([1.0, 1.5, 2.0, 2.5])
ni_base = np.array([11, 13, 19, 56])
ni_defl = np.array([12, 13, 18, 24])
ni_lam = np.array([0.42, 0.63, 0.84, 1.06])   # top soft eigenvalue

# ---- (c) H100 scaling: composite response apply time (s), SAME box -------- #
hw = ["CPU\n(16 threads)", "H100\n(real fp64)"]
apply_s = [54.52, 2.29]
speedup = [1.0, 54.52/2.29]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 4.1))
fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.16, wspace=0.30)

# --- (a) --- #
axA.semilogy(range(1, len(base)+1), base, color=VERM, lw=2, label="baseline Anderson")
axA.semilogy(range(1, len(defl)+1), defl, color=BLUE, lw=2, label="deflated")
axA.axhline(1e-6, color=MUTE, lw=1, ls=":")
axA.text(2, 1.3e-6, "tolerance", color=MUTE, fontsize=8, va="bottom")
axA.scatter([60], [base[-1]], color=VERM, s=42, zorder=5, marker="X")
axA.annotate("baseline stalls\n(cap 60, |r|=7.5e-6)", (60, base[-1]),
             xytext=(38, 2.5e-5), color=VERM, fontsize=8.5, ha="left",
             arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8))
axA.scatter([34], [defl[-1]], color=BLUE, s=48, zorder=5, marker="o")
axA.annotate("deflated converges\n(34 iters)", (34, defl[-1]),
             xytext=(20, 3.0e-7), color=BLUE, fontsize=8.5, ha="left")
axA.set_xlabel("iteration")
axA.set_ylabel(r"relative residual  $\|r\|/\|\bar v\|$")
axA.set_title("(a)  BiFeO$_3$ PAW+U — response solve\nat its AFM instability "
              r"($\lambda$ = 1.05, 1.02)", loc="left")
axA.legend(frameon=False, fontsize=8.5, loc="upper right")
axA.set_ylim(2e-7, 5e-2)
axA.set_xlim(0, 64)

# --- (b) --- #
axB.plot(fxc, ni_base, color=VERM, lw=2, marker="X", ms=8, label="baseline")
axB.plot(fxc, ni_defl, color=BLUE, lw=2, marker="o", ms=7, label="deflated")
# mark where the soft mode crosses 1 (between 2.0 and 2.5)
cross = np.interp(1.0, ni_lam, fxc)
axB.axvspan(cross, 2.6, color=VERM, alpha=0.06)
axB.axvline(cross, color=MUTE, ls="--", lw=1)
axB.text(cross+0.02, 52, r"$\lambda_{\max}\!>\!1$" + "\n(super-critical)",
         color=INK, fontsize=8.5, va="top")
axB.annotate("2.3×", (2.5, 40), color=BLUE, fontsize=11, fontweight="bold", ha="center")
axB.annotate("", (2.5, ni_base[-1]), xytext=(2.5, ni_defl[-1]),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
axB.set_xlabel(r"exchange scaling $s$  (approach to the Stoner instability)")
axB.set_ylabel("iterations to converge")
axB.set_title("(b)  fcc Ni (NC metal) — deflation is neutral\nuntil the soft mode "
              "crosses 1, then it holds", loc="left")
axB.legend(frameon=False, fontsize=8.5, loc="upper left")
axB.set_xlim(0.9, 2.62)
axB.set_ylim(0, 62)

# --- (c) --- #
cols = [MUTE, BLUE]
bars = axC.bar(hw, apply_s, color=cols, width=0.5, zorder=3)
for b, s in zip(bars, apply_s, strict=True):
    axC.text(b.get_x()+b.get_width()/2, s+1.2, f"{s:.1f} s",
             ha="center", va="bottom", fontsize=10, color=INK)
axC.annotate("23.8×", (1, 2.29), xytext=(1, 26), color=BLUE, fontsize=20,
             fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
axC.text(0.5, 45, "same box · strict fp64\nRTX 3050 (crippled fp64):\n2.7× on the same apply",
         fontsize=8, color=INK, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f6f7", ec=MUTE, lw=0.6))
axC.set_ylabel("one composite response apply (s)")
axC.set_title("(c)  the batched fp64 response scales:\n~24× on an H100 "
              "vs CPU", loc="left")
axC.set_ylim(0, 60)

fig.suptitle("Soft-mode deflation for the near-critical DFT response solve — "
             "NC metals → PAW+U multiferroics, and it scales on GPU",
             fontsize=12.5, fontweight="bold", x=0.055, ha="left", y=0.985)

fig.savefig("/home/wladerer/.claude/jobs/13456a1d/tmp/softmode_deflate_figure.pdf")
fig.savefig("/home/wladerer/.claude/jobs/13456a1d/tmp/softmode_deflate_figure.png")
print("wrote softmode_deflate_figure.{pdf,png}")
