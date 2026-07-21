"""Publication figures for the gradwave paper. Data from committed benchmark
results and this session's validation logs; every number is from a real run."""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.optimize import curve_fit

matplotlib.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.4, "grid.color": "#dddbd4", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})
GW, QE, GWCPU = "#2a78d6", "#52514e", "#1baf7a"  # validated palette slots
HERE = Path(__file__).parent
REPO = HERE.parents[2]
SP = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
RY = 13.605693122994
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=250)
    plt.close(fig)
    print("wrote", name)


def bm3(v, e0, v0, b0, b0p):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x))


# ---------------- Fig 1: Delta-factor EOS ----------------
gw = json.loads((REPO / "benchmarks/delta_factor/results/eos_gw.json").read_text())
qe = {}
for ln in (REPO / "benchmarks/delta_factor/results/eos_qe_energies.txt").read_text().splitlines():
    m = re.match(r"([a-z]+)(\d)\.out:!.*=\s*(-[\d.]+)\s*Ry", ln)
    if m:
        qe.setdefault(m.group(1), {})[SCALES[int(m.group(2))]] = float(m.group(3)) * RY

LABELS = {"si": "Si", "c": "C", "gaas": "GaAs", "mgo": "MgO", "al": "Al", "cu": "Cu"}
DELTAS = {"si": 0.0001, "c": 0.0002, "gaas": 0.010, "mgo": 0.0007, "al": 0.055, "cu": 0.163}
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True)
for ax, case in zip(axes.ravel(), ["si", "c", "gaas", "mgo", "al", "cu"], strict=True):
    nat = gw[case]["natoms"]
    vols = np.array([gw[case]["V_A3"][str(s)] for s in SCALES])
    e_g = np.array([gw[case]["E_eV"][str(s)] for s in SCALES])
    e_q = np.array([qe[case][s] for s in SCALES])
    pg, _ = curve_fit(bm3, vols, e_g, p0=[e_g.min(), vols[3], 0.6, 4.0], maxfev=20000)
    pq, _ = curve_fit(bm3, vols, e_q, p0=[e_q.min(), vols[3], 0.6, 4.0], maxfev=20000)
    vv = np.linspace(vols.min(), vols.max(), 300)
    ax.plot(vv / pg[1], (bm3(vv, *pg) - pg[0]) / nat * 1e3, color=GW, zorder=2,
            label="gradwave (fit)")
    ax.plot(vols / pq[1], (e_q - pq[0]) / nat * 1e3, "o", color=QE, ms=3.5,
            mfc="none", mew=1.0, zorder=3, label="QE (points)")
    ax.set_title(LABELS[case], pad=3)
    ax.text(0.5, 0.86, rf"$\Delta = {DELTAS[case]:.4f}$ meV/at.",
            transform=ax.transAxes, ha="center", fontsize=7, color="#52514e")
    ax.grid(axis="y")
for ax in axes[1]:
    ax.set_xlabel(r"$V/V_0$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$E - E_0$ (meV/atom)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, handlelength=1.4,
           bbox_to_anchor=(0.5, 1.05))
fig.tight_layout(w_pad=1.0)
save(fig, "fig_eos")

# ---------------- Fig 2: NVE conservation ----------------
rows = []
for ln in (REPO / "benchmarks/nve_drift/nve_si8_rtx3050.log").read_text().splitlines():
    p = ln.split()
    if len(p) == 5 and not ln.startswith("NVE"):
        rows.append([float(x) for x in p[:4]])
d = np.array(rows)
_, idx = np.unique(d[:, 0], return_index=True)
d = d[np.sort(idx)]
t, ep, ek, et = d.T
na = 8
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True,
                             height_ratios=[1, 1])
a1.plot(t / 1000, (ep - ep[0]) / na * 1e3, color=GW, label=r"$\Delta E_{\rm pot}$")
a1.plot(t / 1000, ek / na * 1e3, color=GWCPU, label=r"$E_{\rm kin}$")
a1.set_ylabel("energy (meV/atom)")
a1.legend(loc="center right", handlelength=1.4)
a1.grid(axis="y")
dev = (et - et.mean()) / na * 1e3
a2.plot(t / 1000, dev, color=QE, lw=1.0)
a2.axhline(0, color="#dddbd4", lw=0.6, zorder=0)
a2.set_ylabel(r"$E_{\rm tot} - \bar E_{\rm tot}$ (meV/atom)")
a2.set_xlabel("time (ps)")
a2.grid(axis="y")
fig.tight_layout(h_pad=0.7)
save(fig, "fig_nve")

# ---------------- Fig 3: speed ----------------
systems = ["Si", "C", "GaAs", "Al", "Cu", "MgO", r"Si$_8$"]
t_qe = [1.5, 1.5, 1.8, 1.8, 2.2, 1.6, 1.9]
t_cpu = [3.5, 1.8, 7.9, 5.1, 10.0, 3.6, 5.8]
t_gpu = [1.5, 0.9, 2.8, 4.2, 5.1, 1.6, 5.4]
y = np.arange(len(systems))
h = 0.26
fig, ax = plt.subplots(figsize=(3.4, 3.2))
for off, vals, col, lab in [(-h, t_qe, QE, "QE 7.5 (8 MPI)"),
                            (0, t_cpu, GWCPU, "gradwave CPU-22"),
                            (h, t_gpu, GW, "gradwave GPU")]:
    ax.barh(y + off, vals, height=h - 0.03, color=col, label=lab)
    for yi, v in zip(y + off, vals, strict=True):
        ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=6.5,
                color="#52514e")
ax.set_yticks(y, systems)
ax.invert_yaxis()
ax.set_xlabel("SCF wall time (s)")
ax.set_xlim(0, 11.8)
ax.legend(loc="lower right", handlelength=1.0)
save(fig, "fig_speed")

# ---------------- Fig 4: XC fit ----------------
its, loss, kap, mu = [], [], [], []
for ln in (SP / "xc_demo.log").read_text().splitlines():
    m = re.match(r"it\s+(\d+): loss=([\d.e+-]+)\s+kappa=([\d.]+) mu=([\d.]+)", ln)
    if m:
        its.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        kap.append(float(m.group(3)))
        mu.append(float(m.group(4)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
a1.semilogy(its, loss, "o-", color=GW, ms=3.5)
a1.set_xlabel("Adam iteration")
a1.set_ylabel(r"loss (eV$^2$)")
a1.grid(axis="y")
a2.plot(its, mu, "o-", color=GW, ms=3.5, label=r"$\mu$")
a2.plot(its, kap, "s-", color=GWCPU, ms=3.5, label=r"$\kappa$")
a2.axhline(0.2195, color=GW, lw=0.8, ls="--")
a2.axhline(0.804, color=GWCPU, lw=0.8, ls="--")
a2.text(14.2, 0.235, r"$\mu_{\rm PBE}$", fontsize=7, color=GW, ha="right")
a2.text(14.2, 0.77, r"$\kappa_{\rm PBE}$", fontsize=7, color=GWCPU, ha="right")
a2.set_xlabel("Adam iteration")
a2.set_ylabel("parameter value")
a2.set_ylim(0.15, 0.9)
a2.legend(loc="center right", handlelength=1.4)
a2.grid(axis="y")
fig.tight_layout(w_pad=2.0)
save(fig, "fig_xc")

# ---------------- Fig 5: Sternheimer U response ----------------
chi1, chi2 = [], []
for ln in (SP / "nio_sternu.log").read_text().splitlines():
    m = re.match(r"\s+response it\s+(\d+): chi_col = \[([-\d.e]+), ([-\d.e]+)\]", ln)
    if m:
        chi1.append(float(m.group(2)))
        chi2.append(float(m.group(3)))
it = np.arange(1, len(chi1) + 1)
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.plot(it, chi1, "o-", color=GW, ms=3.5, label=r"$\chi_{11}$ (on-site)")
ax.plot(it, chi2, "s-", color=GWCPU, ms=3.5, label=r"$\chi_{21}$ (cross)")
ax.axhline(-0.08733, color=GW, lw=0.8, ls="--")
ax.axhline(-0.00005, color=GWCPU, lw=0.8, ls="--")
ax.annotate(r"$\chi_0$ (bare, iteration 1)", (1, chi1[0]), (1.8, -0.195),
            fontsize=7, color="#52514e",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#52514e"))
ax.set_xlabel("Anderson iteration")
ax.set_ylabel(r"$dN_I/d\alpha_1$ (eV$^{-1}$)")
ax.legend(loc="lower right", handlelength=1.4)
ax.grid(axis="y")
save(fig, "fig_uresp")
print("all figures written")
