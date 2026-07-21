"""Figure for the gradient-designed hybrid: the (α, ω) trajectory descending to
the target, and the joint gap loss decaying. Reads train.json. Agg, 16 pt."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SP = Path(__file__).parent
BLUE, AQUA, GRAY = "#2a78d6", "#1baf7a", "#52514e"
plt.rcParams.update({"font.size": 16, "axes.linewidth": 0.8})


def main():
    d = json.loads((SP / "train.json").read_text())
    h = d["history"]
    step = [x["step"] for x in h]
    alpha = np.array([x["alpha"] for x in h])
    omega = np.array([x["omega"] for x in h])
    loss = np.array([x["loss"] for x in h])
    ast, ost = d["alpha_star"], d["omega_star"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    ax0.semilogy(step, loss, "-", color=BLUE, lw=2)
    ax0.set_xlabel("optimizer step")
    ax0.set_ylabel("joint gap loss  (eV²)")

    ax1.plot(alpha, omega, "-", color=GRAY, lw=1.2, zorder=1)
    ax1.scatter(alpha, omega, c=step, cmap="viridis", s=28, zorder=2)
    ax1.scatter([alpha[0]], [omega[0]], marker="o", s=120, facecolors="none",
                edgecolors=BLUE, lw=2, label="start", zorder=3)
    ax1.scatter([ast], [ost], marker="*", s=320, color=AQUA, label="target",
                zorder=3)
    ax1.set_xlabel(r"$\alpha$  (exchange mixing)")
    ax1.set_ylabel(r"$\omega$  (screening, Å$^{-1}$)")
    ax1.legend(frameon=False, loc="upper right")
    for ax in (ax0, ax1):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = SP / "hybrid_train.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  (recovered α,ω = {d['recovered'][0]:.4f}, {d['recovered'][1]:.4f})")


if __name__ == "__main__":
    main()
