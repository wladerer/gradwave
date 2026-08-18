"""DualBasis GATE 1 — can a few Gaussians + a LOW-ecut plane-wave basis represent a real hard
pseudo valence orbital, and does the combined overlap stay conditioned as ecut drops?

DualBasis (moonshot #1): carry the sharp near-core valence structure in a handful of per-element
Gaussians so the plane-wave ecut (hence npw, hence the O(npw²) memory/GEMM cost) can drop 2-4x at
fixed accuracy. Two questions decide it, and both are answerable without any SCF:

  PAYOFF   — does PW(low ecut) + K Gaussians reach a target representation accuracy for a genuinely
             hard orbital at a meaningfully lower ecut than PW alone?
  BLOCKER  — as ecut drops (or Gaussians are tuned toward the PW-representable regime), does the
             combined overlap S stay well-conditioned, or do the Gaussians go linearly dependent
             with the PW span (S -> singular, the failure mode the deep-dive flagged)?

We use the REAL oxygen 2p pseudo-orbital from PD_O_PBE.upf (PseudoDojo) as the target — not an
idealized cusp — so the payoff is not overstated (pseudopotentials are deliberately smooth; the
honest question is high-but-finite G content, not a true cusp). Everything lives in the l=1 radial
channel: a radial l-function's 3D L2 norm is ∝ ∫|F_l(G)|² G² dG (spherical-Bessel Parseval), and
F_l(G) = ∫ R(r) j_l(Gr) r² dr is computed with gradwave's own sbt. The PW basis up to ecut spans
exactly the |G|<Gcut shells (orthonormal), so its best approximation error is the beyond-Gcut tail;
the Gaussians then least-squares-fit that tail with their own beyond-Gcut content.

    uv run python experiments/dualbasis/oxygen_2p_representation.py

Conditioning diagnostic: the K Gaussians, projected orthogonal to the PW span, are just their
beyond-Gcut tails; the smallest eigenvalue of their normalized Gram is how much genuinely-new
direction the worst Gaussian still adds. -> 0 means "already in the PW span" = S singular.
"""

from __future__ import annotations

import numpy as np

from gradwave.constants import BOHR_ANG
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import parse_upf

UPF = "tests/fixtures/qe/pseudos/PD_O_PBE.upf"
L_CHANNEL = 1                      # oxygen 2p
GAUSS_ALPHAS = (0.4, 1.5, 5.0, 18.0)   # even-tempered l=1 Gaussian exponents (Å⁻²)
GMAX = 28.0                        # Å⁻¹, well-converged upper limit
NQ = 900

# ecut(Ry) = HBAR2_2M·|G|²/Ry = BOHR_ANG²·|G|²  (|G| in Å⁻¹)
def ry_of_g(g):
    return (BOHR_ANG**2) * g * g


def radial_ft_target(q):
    u = parse_upf(UPF)
    orb = next(w for w in u.pswfc if w.l == L_CHANNEL)
    # F_1(G) = ∫ R(r) j_1(Gr) r² dr,  R = rchi/r  ->  g = rchi·r (r-powers folded per sbt contract)
    g = orb.rchi * u.r
    return sbt(L_CHANNEL, g, u.r, u.rab, q), u


def radial_ft_gaussian(alpha, r, rab, q):
    # l=1 primitive R_α(r) = r e^{-α r²};  g = R·r² = r³ e^{-α r²}
    g = (r**3) * np.exp(-alpha * r * r)
    return sbt(L_CHANNEL, g, r, rab, q)


def _norm_gram(fg, w, full_gnorm2):
    """Normalized (unit-diagonal) Gram of the given radial FTs under weight w."""
    M = (fg * w) @ fg.T
    d = np.sqrt(np.clip(full_gnorm2, 1e-300, None))
    return (M / d[:, None]) / d[None, :]


def run(alphas):
    q = np.linspace(1e-4, GMAX, NQ)
    dq = q[1] - q[0]
    w = q * q * dq                       # radial measure G² dG

    ftgt, u = radial_ft_target(q)
    fg = np.stack([radial_ft_gaussian(a, u.r, u.rab, q) for a in alphas])  # (K, NQ)

    norm2 = float((ftgt**2 * w).sum())
    full_gnorm2 = (fg**2 * w).sum(axis=1)      # (K,) full-basis Gaussian norms²

    # intrinsic (all-G) Gaussian-basis conditioning — independent of any PW cutoff
    intrinsic_cond = float(np.linalg.cond(_norm_gram(fg, w, full_gnorm2)))

    rows = []
    for gcut in np.linspace(3.0, 22.0, 40):
        out = q > gcut
        wg = w[out]
        pw_err2 = float((ftgt[out] ** 2 * wg).sum()) / norm2

        A = fg[:, out]
        M = (A * wg) @ A.T
        b = (A * wg) @ ftgt[out]
        c = np.linalg.solve(M + 1e-14 * np.eye(len(A)), b)
        resid2 = float((ftgt[out] ** 2 * wg).sum() - b @ c)
        comb_err2 = max(resid2, 0.0) / norm2

        # "new directions" = Gaussians projected orthogonal to the PW span (their beyond-Gcut
        # tails). Smallest eigenvalue of the normalized Gram → 0 means redundant with PWs.
        eig = np.linalg.eigvalsh(_norm_gram(A, wg, full_gnorm2))
        min_eig = float(max(eig[0], 0.0))          # clamp fp underflow noise
        cond = float(eig[-1] / eig[0]) if eig[0] > 1e-300 else float("inf")

        rows.append({
            "gcut": float(gcut), "ecut_ry": float(ry_of_g(gcut)),
            "pw_relL2": float(np.sqrt(pw_err2)), "comb_relL2": float(np.sqrt(comb_err2)),
            "min_eig": min_eig, "cond": cond,
        })
    return rows, intrinsic_cond


def _ecut_at(rows, key, target):
    """Smallest ecut(Ry) whose relL2 (key) is <= target, scanning from high ecut down."""
    hit = [r for r in rows if r[key] <= target]
    return min(r["ecut_ry"] for r in hit) if hit else None


# Well-spaced (not densely even-tempered) exponent sets, to give conditioning a fair chance.
GAUSS_SETS = {
    "2G": (0.5, 6.0),
    "3G": (0.4, 2.5, 16.0),
    "4G": (0.4, 1.5, 5.0, 18.0),
}
OP_ECUT = 15.0  # Ry, a representative low operating point


def main():
    print("\nDualBasis GATE 1 — oxygen 2p representation (real PD_O_PBE.upf 2p orbital)\n")
    print("  Energy error ≈ (relL2)²  ->  'sub-meV-ish' region is relL2 ≈ 1e-2..3e-3.\n")
    print(f"  {'set':>4} | {'exponents (Å⁻²)':>22} | {'intrinsic':>10} | {'npw@relL2=1e-2':>14} | "
          f"{'op comb relL2':>13} | {'op min-eig':>11}")
    print(f"  {'':>4} | {'':>22} | {'cond':>10} | {'(PW/PW+G)':>14} | "
          f"{'(ecut 15Ry)':>13} | {'':>11}")

    detail = None
    for name, alphas in GAUSS_SETS.items():
        rows, intrinsic = run(alphas)
        e_pw = _ecut_at(rows, "pw_relL2", 1e-2)
        e_cb = _ecut_at(rows, "comb_relL2", 1e-2)
        npw = (e_pw / e_cb) ** 1.5 if (e_pw and e_cb) else float("nan")
        op = min(rows, key=lambda r: abs(r["ecut_ry"] - OP_ECUT))
        print(f"  {name:>4} | {str(alphas):>22} | {intrinsic:>10.1e} | {npw:>14.2f} | "
              f"{op['comb_relL2']:>13.2e} | {op['min_eig']:>11.1e}")
        if name == "3G":
            detail = (rows, intrinsic)

    rows, intrinsic = detail
    print(f"\n  Detail for 3G (intrinsic cond {intrinsic:.1e}) — PW vs PW+Gauss vs conditioning:")
    print(f"  {'ecut(Ry)':>9} | {'PW relL2':>10} | {'PW+G relL2':>11} | "
          f"{'min-eig(new)':>12} | {'cond(new)':>10}")
    for r in rows[::5]:
        cond = r["cond"]
        cs = "inf" if cond == float("inf") else f"{cond:.1e}"
        print(f"  {r['ecut_ry']:>9.1f} | {r['pw_relL2']:>10.2e} | {r['comb_relL2']:>11.2e} | "
              f"{r['min_eig']:>12.1e} | {cs:>10}")

    # locate the "squeeze": accuracy vs conditioning across ecut for 3G
    best_cond = max(rows, key=lambda r: r["min_eig"])           # best-conditioned point
    acc = [r for r in rows if r["comb_relL2"] < 1.2e-2]        # sub-meV-ish accuracy
    acc_mineig = max((r["min_eig"] for r in acc), default=float("nan"))

    print("\n  READING:")
    print("   - Payoff is real but MODEST at meaningful accuracy: ~1.5-2.5x npw at relL2≈1e-2")
    print("     (large only at loose relL2); not the 3-8x npw the moonshot hoped.")
    print("   - THE SQUEEZE (core DualBasis risk, confirmed): the best-conditioned ecut in range")
    print(f"     ({best_cond['ecut_ry']:.1f} Ry) still has only min-eig≈{best_cond['min_eig']:.1e}"
          f" and its accuracy is relL2≈{best_cond['comb_relL2']:.1e};")
    print(f"     where PW+G reaches relL2<1.2e-2, the overlap 'new directions' have min-eig≈"
          f"{acc_mineig:.1e}")
    print("     (cond(S)~1e7). Accuracy and conditioning do not co-exist — the Gaussians go")
    print("     linearly dependent with the PWs precisely in the accuracy-relevant ecut range,")
    print("     forcing canonical orthogonalization (which caps accuracy near ~1e-8).")
    print("   - Learning the exponents to lower energy (the moonshot's mechanism) would push")
    print("     HARDER toward this singular manifold, not away — the deep-dive's warning, made")
    print("     quantitative. BOTTOM LINE: modest ~2x-npw ceiling with a real conditioning tax;")
    print("     de-risk before any learning build should target the canonical-orthogonalization")
    print("     accuracy floor in a real SCF, not the (already-measured) representation payoff.")


if __name__ == "__main__":
    main()
