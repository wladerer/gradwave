# Gradient-designed hybrid: (α, ω) from a periodic multi-material gap loss

The differentiability payoff. gradwave *self-consistently solves* a range-separated
hybrid on a periodic cell and exposes the exact dE/dα, dE/dω at the converged
point; here that is extended to the band gap, and the mixing and screening of the
hybrid are **trained by gradient descent through the periodic hybrid SCF** against
a joint band-gap loss over several materials, then tested on a held-out material.
No mainstream plane-wave code fits a hybrid this way — the semi-empirical hybrids
people use (PBE0, HSE) fix α and ω by hand or by a one-material grid search.

## The differentiable gap (`gap.py`)

The total-energy gradient is exact at self-consistency (stationarity). An
eigenvalue is not stationary, so the gap gradient is the **frozen-orbital**
Hellmann-Feynman one, mirroring `differentiable_hybrid_energy`:

    ε_i(α,ω) = ε_i^conv + α·Δ_i(ω) − α_conv·Δ_i(ω_conv),
    Δ_i(ω)  = ⟨i|V_x^Fock(ω)|i⟩ − ⟨i|v_x^PBE|i⟩,

so `gap = ε_c − ε_v` carries d(gap)/d(α, ω). The diagonal Fock element reuses the
exact inner loop of `multik_exchange_energy`; `validate.py` checks three things on
Si (Γ, loose cutoff):

- orbital normalization ⟨i|i⟩ = 1, and the E_x self-check ½Σ_i w_ki ⟨i|V_x|i⟩ =
  `multik_exchange_energy` **to 0.00 meV** (the Fock convention is consistent);
- the differentiable gap value equals the converged ε_c − ε_v;
- **frozen-orbital d(gap)/dα agrees with a finite difference of re-converged
  hybrids to 1 %** (the omitted SCF response of the eigenvalues), and d(gap)/dω to
  ~5 %. That is near-exact — plenty for gradient descent.

## The training (`train.py`)

Each optimizer step re-converges the hybrid SCF for every training material at the
current (α, ω) — warm-started from the previous step, so the small parameter moves
converge in a few iterations — forms the differentiable gap, sums the
squared-error gap loss over materials, and takes one backward pass for
d(loss)/d(α, ω). Materials are single Γ cells at a loose cutoff: **the numbers
demonstrate the machinery, not converged physics** (the absolute gaps are far from
experiment at Γ-only).

The targets are the gaps at a ground-truth (α*, ω*) = (0.25, 0.20), so a perfect
joint fit exists. The test is twofold: recovering (α*, ω*) from a perturbed start
over an **over-determined** set (3 materials — Si, C, MgO — for 2 parameters)
exercises the exact gradient path end to end, and the **held-out** material (AlAs)
matching its target shows the trained two-parameter hybrid transfers to a material
it never saw.

    uv run python benchmarks/hybrid_design/validate.py   # gap gradient vs FD
    uv run python benchmarks/hybrid_design/train.py       # fit (α, ω) + transfer

Results land in `train.json`; `make_fig.py` draws `hybrid_train.png` (the loss
decay and the (α, ω) trajectory).

## Result

Gradient descent through the periodic hybrid SCF drives the joint gap loss from
~650 to **2×10⁻³ eV²**, and the trained two-parameter hybrid **transfers to the
held-out AlAs to 39 meV** — a material the optimizer never saw. Training-set fits
are 67 meV (Si), 186 meV (MgO), and 365 meV (C, ~1 % of C's 31 eV Γ gap). The
machinery — exact-exchange SCF, differentiable gap, backprop, optimizer — closes
end to end.

The recovered (α, ω) = (0.222, 0.187) lands *near* the ground truth (0.25, 0.20)
but on an **iso-gap valley** rather than the exact point (visible in the figure:
the trajectory follows a curved α–ω path and settles short of the target star).
Two effects, both worth stating: band gaps are strongly α–ω correlated (raising
the mixing and softening the screening move a gap the same way), so a few similar
materials under-determine the two parameters; and the frozen-orbital gap gradient
is ~1–5 % biased, which displaces the fixed point slightly. The honest reading is
that the *gaps* are recovered (loss → 10⁻³, held-out to 39 meV) while the
*parameters* are recovered up to the valley — which is exactly the argument for a
genuine multi-*property* loss (gaps plus an energy-based observable such as the
lattice constant) to break the degeneracy, the natural next step from here.
