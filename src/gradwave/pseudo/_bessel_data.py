"""Shared small-argument constants for the two hand-rolled spherical-Bessel
evaluators — radial.sph_jl (numpy/scipy setup path) and radial_torch.jl_t
(differentiable torch mirror).

The numpy/torch split itself is deliberate (custom autograd on one side,
scipy-free numpy on the other) and is NOT unified. Only the *parameters* live
here, so the two series cannot silently drift apart the way they had (2.0 vs
4.0 thresholds, 30 vs 40 terms, dict vs list double-factorials).
tests/unit/test_review_dedup.py pins numpy-vs-torch parity so a future edit to
one path that forgets the other fails loudly.
"""

from __future__ import annotations

# (2l+1)!! for the small-argument ascending series, indexed by l. l ≤ 4 covers
# USPP/PAW augmentation channels (L ≤ 2·l_max_beta); l = 5 appears only in the
# torch derivative path, where j_l' needs j_{l+1}.
DOUBLE_FACTORIAL = (1.0, 3.0, 15.0, 105.0, 945.0, 10395.0)

# Below SERIES_X evaluate the ascending power series (converged to ~1e-17 in at
# most SERIES_TERMS terms); at or above it the closed trigonometric forms are
# stable. Both paths use this one (wider window, more terms) pair: the tighter
# (2.0 / 30) pair the numpy path used to carry was shown to agree with it to
# <2e-15 across [0, 60] for l ≤ 4, so the safer pair is used everywhere.
SERIES_X = 4.0
SERIES_TERMS = 40
