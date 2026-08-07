"""The KB projector form-factor cache (pseudo/kb.py): a cell-independent cubic
spline of F_i(q), built once per pseudopotential and interpolated at each cell's
|k+G| — the variable-cell-relax speedup. These guard that it stays a *speedup*,
not a visible approximation, and that it actually reuses the table across cells.
"""

from pathlib import Path

import numpy as np

from gradwave.api import _load_upf
from gradwave.pseudo.kb import (
    _BETA_FF_CACHE,
    _beta_form_factors_exact,
    beta_form_factors,
)

FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
SI = FIX / "Si_ONCV_PBE-1.2.upf"


def test_cached_form_factors_match_direct_transform():
    """The spline table reproduces the direct SBT to ~1e-9 across a dense q-grid
    deliberately off the table nodes — below every SCF tolerance."""
    upf = _load_upf(SI)
    _BETA_FF_CACHE.clear()
    q = np.linspace(0.005, 14.3, 5000)  # off the 0.02-spaced table nodes
    exact = _beta_form_factors_exact(upf, q)
    cached = beta_form_factors(upf, q)
    assert cached.shape == exact.shape == (upf.n_proj, q.size)
    assert np.abs(cached - exact).max() < 1e-8


def test_form_factor_table_reused_across_cells():
    """A second call within the covered q-range reuses the same spline objects
    (the actual cross-cell speedup) rather than rebuilding; a larger q_max does
    rebuild. Keyed on id(upf), so it survives the per-cell setup_system calls."""
    upf = _load_upf(SI)
    _BETA_FF_CACHE.clear()
    beta_form_factors(upf, np.linspace(0.01, 10.0, 200))
    assert id(upf) in _BETA_FF_CACHE
    splines = _BETA_FF_CACHE[id(upf)][1]

    beta_form_factors(upf, np.linspace(0.01, 7.0, 80))   # smaller range
    assert _BETA_FF_CACHE[id(upf)][1] is splines            # no rebuild

    beta_form_factors(upf, np.linspace(0.01, 25.0, 80))  # exceeds coverage
    assert _BETA_FF_CACHE[id(upf)][1] is not splines        # rebuilt larger


def test_empty_q_is_handled():
    upf = _load_upf(SI)
    out = beta_form_factors(upf, np.array([]))
    assert out.shape == (upf.n_proj, 0)
