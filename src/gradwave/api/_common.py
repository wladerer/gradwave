"""Shared XC registries, numeric defaults, and result duck-typing helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE, SpinXC
from gradwave.inputs import Input

if TYPE_CHECKING:

    from gradwave.scf.loop import SCFResult
    from gradwave.scf.noncollinear import NCResult
    from gradwave.scf.results import USPPNCResult, USPPResult

    # every SCF driver's converged-state result, the type `_get()`/
    # `build_summary()` duck-type over via getattr (field sets differ; see
    # `_get`'s docstring) rather than isinstance branching.
    SCFLike = SCFResult | NCResult | USPPResult | USPPNCResult

logger = logging.getLogger(__name__)


XC_REGISTRY: dict[str, type[XCFunctional]] = {"lda": LDA_PW92, "pbe": PBE,
                                              "r2scan": R2SCAN}


SPIN_XC_REGISTRY: dict[str, type[SpinXC]] = {"lda": LSDA_PW92, "pbe": SpinPBE,
                                             "r2scan": SpinR2SCAN}


_OCC_TOL = 1e-6


# the collinear/NC solvers build a fixed-length Pulay history and need an int
# (None is not accepted, unlike the USPP path); this names the default the api
# forwards, matching the per-scheme default those solvers use internally
_DEFAULT_MIXING_HISTORY = 8


def _mixing_scheme(inp: Input) -> str | None:
    """The mixing scheme the api forwards to the SCF driver. The input default
    ``auto`` maps to None so each formalism's own resolver picks its
    evidence-backed default (johnson for USPP/PAW and for collinear-spin nspin=2
    norm-conserving, pulay otherwise); an explicit pulay|broyden|johnson passes
    through unchanged. Shared by run_scf and _build_relax_calc so both entry
    points defer to the same resolvers. See scf.loop._resolve_mixing_scheme and
    scf.uspp_loop._resolve_uspp_mixing_scheme."""
    scheme = inp.scf.mixing.scheme
    return None if scheme == "auto" else scheme


def _get(res: SCFLike, key: str, default: Any = None) -> Any:
    """Attribute read with a default: every SCF driver returns a result
    dataclass, but the field sets differ (e.g. NCResult has no nspin)."""
    return getattr(res, key, default)


def _gap(eigenvalues: Any, occupations: Any, nspin: int) -> float | None:
    """HOMO-LUMO gap over all k and spins, None when any occupation is
    fractional (metals/smeared systems have no meaningful scalar gap)."""
    import numpy as np

    e = np.asarray(eigenvalues, dtype=float).reshape(-1)
    f = np.asarray(occupations, dtype=float).reshape(-1)
    f_full = 2.0 / nspin
    frac = (f > _OCC_TOL) & (np.abs(f - f_full) > _OCC_TOL)
    if frac.any() or not (f > _OCC_TOL).any() or not (f <= _OCC_TOL).any():
        return None
    homo = e[f > _OCC_TOL].max()
    lumo = e[f <= _OCC_TOL].min()
    return float(lumo - homo) if lumo > homo else 0.0
