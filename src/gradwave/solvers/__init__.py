"""Iterative eigensolvers and preconditioners for the plane-wave Hamiltonian.

``davidson``/``davidson_batched`` (generalized to the USPP metric) and
``chebyshev_filtered_batched`` share the ``*_ms`` mixed-precision wrappers;
``teter``/``teter_b`` are the Teter-Payne-Allan preconditioners they use.
"""

from gradwave.solvers.chebyshev import (
    chebyshev_filtered_batched,
    chebyshev_filtered_batched_ms,
)
from gradwave.solvers.davidson import (
    BatchedDavidsonResult,
    DavidsonResult,
    davidson,
    davidson_batched,
    davidson_batched_ms,
)
from gradwave.solvers.precond import teter, teter_b

__all__ = [
    "BatchedDavidsonResult",
    "DavidsonResult",
    "chebyshev_filtered_batched",
    "chebyshev_filtered_batched_ms",
    "davidson",
    "davidson_batched",
    "davidson_batched_ms",
    "teter",
    "teter_b",
]
