"""Small shared vector helpers for the post-SCF magnetism modules.

``mae`` and ``spin_exchange`` both normalize direction vectors the same way;
the helper lives here so it is defined once and both import it as their
module-private ``_unit``.
"""

from __future__ import annotations

import torch


def unit(v: torch.Tensor) -> torch.Tensor:
    """The unit vector along ``v`` (``v / ‖v‖``)."""
    return v / torch.linalg.norm(v)
