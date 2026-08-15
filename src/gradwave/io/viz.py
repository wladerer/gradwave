"""pymatviz structure and periodic-table visuals over gradwave objects (Layer C).

pymatviz is imported lazily behind the ``viz`` extra, so the core package and the
CLI run without it. Install with ``uv pip install gradwave[viz]`` (it pulls plotly
and pymatgen, so it is deliberately kept out of the core). Every helper returns a
Plotly ``go.Figure``, so results render inline in Jupyter or Marimo and export to
a static image with ``fig.write_image(path)`` (which needs kaleido).

    from gradwave.io import viz
    viz.structure_view(atoms).show()                 # 3D structure, force/magmom
                                                     # vectors drawn when present
    fig = viz.ptable_delta("results/delta_summary.json")
    fig.write_image("delta_ptable.png")              # per-element Δ heatmap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _pmv():
    """Import pymatviz lazily, with an actionable error if the extra is missing."""
    try:
        import pymatviz
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gradwave.io.viz needs pymatviz (uv pip install gradwave[viz])"
        ) from err
    return pymatviz


def _resolve_struct(obj: Any) -> Any:
    """Return something pymatviz can plot from a gradwave object.

    Accepts an ASE ``Atoms`` directly, a gradwave ``Input`` or a ``GradWave``
    calculator (both carry ``.atoms``), or anything pymatviz already understands
    (a pymatgen ``Structure``, a dict, or a sequence of these).
    """
    from ase import Atoms

    if isinstance(obj, Atoms):
        return obj
    if hasattr(obj, "atoms"):
        atoms = obj.atoms
        if isinstance(atoms, Atoms):
            return atoms
        raise TypeError(
            f"{type(obj).__name__}.atoms is not an ASE Atoms "
            f"(got {type(atoms).__name__}). Pass an Atoms, a gradwave Input, or a "
            "GradWave calculator that has run."
        )
    return obj  # pymatgen Structure / dict / sequence, handled by pymatviz


def structure_view(obj: Any, *, mode: str = "3d", **kwargs: Any) -> go.Figure:
    """Interactive structure figure from an ASE ``Atoms``, gradwave ``Input``, or
    ``GradWave`` calculator.

    ``mode`` is ``"3d"`` (default, rotatable) or ``"2d"``. Force and magmom
    vectors are drawn automatically when the atoms carry them, so this doubles as
    a quick look at a relaxed or magnetic result. Extra keyword arguments pass
    through to pymatviz (``show_bonds``, ``site_labels``, ``elem_colors``, ...).
    Returns a Plotly ``go.Figure``.
    """
    pmv = _pmv()
    struct = _resolve_struct(obj)
    plot = pmv.structure_3d if mode == "3d" else pmv.structure_2d
    return plot(struct, **kwargs)


def ptable_delta(
    source: str | Path | dict,
    *,
    key: str = "delta_wien2k",
    colorscale: str = "viridis",
    fmt: str = ".2f",
    **kwargs: Any,
) -> go.Figure:
    """Periodic-table heatmap of the per-element Δ-gauge.

    ``source`` is a path to a ``delta_summary.json`` (as written under
    ``benchmarks/delta_gauge/results/``) or an already-loaded dict of the same
    shape, ``{element: {"delta_wien2k": ..., "dfact_pdojo": ...}}``. ``key``
    selects the column, ``"delta_wien2k"`` for gradwave against the WIEN2k
    all-electron reference or ``"dfact_pdojo"`` for PseudoDojo's published Δ.
    Extra keyword arguments pass through to ``pymatviz.ptable_heatmap``
    (``cscale_range``, ``log``, ``exclude_elements``, ...). Returns a Plotly
    ``go.Figure``.
    """
    pmv = _pmv()
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    values = {el: rec[key] for el, rec in data.items() if key in rec}
    if not values:
        raise ValueError(f"no element carried the key {key!r} in the Δ summary")
    return pmv.ptable_heatmap(values, colorscale=colorscale, fmt=fmt, **kwargs)
