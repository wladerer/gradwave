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
    fig = viz.field_isosurface(drho, cell, positions, symbols)  # signed grid field
    fig.write_image("response.png")                  # e.g. a composition dρ/dλ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go  # ty: ignore[unresolved-import]  # optional viz extra


def _pmv():
    """Import pymatviz lazily, with an actionable error if the extra is missing."""
    try:
        import pymatviz  # ty: ignore[unresolved-import]  # optional viz extra
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gradwave.io.viz needs pymatviz (uv pip install gradwave[viz])"
        ) from err
    return pymatviz


def _go():
    """Import plotly.graph_objects lazily (bundled with the ``viz`` extra)."""
    try:
        import plotly.graph_objects as go  # ty: ignore[unresolved-import]  # optional viz extra
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gradwave.io.viz needs plotly (uv pip install gradwave[viz])"
        ) from err
    return go


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
    source: str | Path | dict[str, Any],
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


def field_isosurface(
    field: Any,
    cell: Any,
    positions: Any,
    symbols: list[str] | None = None,
    *,
    iso_frac: float = 0.3,
    signed: bool = True,
    opacity: float = 0.5,
    pos_color: str = "#d62728",
    neg_color: str = "#1f77b4",
    atom_color: str = "#222222",
    **layout: Any,
) -> go.Figure:
    """Isosurface of a real-space grid field over a crystal cell, as a Plotly figure.

    ``field`` is a real ``(n1, n2, n3)`` array sampled on the fractional grid of
    ``cell`` (lattice rows a_i in Å, e.g. ``system.grid.cell``); ``positions`` are
    the Cartesian atom coordinates ``(na, 3)`` in Å; ``symbols`` are optional
    per-atom element labels for the markers.

    For a SIGNED field — a density response ``dρ/dλ``, a deformation density, a
    frontier-orbital difference — the positive and negative lobes are drawn as two
    isosurfaces (charge flowing in vs out) at ``±iso_frac·max|field|``, which is
    what makes a composition response legible: the screening cloud that the frozen
    (first-order) picture omits. A magnitude field (``signed=False``) draws a
    single surface at ``iso_frac·max(field)``. Atoms are overlaid as markers.

    Returns a rotatable Plotly ``go.Figure``; ``fig.write_image(path)`` writes a
    static PNG (needs kaleido + a browser). Extra keyword args pass to
    ``fig.update_layout`` (``title``, ``width``, ...).
    """
    go = _go()
    field = np.asarray(field, dtype=float)
    if field.ndim != 3:
        raise ValueError(f"field must be a 3-D grid, got shape {field.shape}")
    n1, n2, n3 = field.shape
    cell = np.asarray(cell, dtype=float)

    # Cartesian coordinate of each grid point: fractional (i/n_i) @ cell. Handles
    # a non-orthogonal cell (the coords carry the skew), matching the SCF grid.
    fa, fb, fc = (np.arange(n) / n for n in (n1, n2, n3))
    grid = np.meshgrid(fa, fb, fc, indexing="ij")
    xyz = np.stack([g.ravel() for g in grid], axis=1) @ cell
    x, y, z = xyz.T
    val = field.ravel()
    amax = float(np.abs(val).max()) or 1.0
    iso = iso_frac * amax
    caps = {"x_show": False, "y_show": False, "z_show": False}

    data = []
    if signed:
        data.append(go.Isosurface(
            x=x, y=y, z=z, value=val, isomin=iso, isomax=amax, surface_count=1,
            opacity=opacity, showscale=False, caps=caps, name="δ > 0",
            colorscale=[[0, pos_color], [1, pos_color]]))
        data.append(go.Isosurface(
            x=x, y=y, z=z, value=val, isomin=-amax, isomax=-iso, surface_count=1,
            opacity=opacity, showscale=False, caps=caps, name="δ < 0",
            colorscale=[[0, neg_color], [1, neg_color]]))
    else:
        data.append(go.Isosurface(
            x=x, y=y, z=z, value=np.abs(val), isomin=iso, isomax=amax,
            surface_count=1, opacity=opacity, showscale=False, caps=caps,
            name="field"))

    if positions is not None:
        px, py, pz = np.asarray(positions, dtype=float).T
        data.append(go.Scatter3d(
            x=px, y=py, z=pz, mode="markers+text" if symbols else "markers",
            marker={"size": 5, "color": atom_color}, text=symbols,
            textposition="top center", name="atoms"))

    fig = go.Figure(data=data)
    fig.update_layout(scene={"aspectmode": "data"}, **layout)
    return fig
