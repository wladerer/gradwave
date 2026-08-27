"""Render a :class:`~gradwave.profiling.harness.ProfileResult` to a self-contained
HTML report — the glanceable tier of the two-tier harness.

``summary.html`` is ONE offline file: a provenance header, the headline
wall/RSS (from the unprofiled timed loop), matplotlib plots rendered to inline
SVG strings (no JS charting library, no CDN, no external files), the top-N op
table (collapsible, sorted by ~20 lines of inline vanilla JS), and the inlined
py-spy flamegraph. The Perfetto trace, memray HTML, and raw op table
(parquet/json) are LINKED, not inlined.

``compare.html`` (via :func:`compare` / :func:`build_compare_html`) puts two
commits' op tables side by side with a color-coded per-op delta, keyed by the
SHA in each output directory's ``profile.json`` — the diffing entry point.
"""

from __future__ import annotations

import html
import json
from io import StringIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# matplotlib → inline SVG
# ---------------------------------------------------------------------------


def _svg(fig: Any) -> str:
    """A matplotlib Figure → a bare ``<svg>…</svg>`` string (XML header stripped)
    ready to drop into the HTML body. Closes the figure to free it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buf.getvalue()
    i = text.find("<svg")
    return text[i:] if i != -1 else text


def _phase_bar_svg(phase_times_s: dict[str, float]) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in phase_times_s.items() if v > 0]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylabel("self CPU time (s)")
    ax.set_title("Per-phase time (torch.profiler op grouping)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    return _svg(fig)


def _rss_line_svg(iterations: list[dict[str, Any]]) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [it.get("iter") for it in iterations]
    ys = [it.get("rss_mb") for it in iterations]
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False)
             if isinstance(x, int | float) and isinstance(y, int | float)]
    if len(pairs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "-o", color="#F58518", ms=3)
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("RSS (MB)")
    ax.set_title("Resident memory vs iteration")
    return _svg(fig)


def _topn_bar_svg(op_rows: list[dict[str, Any]], n: int = 12) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(op_rows, key=lambda r: r.get("self_cpu_time_us", 0.0), reverse=True)
    rows = [r for r in rows if r.get("self_cpu_time_us", 0.0) > 0][:n]
    if not rows:
        return None
    labels = [r["name"][:34] for r in rows][::-1]
    vals = [r["self_cpu_time_us"] / 1e3 for r in rows][::-1]  # ms
    fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.34 * len(rows) + 0.8)))
    ax.barh(labels, vals, color="#54A24B")
    ax.set_xlabel("self CPU time (ms)")
    ax.set_title(f"Top {len(rows)} ops by self CPU time")
    return _svg(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 margin:0 auto;max-width:900px;padding:24px;color:#1a1a1a;background:#fff}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px;
 border-bottom:1px solid #e5e5e5;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #eee}
th{cursor:pointer;background:#f6f8fa;user-select:none}
.kv td:first-child{color:#555;width:38%}
.metric{display:inline-block;margin:0 24px 8px 0}
.metric b{font-size:24px;display:block}.metric span{color:#555;font-size:12px}
.plot{margin:12px 0;overflow-x:auto}.plot svg{max-width:100%;height:auto}
.notes{background:#fffbe6;border:1px solid #f0e0a0;padding:8px 12px;border-radius:6px;
 font-size:13px;color:#664d03}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}
a{color:#0969da}.dirty{color:#b00;font-weight:700}
.reg{background:#ffe0e0}.imp{background:#e0ffe0}.mono{font-family:ui-monospace,monospace}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px}
"""

# ~20 lines of dependency-free column sorting for the op table.
_SORT_JS = """
function sortTable(t,c){var tb=t.tBodies[0],rows=Array.prototype.slice.call(tb.rows);
var asc=t.getAttribute('data-sc')==c+'a';var nk=t.getAttribute('data-sc')==c+'a'?c+'d':c+'a';
t.setAttribute('data-sc',nk);
rows.sort(function(x,y){var a=x.cells[c].getAttribute('data-v'),b=y.cells[c].getAttribute('data-v');
var fa=parseFloat(a),fb=parseFloat(b);
if(!isNaN(fa)&&!isNaN(fb)){a=fa;b=fb;}else{a=a||'';b=b||'';}
return (a<b?-1:a>b?1:0)*(asc?-1:1);});
rows.forEach(function(r){tb.appendChild(r);});}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _provenance_table(prov: dict[str, Any], spec: dict[str, Any]) -> str:
    code = prov.get("code", {})
    host = prov.get("host", {})
    cpu = prov.get("cpu", {})
    dirty = code.get("git_dirty")
    sha_full = code.get("git_full") or code.get("git") or "?"
    dirty_html = (' <span class="dirty">DIRTY (uncommitted changes)</span>'
                  if dirty else " (clean)")
    rows: list[tuple[str, str]] = [
        ("git commit", f'<span class="mono">{_esc(sha_full)}</span>{dirty_html}'),
        ("hostname", _esc(host.get("hostname"))),
        ("os / arch", f"{_esc(host.get('os'))} / {_esc(host.get('arch'))}"),
        ("gradwave", _esc(code.get("gradwave"))),
        ("python / torch", f"{_esc(code.get('python'))} / {_esc(code.get('torch'))}"),
        ("CPU", _esc(cpu.get("model"))),
        ("cores / torch threads",
         f"{_esc(cpu.get('logical_cores'))} / {_esc(cpu.get('torch_threads'))}"),
        ("timestamp", _esc(prov.get("timestamp"))),
    ]
    gpu = prov.get("gpu")
    if gpu:
        rows.append(("GPU", _esc(gpu.get("name"))))
    # workload spec
    for key in ("system", "hardness", "n_atoms", "ecut_ry", "kmesh", "nbands",
                "smearing", "profiled"):
        if key in spec:
            rows.append((f"workload · {key}", _esc(spec.get(key))))
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv">{body}</table>'


def _headline_html(headline: dict[str, Any]) -> str:
    wall = headline.get("wall_median_s")
    rss = headline.get("peak_rss_gb")
    n = headline.get("n_timed")
    warm = headline.get("warmup_discarded")
    thr = headline.get("torch_threads")
    return (
        f'<div class="metric"><b>{_esc(wall)} s</b>'
        f'<span>median wall ({_esc(n)} runs, {_esc(warm)} warmup discarded)</span></div>'
        f'<div class="metric"><b>{_esc(rss)} GB</b><span>peak RSS</span></div>'
        f'<div class="metric"><b>{_esc(thr)}</b><span>torch threads (pinned)</span></div>'
        f'<p class="notes">{_esc(headline.get("note"))}</p>'
    )


def _op_table_html(op_rows: list[dict[str, Any]], limit: int = 25) -> str:
    cols = [("name", "op", False), ("count", "count", True),
            ("self_cpu_time_us", "self CPU (µs)", True),
            ("cpu_time_us", "CPU total (µs)", True),
            ("self_cuda_time_us", "self CUDA (µs)", True),
            ("self_cpu_mem_bytes", "self CPU mem (B)", True)]
    rows = sorted(op_rows, key=lambda r: r.get("self_cpu_time_us", 0.0),
                  reverse=True)[:limit]
    head = "".join(
        f'<th onclick="sortTable(this.closest(\'table\'),{i})">{_esc(lbl)}</th>'
        for i, (_, lbl, _num) in enumerate(cols))
    body_rows = []
    for r in rows:
        cells = []
        for key, _lbl, num in cols:
            v = r.get(key)
            disp = f"{v:.0f}" if (num and isinstance(v, int | float)) else _esc(v)
            cells.append(f'<td data-v="{_esc(v)}">{disp}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table data-sc=""><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table>')


def _links_html(result_dir: Path, trace_path: str | None,
                memray_html: str | None) -> str:
    links: list[str] = []
    if trace_path:
        name = Path(trace_path).name
        links.append(
            f'<li><a href="{_esc(name)}">{_esc(name)}</a> — Perfetto timeline; '
            "open at <span class=mono>ui.perfetto.dev</span> or "
            "<span class=mono>chrome://tracing</span></li>")
    if memray_html:
        links.append(f'<li><a href="{_esc(memray_html)}">{_esc(memray_html)}</a>'
                     " — memray allocation flamegraph (self-contained)</li>")
    for fn in ("op_table.parquet", "op_table.json", "profile.json"):
        if (result_dir / fn).exists():
            links.append(f'<li><a href="{_esc(fn)}">{_esc(fn)}</a></li>')
    if not links:
        return ""
    return "<ul>" + "".join(links) + "</ul>"


def build_summary_html(result: Any) -> str:
    """Render a :class:`~gradwave.profiling.harness.ProfileResult` to the
    self-contained ``summary.html`` string."""
    prov = result.provenance
    spec = result.workload.get("spec", {})
    out_dir = Path(result.out_dir)

    parts: list[str] = [
        f"<style>{_CSS}</style>",
        f"<script>{_SORT_JS}</script>",
        f"<h1>gradwave profile · {_esc(result.workload.get('name'))}</h1>",
        "<h2>Provenance</h2>", _provenance_table(prov, spec),
        "<h2>Headline metrics</h2>", _headline_html(result.headline),
    ]

    if result.notes:
        parts.append('<p class="notes">Notes: ' +
                     "<br>".join(_esc(n) for n in result.notes) + "</p>")

    parts.append("<h2>Plots</h2>")
    for svg in (_phase_bar_svg(result.phase_times_s),
                _rss_line_svg(result.iterations),
                _topn_bar_svg(result.op_rows)):
        if svg:
            parts.append(f'<div class="plot">{svg}</div>')

    parts.append("<h2>Top ops (torch.profiler)</h2>")
    parts.append("<details open><summary>op table (click a header to sort)</summary>"
                 + _op_table_html(result.op_rows) + "</details>")

    if result.flame_svg:
        parts.append("<h2>py-spy flamegraph (Python-dispatch / eager-glue view)</h2>")
        parts.append('<details><summary>flamegraph (click to expand)</summary>'
                     f'<div class="plot">{result.flame_svg}</div></details>')

    links = _links_html(out_dir, result.trace_path, result.memray_html)
    if links:
        parts.append("<h2>Deep artifacts (linked)</h2>")
        parts.append(links)

    parts.append("<details><summary>raw torch.profiler table</summary>"
                 f"<pre>{_esc(result.op_table_str)}</pre></details>")

    return "\n".join(parts)


def write_summary(result: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``summary.html`` next to the result's artifacts. Returns its path."""
    d = Path(out_dir) if out_dir is not None else Path(result.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "summary.html"
    path.write_text(build_summary_html(result))
    return path


# ---------------------------------------------------------------------------
# Cross-commit comparison (the diffing entry point)
# ---------------------------------------------------------------------------


def _load_ops(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a profile output dir's provenance + op rows (from op_table.json,
    falling back to profile.json's embedded op_rows)."""
    d = Path(run_dir)
    prof: dict[str, Any] = {}
    pj = d / "profile.json"
    if pj.exists():
        prof = json.loads(pj.read_text())
    ops: list[dict[str, Any]] = []
    oj = d / "op_table.json"
    if oj.exists():
        ops = json.loads(oj.read_text())
    elif prof.get("op_rows"):
        ops = prof["op_rows"]
    return prof, ops


def compare(dir_a: str | Path, dir_b: str | Path,
            metric: str = "self_cpu_time_us") -> list[dict[str, Any]]:
    """Signed per-op delta between two profile output dirs (two commits).

    Follows :mod:`gradwave.bench.analyze`'s "return a plain list of row dicts"
    style. Each row: op name, ``a``/``b`` metric values, absolute + percent
    delta (b − a; positive = slower/heavier on b = a regression). Ops present in
    only one side appear with the missing value 0. Sorted by absolute delta."""
    _pa, ops_a = _load_ops(dir_a)
    _pb, ops_b = _load_ops(dir_b)
    map_a = {r["name"]: float(r.get(metric, 0.0)) for r in ops_a}
    map_b = {r["name"]: float(r.get(metric, 0.0)) for r in ops_b}
    rows: list[dict[str, Any]] = []
    for name in sorted(set(map_a) | set(map_b)):
        a = map_a.get(name, 0.0)
        b = map_b.get(name, 0.0)
        delta = b - a
        pct = (delta / a * 100.0) if a > 0 else (float("inf") if b > 0 else 0.0)
        rows.append({"name": name, "a": a, "b": b, "delta": delta, "pct": pct})
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows


def build_compare_html(dir_a: str | Path, dir_b: str | Path,
                       metric: str = "self_cpu_time_us", limit: int = 40) -> str:
    """Side-by-side per-op delta table for two commits, color-coded
    (red = regression / slower on B, green = improvement)."""
    prof_a, _ = _load_ops(dir_a)
    prof_b, _ = _load_ops(dir_b)
    rows = compare(dir_a, dir_b, metric=metric)

    def _sha(p: dict[str, Any]) -> str:
        code = p.get("provenance", {}).get("code", {})
        sha = str(code.get("git") or "?")
        return sha + ("-dirty" if code.get("git_dirty") else "")

    sha_a, sha_b = _sha(prof_a), _sha(prof_b)
    body = []
    for r in rows[:limit]:
        cls = "reg" if r["delta"] > 0 else ("imp" if r["delta"] < 0 else "")
        pct = "∞" if r["pct"] == float("inf") else f"{r['pct']:+.1f}%"
        body.append(
            f'<tr class="{cls}"><td class=mono>{_esc(r["name"])}</td>'
            f'<td>{r["a"]:.0f}</td><td>{r["b"]:.0f}</td>'
            f'<td>{r["delta"]:+.0f}</td><td>{_esc(pct)}</td></tr>')
    return "\n".join([
        f"<style>{_CSS}</style>",
        "<h1>gradwave profile comparison</h1>",
        f'<p>metric: <span class=mono>{_esc(metric)}</span> · '
        f'A = <span class=mono>{_esc(sha_a)}</span> · '
        f'B = <span class=mono>{_esc(sha_b)}</span> · '
        "positive delta = slower/heavier on B (regression, red)</p>",
        "<table><thead><tr><th>op</th>"
        f"<th>A ({_esc(sha_a)})</th><th>B ({_esc(sha_b)})</th>"
        "<th>Δ</th><th>Δ%</th></tr></thead><tbody>",
        "\n".join(body), "</tbody></table>",
    ])


def write_compare(dir_a: str | Path, dir_b: str | Path,
                  out_path: str | Path) -> Path:
    """Write ``compare.html`` for two profile dirs. Returns its path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_compare_html(dir_a, dir_b))
    return path
