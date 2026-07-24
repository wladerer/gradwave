"""Generate a self-contained HTML dashboard for the gradwave fleet.

Two panels, no JavaScript, no external assets — a single static file you can
open locally or drop somewhere to serve:

  1. Live queue: `pueue status --json` from each host (reusing gwq_status's
     collectors), grouped and coloured by state.
  2. Benchmark history: the JSON records `benchmarks/_capture.py` writes under
     benchmarks/results/<host>/ — most recent runs per benchmark, with a tiny
     inline-SVG wall-time sparkline when there are enough points to show a trend.

Usage:
  python scripts/dashboard.py                       # -> dashboard.html (this host + asus)
  python scripts/dashboard.py --hosts thinkpad asus --out /tmp/dash.html
  python scripts/dashboard.py --collect             # rsync remote results in first

Hosting (no rebuild needed): generate on a box that can reach the fleet, rsync
the single file to homelab, and serve it with `tailscale serve`. See
docs/queue.md.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the queue collectors so the dashboard and `gwq status` never diverge.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwq_status as qs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"

STATE_CLASS = {
    "Running": "run", "Queued": "queue", "Paused": "pause",
    "Done": "done", "Stashed": "stash",
}


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #

def collect_results(hosts: list[str]) -> None:
    """rsync each remote host's results dir into the local tree (best-effort)."""
    for host in hosts:
        if host == qs.this_host():
            continue
        dest = RESULTS
        dest.mkdir(parents=True, exist_ok=True)
        src = f"{host}:~/github/gradwave/benchmarks/results/"
        try:
            subprocess.run(["rsync", "-az", "--timeout=15", src, str(dest) + "/"],
                           capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            pass  # host asleep / unreachable — just show what we have


def load_results() -> dict[str, list[dict]]:
    """All benchmark records grouped by benchmark name, newest first."""
    by_name: dict[str, list[dict]] = {}
    if not RESULTS.exists():
        return by_name
    for path in RESULTS.rglob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # display host: fold asus's OS hostname back to the logical name
        rec["_host"] = qs.HOST_ALIASES.get(rec.get("host", "?"), rec.get("host", "?"))
        by_name.setdefault(rec.get("name", "?"), []).append(rec)
    for recs in by_name.values():
        recs.sort(key=lambda r: r.get("started_utc", ""), reverse=True)
    return by_name


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def esc(x) -> str:
    return html.escape(str(x))


def sparkline(vals: list[float], w: int = 120, h: int = 24) -> str:
    """Minimal inline-SVG sparkline of wall times (oldest -> newest)."""
    if len(vals) < 3:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(
        f"{i / (n - 1) * (w - 2) + 1:.1f},"
        f"{h - 1 - (v - lo) / span * (h - 2):.1f}"
        for i, v in enumerate(vals)
    )
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{pts}" fill="none" stroke="currentColor" '
            f'stroke-width="1.5"/></svg>')


def render_queue(host: str, data: dict | None) -> str:
    if data is None:
        return (f'<section class="host"><h2>{esc(host)} '
                f'<span class="muted">— unreachable</span></h2></section>')
    tasks = data.get("tasks", {}) or {}
    groups = data.get("groups", {}) or {}

    gtags = []
    for gname, g in sorted(groups.items()):
        par = g.get("parallel_tasks", g.get("parallel", "?")) if isinstance(g, dict) else "?"
        st = g.get("status", "") if isinstance(g, dict) else ""
        paused = " (paused)" if str(st).lower() == "paused" else ""
        gtags.append(f'<span class="tag">{esc(gname)}={esc(par)}{paused}</span>')

    rows = []
    for tid, t in tasks.items():
        kind, result = qs._status_kind(t.get("status"))
        rows.append((qs.ORDER.get(kind, 9),
                     int(tid) if str(tid).isdigit() else 0,
                     tid, t.get("group", "?"), kind, result, t))
    rows.sort(key=lambda r: (r[0], -r[1] if r[0] == 4 else r[1]))

    trs, done_shown = [], 0
    for bucket, _i, tid, group, kind, result, t in rows:
        if bucket == 4:
            if done_shown >= 8:
                continue
            done_shown += 1
        state = kind if not result else f"{kind}:{result}"
        cls = STATE_CLASS.get(kind, "")
        if result and result != "Success":
            cls = "fail"
        label = t.get("label") or (t.get("command", "")[:70])
        trs.append(
            f'<tr><td class="num">{esc(tid)}</td><td>{esc(group)}</td>'
            f'<td><span class="state {cls}">{esc(state)}</span></td>'
            f'<td class="num">{esc(qs._elapsed(t, kind))}</td>'
            f'<td class="cmd">{esc(label)}</td></tr>')

    body = ("".join(trs) if trs
            else '<tr><td colspan="5" class="muted">queue empty</td></tr>')
    return (f'<section class="host"><h2>{esc(host)}</h2>'
            f'<div class="tags">{"".join(gtags)}</div>'
            f'<table><thead><tr><th>id</th><th>group</th><th>state</th>'
            f'<th>elapsed</th><th>label / command</th></tr></thead>'
            f'<tbody>{body}</tbody></table></section>')


def render_benchmarks(by_name: dict[str, list[dict]]) -> str:
    if not by_name:
        return ('<section class="bench"><h2>Benchmarks</h2>'
                '<p class="muted">No benchmark records yet. Run '
                '<code>gwq bench &lt;name&gt; …</code> — results land in '
                'benchmarks/results/&lt;host&gt;/.</p></section>')
    blocks = []
    for name in sorted(by_name):
        recs = by_name[name]
        # sparkline over chronological wall times (all hosts pooled)
        chrono = sorted(recs, key=lambda r: r.get("started_utc", ""))
        spark = sparkline([float(r.get("wall_s", 0) or 0) for r in chrono])
        trs = []
        for r in recs[:8]:
            ok = r.get("status") == "ok"
            when = (r.get("started_utc", "") or "")[:16].replace("T", " ")
            trs.append(
                f'<tr><td>{esc(r["_host"])}</td>'
                f'<td class="num">{esc(when)}</td>'
                f'<td class="mono">{esc(r.get("git_sha", "?"))}</td>'
                f'<td class="num">{esc(r.get("wall_s", "?"))}s</td>'
                f'<td><span class="state {"done" if ok else "fail"}">'
                f'{esc(r.get("status", "?"))}</span></td>'
                f'<td class="cmd">{esc(r.get("reported", ""))}</td></tr>')
        blocks.append(
            f'<div class="benchblock"><h3>{esc(name)} '
            f'<span class="spark-wrap">{spark}</span></h3>'
            f'<table><thead><tr><th>host</th><th>when (UTC)</th><th>sha</th>'
            f'<th>wall</th><th>status</th><th>reported</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')
    return f'<section class="bench"><h2>Benchmarks</h2>{"".join(blocks)}</section>'


CSS = """
:root{color-scheme:dark light}
*{box-sizing:border-box}
body{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;
  padding:1.5rem;background:#14161a;color:#e6e6e6}
h1{font-size:1.15rem;margin:0 0 .2rem}
h2{font-size:1rem;margin:1.4rem 0 .4rem;border-bottom:1px solid #2c2f36;padding-bottom:.25rem}
h3{font-size:.9rem;margin:1rem 0 .3rem;display:flex;align-items:center;gap:.6rem}
.muted{color:#8b909a}
.sub{color:#8b909a;margin:0 0 1rem;font-size:.85rem}
table{border-collapse:collapse;width:100%;max-width:100%;margin:.2rem 0 .6rem}
th,td{text-align:left;padding:.22rem .6rem;border-bottom:1px solid #23262c;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
th{color:#8b909a;font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:inherit;color:#b7c6ff}
.cmd{max-width:0;width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tags{margin:.1rem 0 .3rem}
.tag{display:inline-block;background:#22262e;border-radius:4px;padding:.05rem .45rem;
  margin:.1rem .3rem .1rem 0;color:#b6bcc6;font-size:.8rem}
.state{border-radius:4px;padding:.02rem .4rem;font-size:.8rem}
.state.run{background:#1d3b2a;color:#7ee2a8}
.state.queue{background:#333a1d;color:#d9e27e}
.state.pause{background:#2a2a2a;color:#b0b0b0}
.state.done{background:#1d2b3b;color:#8ab6e2}
.state.fail{background:#3b1d1d;color:#e28a8a}
.spark{color:#7aa2ff;vertical-align:middle}
.wrap{max-width:1100px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}
code{background:#22262e;padding:.05rem .3rem;border-radius:3px}
"""


def build_html(hosts: list[str], queues: dict, results: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    qsec = '<div class="grid">' + "".join(
        render_queue(h, queues.get(h)) for h in hosts) + "</div>"
    bsec = render_benchmarks(results)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>gradwave fleet</title><style>{CSS}</style></head><body>"
            f"<div class='wrap'><h1>gradwave fleet</h1>"
            f"<p class='sub'>generated {esc(now)} · hosts: {esc(', '.join(hosts))}</p>"
            f"<h2>Queue</h2>{qsec}{bsec}</div></body></html>")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="generate the gradwave fleet dashboard")
    p.add_argument("--hosts", nargs="*", default=[qs.this_host(), "asus"])
    p.add_argument("--out", default=str(REPO / "dashboard.html"))
    p.add_argument("--collect", action="store_true",
                   help="rsync remote hosts' benchmark results in before rendering")
    a = p.parse_args(argv)
    # de-dup while preserving order (this_host may equal a listed host)
    hosts = list(dict.fromkeys(a.hosts))

    if a.collect:
        collect_results(hosts)
    queues = {h: qs.fetch(h) for h in hosts}
    results = load_results()
    Path(a.out).write_text(build_html(hosts, queues, results))
    print(f"wrote {a.out}  ({sum(1 for v in queues.values() if v)}/{len(hosts)} hosts, "
          f"{sum(len(v) for v in results.values())} bench records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
