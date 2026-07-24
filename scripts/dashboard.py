#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard for the gradwave fleet.

Panels (all best-effort — an unreachable host or missing tool degrades that
panel, never the page):

  1. Host health   — load vs cores, memory, GPU (asus), uptime, disk. The
     contention view: is a box busy right now?
  2. Queue         — live pueue tasks per host + 24h throughput (done/failed).
  3. Agents        — active worktrees per host, which have a live agent parked
     inside, drift vs main, and files edited in >1 worktree (conflict incoming).
  4. Benchmarks    — recent runs per benchmark from benchmarks/results/<host>/,
     wall-time sparkline + a regression flag.

No external assets; a validated light/dark palette (see the dataviz skill) is
inlined. The page carries a 60s meta-refresh to match the server-side cadence.

Usage:
  python scripts/dashboard.py                 # -> dashboard.html
  python scripts/dashboard.py --collect --out /tmp/gwdash.html
The asus `gwdash` systemd timer runs it every 60s and pushes to homelab.
"""
# ruff: noqa: E501  (embedded CSS/HTML string literals are intentionally long)

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwq_status as qs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
PYREMOTE = "~/.venvs/base/bin/python3"  # stdlib python present on every box

# host, whether it runs pueue, whether it has a gradwave checkout, whether it has a GPU
FLEET = [
    {"host": "thinkpad", "queue": True, "worktrees": True, "gpu": False},
    {"host": "asus", "queue": True, "worktrees": True, "gpu": True},
    {"host": "homelab", "queue": False, "worktrees": False, "gpu": False},
]

METRICS_SH = r"""
echo "ncpu=$(nproc 2>/dev/null)"
echo "load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)"
awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print "memtotal="t; print "memavail="a}' /proc/meminfo 2>/dev/null
echo "uptime=$(awk '{print int($1)}' /proc/uptime 2>/dev/null)"
echo "diskpct=$(df -P / 2>/dev/null | awk 'NR==2{gsub("%","",$5);print $5}')"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | \
  awk -F', *' '{print "gpu_util="$1; print "gpu_memused="$2; print "gpu_memtotal="$3; print "gpu_temp="$4}'
true  # keep the script's exit status 0 even on no-GPU hosts (nvidia check fails)
"""


# --------------------------------------------------------------------------- #
# collection (all best-effort)
# --------------------------------------------------------------------------- #

def run(argv: list[str], timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def remote(host: str, script: str, timeout: int = 20) -> str | None:
    local = host == qs.this_host()
    if local:
        return run(["bash", "-c", script], timeout)
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, script], timeout)


def host_metrics(host: str) -> dict | None:
    out = remote(host, METRICS_SH, timeout=15)
    if out is None:
        return None
    m: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            m[k.strip()] = v.strip()
    return m or None


def collect_results(hosts: list[str]) -> None:
    for host in hosts:
        if host == qs.this_host():
            continue
        RESULTS.mkdir(parents=True, exist_ok=True)
        run(["rsync", "-az", "--timeout=15", "-e", "ssh -o StrictHostKeyChecking=accept-new",
             f"{host}:~/github/gradwave/benchmarks/results/", str(RESULTS) + "/"], timeout=30)


def load_results() -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    if not RESULTS.exists():
        return by_name
    for path in RESULTS.rglob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rec["_host"] = qs.HOST_ALIASES.get(rec.get("host", "?"), rec.get("host", "?"))
        by_name.setdefault(rec.get("name", "?"), []).append(rec)
    for recs in by_name.values():
        recs.sort(key=lambda r: r.get("started_utc", ""), reverse=True)
    return by_name


def fetch_worktrees(host: str) -> list[dict]:
    script = f"cd ~/github/gradwave 2>/dev/null && {PYREMOTE} scripts/worktrees.py --json 2>/dev/null"
    out = remote(host, script, timeout=25)
    if not out:
        return []
    try:
        rows = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return []
    for r in rows:
        r["_host"] = host
    return [r for r in rows if r.get("is_wt")]


def throughput(data: dict) -> dict:
    """Count done/failed in the last 24h + current running/queued from pueue json."""
    now = datetime.now(timezone.utc)
    done = failed = running = queued = 0
    for t in (data.get("tasks", {}) or {}).values():
        kind, result = qs._status_kind(t.get("status"))
        if kind == "Running":
            running += 1
        elif kind == "Queued":
            queued += 1
        elif kind == "Done":
            body = t.get("status", {})
            end = qs._parse_ts(body.get("Done", {}).get("end") if isinstance(body, dict) else None)
            if end and (now - end).total_seconds() <= 86400:
                if result == "Success":
                    done += 1
                else:
                    failed += 1
    return {"done": done, "failed": failed, "running": running, "queued": queued}


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #

def esc(x) -> str:
    return html.escape(str(x))


def fnum(x, default="?"):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def human_dur(secs: float) -> str:
    secs = int(secs)
    d, secs = divmod(secs, 86400)
    h, secs = divmod(secs, 3600)
    if d:
        return f"{d}d {h}h"
    m = secs // 60
    return f"{h}h {m}m" if h else f"{m}m"


def meter(frac: float, hue: str = "blue") -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return (f'<div class="meter"><div class="fill {hue}" '
            f'style="width:{pct:.0f}%"></div></div>')


def chip(text: str, status: str = "") -> str:
    return f'<span class="chip {status}">{esc(text)}</span>'


def sparkline(vals: list[float], w: int = 128, h: int = 28) -> str:
    if len(vals) < 3:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    def pt(i, v):
        return (i / (n - 1) * (w - 4) + 2, h - 3 - (v - lo) / span * (h - 6))
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, v) for i, v in enumerate(vals)))
    ex, ey = pt(n - 1, vals[-1])
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.5" fill="var(--series-1)"/></svg>')


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #

def render_host_card(spec: dict, m: dict | None) -> str:
    host = spec["host"]
    if m is None:
        return (f'<div class="card"><div class="card-h"><b>{esc(host)}</b>'
                f'<span class="dot muted"></span><span class="muted">unreachable</span>'
                f'</div></div>')
    ncpu = fnum(m.get("ncpu"), 0) or 1
    load = fnum(m.get("load"), 0)
    ratio = load / ncpu if ncpu else 0
    if ratio >= 1.1:
        st, word = "critical", "saturated"
    elif ratio >= 0.7:
        st, word = "warning", "busy"
    else:
        st, word = "good", "idle"

    memtot = fnum(m.get("memtotal"), 0)
    memav = fnum(m.get("memavail"), 0)
    memused = (memtot - memav) if (memtot and memav != "?") else 0
    mem_frac = (memused / memtot) if memtot else 0
    memg = lambda kb: f"{kb / 1048576:.0f}" if isinstance(kb, float) else "?"  # noqa: E731

    rows = [
        f'<div class="stat"><span class="lbl">load</span>'
        f'<span class="val">{load:.2f}<span class="unit"> / {int(ncpu)}</span></span></div>'
        f'{meter(ratio, "blue")}',
        f'<div class="stat"><span class="lbl">mem</span>'
        f'<span class="val">{memg(memused)}<span class="unit"> / {memg(memtot)} GiB</span></span></div>'
        f'{meter(mem_frac, "aqua")}',
    ]
    if spec.get("gpu") and "gpu_util" in m:
        gu = fnum(m.get("gpu_util"), 0)
        gmu, gmt = fnum(m.get("gpu_memused"), 0), fnum(m.get("gpu_memtotal"), 1)
        gt = m.get("gpu_temp", "?")
        rows.append(
            f'<div class="stat"><span class="lbl">gpu</span>'
            f'<span class="val">{gu:.0f}%<span class="unit"> · {gmu:.0f}/{gmt:.0f}MB · {esc(gt)}°C</span>'
            f'</span></div>{meter(gu / 100, "blue")}')

    disk = m.get("diskpct", "?")
    up = human_dur(fnum(m.get("uptime"), 0)) if m.get("uptime") else "?"
    foot = (f'<div class="card-f muted">/ {esc(disk)}% · up {esc(up)}</div>')
    return (f'<div class="card"><div class="card-h"><b>{esc(host)}</b>'
            f'<span class="dot {st}"></span><span class="{st}-t">{word}</span></div>'
            f'{"".join(rows)}{foot}</div>')


def render_queue(spec: dict, data: dict | None) -> str:
    host = spec["host"]
    if data is None:
        return (f'<div class="qcol"><h3>{esc(host)} <span class="muted">— unreachable</span></h3></div>')
    tp = throughput(data)
    tiles = (
        f'<div class="tiles">'
        f'<div class="tile"><span class="tv">{tp["running"]}</span><span class="tl">running</span></div>'
        f'<div class="tile"><span class="tv">{tp["queued"]}</span><span class="tl">queued</span></div>'
        f'<div class="tile"><span class="tv good-t">{tp["done"]}</span><span class="tl">done · 24h</span></div>'
        f'<div class="tile"><span class="tv {"critical-t" if tp["failed"] else "muted"}">{tp["failed"]}</span>'
        f'<span class="tl">failed · 24h</span></div></div>')

    groups = data.get("groups", {}) or {}
    gtags = []
    for g, gd in sorted(groups.items()):
        par = gd.get("parallel_tasks", gd.get("parallel", "?")) if isinstance(gd, dict) else "?"
        paused = isinstance(gd, dict) and str(gd.get("status", "")).lower() == "paused"
        gtags.append(chip(f"{g}={par}" + ("·paused" if paused else ""), "muted" if paused else ""))

    rows = []
    for tid, t in (data.get("tasks", {}) or {}).items():
        kind, result = qs._status_kind(t.get("status"))
        rows.append((qs.ORDER.get(kind, 9), int(tid) if str(tid).isdigit() else 0,
                     tid, t.get("group", "?"), kind, result, t))
    rows.sort(key=lambda r: (r[0], -r[1] if r[0] == 4 else r[1]))
    STMAP = {"Running": "good", "Queued": "warning", "Paused": "muted", "Done": ""}
    trs, shown_done = [], 0
    for bucket, _i, tid, group, kind, result, t in rows:
        if bucket == 4:
            if shown_done >= 5:
                continue
            shown_done += 1
        st = STMAP.get(kind, "")
        if result and result != "Success":
            st, kind = "critical", "Failed"
        label = t.get("label") or t.get("command", "")[:60]
        trs.append(
            f'<tr><td class="num">{esc(tid)}</td><td>{esc(group)}</td>'
            f'<td>{chip(kind, st)}</td><td class="num">{esc(qs._elapsed(t, kind))}</td>'
            f'<td class="cmd">{esc(label)}</td></tr>')
    body = "".join(trs) or '<tr><td colspan="5" class="muted">queue empty</td></tr>'
    return (f'<div class="qcol"><h3>{esc(host)}</h3>{tiles}'
            f'<div class="tags">{"".join(gtags)}</div>'
            f'<table><tbody>{body}</tbody></table></div>')


def render_agents(wts: list[dict]) -> str:
    if not wts:
        return ('<section><h2>Agents &amp; worktrees</h2>'
                '<p class="muted">no worktree data (hosts unreachable or no checkout)</p></section>')
    # fleet-wide file overlap
    owners: dict[str, list[str]] = {}
    for r in wts:
        if r.get("stale"):
            continue
        for f in r.get("files", []):
            owners.setdefault(f, []).append(f'{r["_host"]}/{r["name"]}')
    clashes = {f: w for f, w in owners.items() if len(set(w)) > 1}

    trs = []
    for r in sorted(wts, key=lambda r: (r["_host"], r.get("stale", False), r["name"])):
        if r.get("stale"):
            state, sc = "stale", "muted"
        elif r.get("has_agent"):
            state, sc = "agent live", "good"
        elif int(r.get("dirty", 0) or 0) > 0:
            state, sc = "active", "warning"
        else:
            state, sc = "idle", ""
        ba = f'{r.get("behind","?")}/{r.get("ahead","?")}'
        drift = "warning" if (str(r.get("behind", "0")).isdigit() and int(r["behind"]) >= 10) else ""
        trs.append(
            f'<tr><td>{esc(r["_host"])}</td><td>{esc(r["name"])}</td>'
            f'<td class="cmd">{esc(r.get("branch",""))}</td>'
            f'<td class="num {drift}-t">{esc(ba)}</td>'
            f'<td class="num">{esc(r.get("dirty",0))}</td><td>{chip(state, sc)}</td></tr>')

    overlap = ""
    if clashes:
        items = "".join(
            f'<li><code>{esc(f)}</code> — {esc(", ".join(sorted(set(w))))}</li>'
            for f, w in sorted(clashes.items()))
        overlap = (f'<div class="warnbox"><b class="critical-t">⚠ file overlap across worktrees'
                   f'</b><ul>{items}</ul></div>')
    return (f'<section><h2>Agents &amp; worktrees</h2>{overlap}'
            f'<table><thead><tr><th>host</th><th>worktree</th><th>branch</th>'
            f'<th>b/a</th><th>dirty</th><th>state</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></section>')


def render_benchmarks(by_name: dict[str, list[dict]]) -> str:
    if not by_name:
        return ('<section><h2>Benchmarks</h2><p class="muted">No records yet — '
                'run <code>gwq bench &lt;name&gt; …</code>.</p></section>')
    blocks = []
    for name in sorted(by_name):
        recs = by_name[name]
        chrono = sorted(recs, key=lambda r: r.get("started_utc", ""))
        walls = [float(r.get("wall_s", 0) or 0) for r in chrono]
        flag = ""
        if len(walls) >= 4 and walls[-1] > 1.3 * (median(walls[:-1]) or walls[-1]):
            flag = chip("↑ slower", "warning")
        trs = []
        for r in recs[:6]:
            ok = r.get("status") == "ok"
            when = (r.get("started_utc", "") or "")[:16].replace("T", " ")
            trs.append(
                f'<tr><td>{esc(r["_host"])}</td><td class="num">{esc(when)}</td>'
                f'<td class="mono">{esc(r.get("git_sha","?"))}</td>'
                f'<td class="num">{esc(r.get("wall_s","?"))}s</td>'
                f'<td>{chip("ok" if ok else "fail", "good" if ok else "critical")}</td>'
                f'<td class="cmd">{esc(r.get("reported",""))}</td></tr>')
        blocks.append(
            f'<div class="benchblock"><h3>{esc(name)} {flag}'
            f'<span class="sparkwrap">{sparkline(walls)}</span></h3>'
            f'<table><tbody>{"".join(trs)}</tbody></table></div>')
    return f'<section><h2>Benchmarks</h2>{"".join(blocks)}</section>'


CSS = """
:root{
  --surface-1:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--aqua:#1baf7a;
  --good:#0ca30c;--warning:#eda100;--serious:#ec835a;--critical:#d03b3b;--track:#eceae4;}
@media (prefers-color-scheme:dark){:root{
  --surface-1:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--border:rgba(255,255,255,.10);--series-1:#3987e5;--aqua:#199e70;
  --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#e05a5a;--track:#2c2c2a;}}
*{box-sizing:border-box}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:1.2rem;
  background:var(--plane);color:var(--ink);}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.2rem;margin:0}
h2{font-size:.95rem;margin:1.6rem 0 .5rem;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.04em;font-weight:600}
h3{font-size:.9rem;margin:.2rem 0 .5rem;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.sub{color:var(--muted);margin:.2rem 0 0;font-size:.82rem}
.muted{color:var(--muted)} .num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-variant-numeric:tabular-nums;color:var(--series-1)}
.cmd{max-width:0;width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.good-t{color:var(--good)}.warning-t{color:var(--warning)}.critical-t{color:var(--critical)}
.-t{}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.8rem;margin-top:.4rem}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:.8rem .9rem}
.card-h{display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;font-size:.95rem}
.card-h b{margin-right:auto}
.card-f{margin-top:.5rem;font-size:.78rem}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);display:inline-block}
.dot.good{background:var(--good)}.dot.warning{background:var(--warning)}.dot.critical{background:var(--critical)}
.good-t{color:var(--good)}.warning-t{color:var(--warning)}.critical-t{color:var(--critical)}
.stat{display:flex;justify-content:space-between;align-items:baseline;margin:.35rem 0 .15rem}
.stat .lbl{color:var(--muted);font-size:.8rem}
.stat .val{font-variant-numeric:tabular-nums;font-size:.95rem}
.stat .unit{color:var(--muted);font-size:.8rem}
.meter{height:6px;background:var(--track);border-radius:4px;overflow:hidden;margin-bottom:.2rem}
.meter .fill{height:100%;border-radius:4px}
.meter .fill.blue{background:var(--series-1)} .meter .fill.aqua{background:var(--aqua)}
.qgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}
.qcol{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:.7rem .9rem}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.4rem;margin:.4rem 0 .6rem}
.tile{background:var(--plane);border:1px solid var(--border);border-radius:8px;padding:.4rem;text-align:center}
.tile .tv{display:block;font-size:1.2rem;font-variant-numeric:tabular-nums}
.tile .tl{display:block;color:var(--muted);font-size:.68rem}
.tags{margin:.2rem 0 .3rem}
.chip{display:inline-block;border:1px solid var(--border);border-radius:5px;padding:.02rem .4rem;
  margin:.1rem .25rem .1rem 0;font-size:.76rem;color:var(--ink2)}
.chip.good{color:var(--good);border-color:var(--good)}
.chip.warning{color:var(--warning);border-color:var(--warning)}
.chip.critical{color:var(--critical);border-color:var(--critical)}
.chip.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:.2rem 0}
th,td{text-align:left;padding:.22rem .5rem;border-bottom:1px solid var(--grid);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.85rem}
th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.03em}
section{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
  padding:.4rem 1rem 1rem;margin-top:0}
section>h2{margin-top:.8rem}
.benchblock{margin:.6rem 0}
.sparkwrap{margin-left:auto}
.warnbox{border:1px solid var(--critical);border-radius:8px;padding:.5rem .8rem;margin:.5rem 0;
  background:color-mix(in srgb,var(--critical) 8%,transparent)}
.warnbox ul{margin:.3rem 0 0;padding-left:1.1rem} .warnbox code{color:var(--ink)}
.overflow{overflow-x:auto}
"""


def build_html(host_cards, queues, wts, results) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = '<div class="cards">' + "".join(host_cards) + "</div>"
    qcols = '<div class="qgrid">' + "".join(queues) + "</div>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='refresh' content='60'>"
        "<title>gradwave fleet</title><style>" + CSS + "</style></head><body><div class='wrap'>"
        f"<h1>gradwave fleet</h1><p class='sub'>updated {esc(now)} · auto-refresh 60s</p>"
        f"<section><h2>Hosts</h2>{cards}</section>"
        f"<section><h2>Queue</h2>{qcols}</section>"
        f"{render_agents(wts)}{render_benchmarks(results)}"
        "</div></body></html>")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="generate the gradwave fleet dashboard")
    p.add_argument("--out", default=str(REPO / "dashboard.html"))
    p.add_argument("--collect", action="store_true",
                   help="rsync remote benchmark results in before rendering")
    a, _ = p.parse_known_args(argv)  # tolerate legacy --hosts

    if a.collect:
        collect_results([s["host"] for s in FLEET if s["queue"]])

    metrics = {s["host"]: host_metrics(s["host"]) for s in FLEET}
    host_cards = [render_host_card(s, metrics[s["host"]]) for s in FLEET]
    queues = [render_queue(s, qs.fetch(s["host"])) for s in FLEET if s["queue"]]
    wts: list[dict] = []
    for s in FLEET:
        if s.get("worktrees"):
            wts.extend(fetch_worktrees(s["host"]))
    results = load_results()

    Path(a.out).write_text(build_html(host_cards, queues, wts, results))
    up = sum(1 for m in metrics.values() if m)
    print(f"wrote {a.out}  (hosts {up}/{len(FLEET)}, queues {len(queues)}, "
          f"worktrees {len(wts)}, bench {sum(len(v) for v in results.values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
