#!/usr/bin/env python3
"""Render a compact, fleet-wide view of the pueue queues.

`gwq status [hosts...]` calls this. For each host it runs `pueue status --json`
(locally or over ssh), normalises pueue's version-dependent status enum, and
prints a small table: running first, then queued, then the most recent finished
jobs. This is the live half of the dashboard; the historical benchmark trends
live in benchmarks/results/<host>/ and get their own view later.

Stdlib only — no project env needed.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from datetime import datetime, timezone

# asus's OS hostname is "nixos"; map it to the logical name used by callers.
HOST_ALIASES = {"nixos": "asus"}


def this_host() -> str:
    raw = socket.gethostname()
    return HOST_ALIASES.get(raw, raw)


def fetch(host: str) -> dict | None:
    local = host == this_host()
    argv = ["pueue", "status", "--json"] if local else \
        ["ssh", host, "pueue status --json"]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"  ({host} unreachable: {e})")
        return None
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()[-1:] or [""]
        print(f"  ({host}: pueue error: {err[0]})")
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        print(f"  ({host}: could not parse pueue json)")
        return None


def _status_kind(status) -> tuple[str, str | None]:
    """Normalise pueue's status enum -> (kind, result). kind is one of
    Running/Queued/Paused/Done/Stashed/... ; result is Success/Failed/... or None."""
    if isinstance(status, str):
        return status, None
    if isinstance(status, dict):
        kind = next(iter(status), "?")
        body = status[kind]
        result = None
        if isinstance(body, dict) and "result" in body:
            res = body["result"]
            if isinstance(res, str):
                result = res
            elif isinstance(res, dict):
                # {"Failed": 1} etc.
                k = next(iter(res), "?")
                result = f"{k}({res[k]})"
        return kind, result
    return "?", None


def _parse_ts(val):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _elapsed(task, kind) -> str:
    body = task.get("status")
    start = end = None
    if isinstance(body, dict):
        inner = next(iter(body.values()))
        if isinstance(inner, dict):
            start = _parse_ts(inner.get("start"))
            end = _parse_ts(inner.get("end"))
    if start is None:
        return "-"
    stop = end or datetime.now(timezone.utc)
    try:
        secs = (stop - start).total_seconds()
    except TypeError:
        return "-"
    if secs < 0:
        return "-"
    if secs < 90:
        return f"{secs:.0f}s"
    if secs < 5400:
        return f"{secs/60:.0f}m"
    return f"{secs/3600:.1f}h"


ORDER = {"Running": 0, "Paused": 1, "Queued": 2, "Stashed": 3, "Done": 4}


def render(host: str, data: dict) -> None:
    tasks = data.get("tasks", {}) or {}
    groups = data.get("groups", {}) or {}

    gline = []
    for gname, g in sorted(groups.items()):
        par = g.get("parallel_tasks", g.get("parallel", "?")) if isinstance(g, dict) else "?"
        st = g.get("status", "") if isinstance(g, dict) else ""
        tag = f"{gname}={par}" + ("(paused)" if str(st).lower() == "paused" else "")
        gline.append(tag)
    print(f"\n=== {host} ===")
    print("  groups: " + (", ".join(gline) if gline else "(none — run `gwq init`)"))

    rows = []
    for tid, t in tasks.items():
        kind, result = _status_kind(t.get("status"))
        rows.append((ORDER.get(kind, 9), int(tid) if str(tid).isdigit() else 0,
                     tid, t.get("group", "?"), kind, result, t))

    # newest finished first; running/queued by id
    def sort_key(r):
        bucket = r[0]
        return (bucket, -r[1] if bucket == 4 else r[1])

    rows.sort(key=sort_key)

    # cap the finished list so the view stays small
    shown, done_shown = [], 0
    for r in rows:
        if r[0] == 4:
            if done_shown >= 6:
                continue
            done_shown += 1
        shown.append(r)

    if not shown:
        print("  (queue empty)")
        return

    print(f"  {'id':>4}  {'group':<7} {'state':<8} {'elapsed':>7}  label / command")
    for _b, _i, tid, group, kind, result, t in shown:
        state = kind if not result else f"{kind}:{result}"
        elapsed = _elapsed(t, kind)
        label = t.get("label") or ""
        cmd = t.get("command", "")
        desc = label if label else (cmd[:60] + ("…" if len(cmd) > 60 else ""))
        mark = "✗" if (result and result != "Success") else " "
        print(f"  {tid:>4}  {group:<7} {state:<8} {elapsed:>7} {mark} {desc}")


def main(argv: list[str]) -> int:
    hosts = argv or ["thinkpad", "asus"]
    print(f"gradwave queue — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    any_ok = False
    for host in hosts:
        data = fetch(host)
        if data is not None:
            render(host, data)
            any_ok = True
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
