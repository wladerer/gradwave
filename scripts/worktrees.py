#!/usr/bin/env python3
"""Fleet worktree overview: drift, staleness, and cross-worktree file overlap.

Multiple agents each work in their own git worktree under .claude/worktrees/.
Worktrees isolate files — agents cannot clobber each other's tracked code — but
three things still cause pain, and this surfaces all three:

  * drift    — how far each branch is behind origin/main (rebase before it hurts)
  * stale    — worktrees whose branch is already merged (safe to prune)
  * overlap  — a file edited in more than one active worktree (a coming conflict)

Only `overlap` is otherwise invisible; it's the early warning that two agents are
about to collide in the same code.

Usage:
  make worktrees                       # report
  python scripts/worktrees.py
  python scripts/worktrees.py --prune  # remove stale + clean + idle worktrees
                                       #   (only under .claude/worktrees/, never
                                       #    the primary checkout or the current one)

Staleness uses two signals: no commits unique to the branch (ahead==0), or — for
squash-merged branches, whose commits differ from main — a merged PR reported by
`gh` (best-effort; skipped if gh is unavailable/offline).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import lru_cache


def git(*args: str, cwd: str | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def main_ref() -> str:
    for ref in ("origin/main", "main", "origin/master", "master"):
        if git("rev-parse", "--verify", "--quiet", ref):
            return ref
    return "origin/main"


def list_worktrees() -> list[dict]:
    out = git("worktree", "list", "--porcelain")
    wts, cur = [], {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur:
                wts.append(cur)
            cur = {"path": line[len("worktree "):]}
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):].replace("refs/heads/", "")
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
        elif line.startswith("detached"):
            cur["branch"] = "(detached)"
    if cur:
        wts.append(cur)
    return wts


@lru_cache(maxsize=None)
def pr_merged(branch: str) -> bool:
    """Best-effort: has this branch's PR been merged? (catches squash-merges)."""
    if not branch or branch == "(detached)":
        return False
    try:
        r = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "merged",
             "--json", "number", "-q", "length"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() not in ("", "0")


def changed_files(path: str, base: str) -> set[str]:
    """Files this worktree touches: committed since base + uncommitted."""
    files = set()
    if base:
        files |= set(git("diff", "--name-only", f"{base}", "HEAD", cwd=path).splitlines())
    for line in git("status", "--porcelain", cwd=path).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name:
            files.add(name)
    files.discard("")
    return files


def procs_in(path: str) -> list[str]:
    """PIDs whose cwd is inside `path` (best-effort; Linux /proc)."""
    hits = []
    if not os.path.isdir("/proc"):
        return hits
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if cwd == path or cwd.startswith(path + "/"):
            hits.append(pid)
    return hits


def gather(mref: str, use_gh: bool = True) -> list[dict]:
    rows = []
    primary_common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    for wt in list_worktrees():
        path, branch = wt["path"], wt.get("branch", "(detached)")
        base = git("merge-base", mref, "HEAD", cwd=path)
        lr = git("rev-list", "--left-right", "--count", f"{mref}...HEAD", cwd=path)
        behind, ahead = (lr.split() + ["?", "?"])[:2]
        dirty = len(git("status", "--porcelain", cwd=path).splitlines())
        is_wt = "/.claude/worktrees/" in path
        stale = (ahead == "0" and behind not in ("0", "?")) or (use_gh and pr_merged(branch))
        rows.append({
            "path": path, "name": os.path.basename(path), "branch": branch,
            "behind": behind, "ahead": ahead, "dirty": dirty,
            "base": base[:8], "is_wt": is_wt, "stale": stale,
            "has_agent": bool(procs_in(path)) if is_wt else False,
            "files": changed_files(path, base) if is_wt else set(),
        })
    _ = primary_common
    return rows


# --------------------------------------------------------------------------- #

def report(rows: list[dict], mref: str) -> None:
    print(f"worktrees vs {mref}  (behind/ahead)\n")
    print(f"  {'worktree':<32} {'branch':<34} {'b/a':>7} {'dirty':>5}  state")
    for r in sorted(rows, key=lambda r: (not r["is_wt"], r["stale"], r["name"])):
        if not r["is_wt"]:
            state = "primary"
        elif r["stale"]:
            state = "STALE (merged)"
        elif r["dirty"]:
            state = "active*"
        else:
            state = "active"
        ba = f"{r['behind']}/{r['ahead']}"
        print(f"  {r['name']:<32} {r['branch']:<34} {ba:>7} {r['dirty']:>5}  {state}")

    stale = [r for r in rows if r["is_wt"] and r["stale"]]
    if stale:
        print("\nstale (branch merged — safe to prune):")
        for r in stale:
            print(f"  {r['name']}  [{r['branch']}]")
        print("  → prune with: make worktrees-prune")

    # overlap across ACTIVE worktrees only
    active = [r for r in rows if r["is_wt"] and not r["stale"]]
    owners: dict[str, list[str]] = {}
    for r in active:
        for f in r["files"]:
            owners.setdefault(f, []).append(r["name"])
    clashes = {f: w for f, w in owners.items() if len(w) > 1}
    if clashes:
        print("\n⚠ file overlap across active worktrees (conflict incoming):")
        for f, w in sorted(clashes.items()):
            print(f"  {f}\n      touched by: {', '.join(sorted(w))}")
    elif active:
        print("\n✓ no file overlap across active worktrees")

    drifters = [r for r in active if r["behind"].isdigit() and int(r["behind"]) >= 10]
    if drifters:
        print("\n↯ drifting ≥10 commits behind — rebase on main soon:")
        for r in drifters:
            print(f"  {r['name']}  ({r['behind']} behind)")


def prune(rows: list[dict]) -> int:
    here = git("rev-parse", "--path-format=absolute", "--show-toplevel") or os.getcwd()
    removed = 0
    for r in rows:
        if not (r["is_wt"] and r["stale"]):
            continue
        path, name, branch = r["path"], r["name"], r["branch"]
        if path == here:
            print(f"  skip {name}: it is the current worktree "
                  f"(exit it first, then prune)")
            continue
        if r["dirty"]:
            print(f"  skip {name}: {r['dirty']} uncommitted change(s)")
            continue
        busy = procs_in(path)
        if busy:
            print(f"  skip {name}: {len(busy)} process(es) still cwd'd inside "
                  f"(pids {','.join(busy[:5])})")
            continue
        rm = subprocess.run(["git", "worktree", "remove", path],
                            capture_output=True, text=True)
        if rm.returncode != 0:
            print(f"  FAIL {name}: {rm.stderr.strip().splitlines()[-1:] or ['?']}")
            continue
        # merged branch: safe to delete (force covers squash-merges)
        if branch and branch != "(detached)":
            subprocess.run(["git", "branch", "-D", branch],
                           capture_output=True, text=True)
        print(f"  removed {name}  [{branch}]")
        removed += 1
    print(f"\npruned {removed} worktree(s)")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="fleet worktree overview")
    p.add_argument("--prune", action="store_true",
                   help="remove stale + clean + idle worktrees under .claude/worktrees/")
    p.add_argument("--json", action="store_true",
                   help="machine-readable dump (fast: skips the gh merged-PR check)")
    p.add_argument("--no-fetch", action="store_true", help="skip the origin/main fetch")
    a = p.parse_args(argv)

    if not a.no_fetch and not a.json:
        subprocess.run(["git", "fetch", "origin", "main", "-q"],
                       capture_output=True, timeout=30)
    mref = main_ref()
    rows = gather(mref, use_gh=not a.json)
    if a.json:
        import json
        out = [{**r, "files": sorted(r["files"])} for r in rows]
        print(json.dumps(out))
        return 0
    if a.prune:
        return prune(rows)
    report(rows, mref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
