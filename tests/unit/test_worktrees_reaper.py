"""Tests for the worktree-reaper decision logic (scripts/worktrees.py).

The gh/git plumbing is integration-only; the reaping *decisions* — which PR
states retire a worktree, and which branch wins when a name carries several PRs —
are pure functions and are pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "worktrees.py"


def _load():
    spec = importlib.util.spec_from_file_location("worktrees", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wt = _load()


# --- _classify: which worktrees are stale (reapable) ------------------------- #

def test_merged_pr_is_stale():
    assert wt._classify("MERGED", ahead="5", behind="0") == (True, "merged")


def test_closed_pr_is_stale_even_with_unique_commits():
    # the gap the old merged-only check missed: a closed-but-unmerged branch
    assert wt._classify("CLOSED", ahead="6", behind="12") == (True, "PR closed")


def test_open_pr_is_never_reaped():
    assert wt._classify("OPEN", ahead="3", behind="40") == (False, "")
    # even if it momentarily looks orphaned (stale local checkout post-rebase)
    assert wt._classify("OPEN", ahead="0", behind="5") == (False, "")


def test_orphan_when_no_pr_and_no_unique_commits():
    assert wt._classify("", ahead="0", behind="7") == (True, "orphan")


def test_active_when_no_pr_and_ahead():
    assert wt._classify("", ahead="4", behind="9") == (False, "")


def test_unknown_counts_are_not_orphaned():
    # ahead==0/behind=='?' (a fresh/odd worktree) must not be reaped
    assert wt._classify("", ahead="0", behind="?") == (False, "")


# --- _pick_states: one state per branch across multiple PRs ------------------ #

def test_pick_states_open_outranks_finished():
    prs = [
        {"headRefName": "feat-x", "state": "CLOSED"},
        {"headRefName": "feat-x", "state": "OPEN"},   # a reopened / reused branch
        {"headRefName": "feat-y", "state": "MERGED"},
    ]
    assert wt._pick_states(prs) == {"feat-x": "OPEN", "feat-y": "MERGED"}


def test_pick_states_merged_outranks_closed():
    prs = [
        {"headRefName": "b", "state": "CLOSED"},
        {"headRefName": "b", "state": "MERGED"},
    ]
    assert wt._pick_states(prs) == {"b": "MERGED"}


def test_pick_states_ignores_blank_branch_and_empty():
    assert wt._pick_states([]) == {}
    assert wt._pick_states([{"headRefName": "", "state": "MERGED"}]) == {}
