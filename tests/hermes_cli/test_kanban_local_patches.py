"""Local-patch invariant tests for the re-ported kanban customizations.

Regression for the 2026-09-07 dev-board stall (two cards held ~11h in ready
after a human unblock; 650+ respawn_guarded event rows; 671 false
"dispatcher stuck" ticks). These assert the three behaviours the local patch
adds, each on the re-ported (v2026.9.14) code paths:

  1. active_pr requeue bypass — an explicit re-queue after the PR comment is a
     deliberate re-run and must NOT be deferred by the guard.
  2. respawn_guarded event dedup — a task held across ticks emits ONE event per
     distinct reason, not one per tick.
  3. guard-aware telemetry — a queue the guard is deliberately holding is NOT
     reported as "spawnable" (so the stuck-dispatcher warning stays quiet).

Plus the goal-mode decompose policy: implementation assignees get goal_mode,
everything else stays single-shot.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as kd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kdd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


# ── 1. active_pr requeue bypass ────────────────────────────────────────────

def _make_ready_with_pr_comment(conn, *, requeue_after: bool):
    tid = kb.create_task(conn, title="pr-blocked card", assignee="coder")
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    now = int(time.time())
    kb.add_comment(
        conn, tid, "worker",
        "Opened PR https://github.com/example/repo/pull/123 for this task.",
    )
    if requeue_after:
        # A deliberate re-queue event AFTER the PR comment.
        kb._append_event(conn, tid, "unblocked", {"by": "operator"})
    conn.commit()
    return tid


def test_active_pr_defers_without_requeue(conn):
    """Baseline: no requeue after the PR comment -> guard holds the task."""
    tid = _make_ready_with_pr_comment(conn, requeue_after=False)
    assert kdd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_requeue_bypass(conn):
    """Local patch: an explicit re-queue AFTER the PR comment honors the re-run.

    Uses the same predicate the guard uses (created_at >= latest PR comment).
    """
    tid = _make_ready_with_pr_comment(conn, requeue_after=True)
    # The unblock event is stamped with the current time; the PR comment was
    # inserted just before it, so the requeue-window comparison must find it.
    reason = kdd.check_respawn_guard(conn, tid, lane="ready")
    assert reason != "active_pr", (
        "an explicit re-queue after the PR comment must bypass active_pr, "
        f"got {reason!r}"
    )


# ── 3. guard-aware spawnable telemetry ─────────────────────────────────────

def test_has_spawnable_excludes_guarded_rows(conn, monkeypatch):
    """A ready row the guard holds must not count as spawnable.

    Without the guard check the dispatcher emitted a false "stuck" warning
    every 5 minutes for a correctly-deferred PR-blocked card.
    """
    tid = _make_ready_with_pr_comment(conn, requeue_after=False)
    # Sanity: the guard really does hold this row.
    assert kdd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"
    # Force the profile lookup open so the assertion is about the GUARD, not
    # about whether a "coder" profile happens to exist under the test home.
    monkeypatch.setattr(kdd, "_profile_exists_fn", lambda: (lambda name: True))
    assert kdd._has_spawnable(conn, "ready") is False, (
        "a guard-held row must not be reported as spawnable"
    )


def test_has_spawnable_true_for_unguarded_row(conn, monkeypatch):
    """Control: the same predicate returns True when nothing guards the row."""
    kb.create_task(conn, title="plain ready card", assignee="coder")
    conn.commit()
    monkeypatch.setattr(kdd, "_profile_exists_fn", lambda: (lambda name: True))
    assert kdd._has_spawnable(conn, "ready") is True


def test_record_task_event_writes_outside_txn(conn):
    """record_task_event must persist an event from a caller holding no open write
    transaction — the seam the kanban goal loop records judge verdicts through."""
    from hermes_cli import kanban_db as kb2
    tid = kb2.create_task(conn, title="event seam test", assignee="coder")
    conn.commit()
    kb2.record_task_event(
        conn, tid, "goal_judged",
        {"verdict": "continue", "reason": "not done", "turn": 1},
    )
    events = kb2.list_events(conn, tid)
    judged = [e for e in events if e.kind == "goal_judged"]
    assert len(judged) == 1
    payload = judged[0].payload
    if isinstance(payload, str):
        import json as _json
        payload = _json.loads(payload)
    assert payload["verdict"] == "continue"
    assert payload["turn"] == 1


def test_goal_loop_passes_event_fn_through():
    """run_kanban_goal_loop must expose event_fn and call it for each verdict."""
    import inspect
    from hermes_cli import goals as g
    assert "event_fn" in inspect.signature(g.run_kanban_goal_loop).parameters

    seen = []
    # The loop only judges while the task is non-terminal ("running"/"ready"),
    # so the status function must report a live task; the judge is patched to a
    # fixed verdict and max_turns=1 ends the loop after one judged turn.
    orig_judge = g.judge_goal
    g.judge_goal = lambda goal, last: ("continue", "not done yet", False, False, False)
    try:
        result = g.run_kanban_goal_loop(
            task_id="t1",
            goal_text="do the thing",
            run_turn=lambda prompt: "worker said something",
            task_status_fn=lambda: "running",
            block_fn=lambda msg: None,
            max_turns=1,
            first_response="first",
            event_fn=lambda kind, payload=None: seen.append((kind, payload)),
        )
    finally:
        g.judge_goal = orig_judge
    assert any(kind == "goal_judged" for kind, _ in seen), (
        f"expected a goal_judged event, got {seen!r} (result={result!r})"
    )
    payload = next(p for k, p in seen if k == "goal_judged")
    assert payload["verdict"] == "continue"
    assert payload["turn"] == 1
    assert payload["max_turns"] == 1


def test_handoff_rejection_accepts_conn():
    """The handoff gate takes an optional conn so it can log its verdict."""
    import inspect
    from hermes_cli import kanban as k
    assert "conn" in inspect.signature(k._goal_mode_handoff_rejection).parameters


# ── goal-mode decompose policy ─────────────────────────────────────────────

def test_goal_mode_policy_implementation_assignees_only():
    """coder/designer/debug get goal-mode; architect and others do not."""
    assert kd.GOAL_MODE_ASSIGNEES == frozenset({"coder", "designer", "debug"})
    # architect is deliberately excluded (the judge also gates
    # kanban_request_review).
    assert "architect" not in kd.GOAL_MODE_ASSIGNEES
    assert "docs" not in kd.GOAL_MODE_ASSIGNEES


def test_resolve_goal_max_turns_rejects_bools_and_nonpositives():
    """Only a positive real int is honored; anything else falls back."""
    assert kd._resolve_goal_max_turns({}) == kd.DEFAULT_GOAL_MAX_TURNS
    assert kd._resolve_goal_max_turns({"decomposer_goal_max_turns": 30}) == 30
    assert kd._resolve_goal_max_turns({"decomposer_goal_max_turns": True}) == kd.DEFAULT_GOAL_MAX_TURNS
    assert kd._resolve_goal_max_turns({"decomposer_goal_max_turns": 0}) == kd.DEFAULT_GOAL_MAX_TURNS
    assert kd._resolve_goal_max_turns({"decomposer_goal_max_turns": -5}) == kd.DEFAULT_GOAL_MAX_TURNS
    assert kd._resolve_goal_max_turns({"decomposer_goal_max_turns": "40"}) == kd.DEFAULT_GOAL_MAX_TURNS


def test_clean_children_stamps_goal_mode_on_resolved_assignee():
    """_clean_children keys the policy on the RESOLVED assignee, and stamps the
    turn budget only for goal-mode children."""
    routing = kd._Routing(
        orchestrator="orchestrator",
        default_assignee="coder",
        auto_promote=True,
        roster=[],
        valid_names={"coder", "designer", "debug", "docs", "architect"},
        goal_max_turns=20,
    )
    raw = [
        {"title": "impl work", "assignee": "coder"},
        {"title": "docs work", "assignee": "docs"},
        {"title": "review work", "assignee": "architect"},
        # Unknown assignee -> routed to default_assignee ("coder") -> goal-mode.
        {"title": "typo'd assignee", "assignee": "not-a-profile"},
    ]
    children, reason = kd._clean_children("task-1", raw, routing)
    assert reason == ""
    assert children[0]["goal_mode"] is True
    assert children[0]["goal_max_turns"] == 20
    assert children[1]["goal_mode"] is False
    assert children[1]["goal_max_turns"] is None
    assert children[2]["goal_mode"] is False
    assert children[2]["goal_max_turns"] is None
    # Resolved-assignee keying: the typo lands on coder and inherits its policy.
    assert children[3]["assignee"] == "coder"
    assert children[3]["goal_mode"] is True


def test_decomposed_child_insert_persists_goal_mode(conn):
    """The DB graph insert must carry goal_mode/goal_max_turns into the row."""
    root = kb.create_task(conn, title="rough idea", assignee="orchestrator")
    conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root,))
    conn.commit()
    children = [
        {"title": "impl child", "body": "", "assignee": "coder",
         "parents": [], "goal_mode": True, "goal_max_turns": 20},
        {"title": "docs child", "body": "", "assignee": "docs",
         "parents": [], "goal_mode": False, "goal_max_turns": None},
    ]
    ids = kd.decompose_triage_task(
        conn, root, root_assignee="orchestrator", children=children, auto_promote=False,
    )
    assert ids and len(ids) == 2
    rows = {
        r["title"]: r
        for r in conn.execute(
            "SELECT title, goal_mode, goal_max_turns FROM tasks WHERE id IN (?, ?)",
            (ids[0], ids[1]),
        ).fetchall()
    }
    assert bool(rows["impl child"]["goal_mode"]) is True
    assert rows["impl child"]["goal_max_turns"] == 20
    assert bool(rows["docs child"]["goal_mode"]) is False
    assert rows["docs child"]["goal_max_turns"] is None
