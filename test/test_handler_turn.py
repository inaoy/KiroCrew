"""The native Slack turn's object and the plumbing between its phases.

``kiro_crew.slack.handler.handle_message`` builds one ``_SlackTurn`` per call and runs its
turn as three phase coroutines that live in ``slack/handler_runtime`` --
``_dispatch_before_turn`` (``dispatch.py``), ``_run_turn`` (``turn.py``) and
``_deliver_answer`` (``delivery.py``) -- with the stream loop handing each tool call and
permission request to ``permission_ladder.py``. Each phase opens with read-only aliases of
the turn values it reads and ends by copying onto the turn what the next phase reads.
These tests pin what that composition promises:

* a turn value read before anything set it raises what reading the unbound local did;
* the turn's permit and verdict methods do what the delivery closures they replace did,
  in the same order;
* every value a phase reads off the turn was set before it -- by ``handle_message``, by
  an earlier phase, or earlier in the same phase -- and every value a phase copies onto
  the turn is read by a later one;
* every import the facade binds is read by the facade or an owner, or is one of the
  owner names the facade exported at the base for other modules (a frozen list), and a
  function-local import is read in its own function (the property ``setup.cfg``'s
  per-file F401 row stands in for);
* the sync-io-in-async gate still reads every owner coroutine;
* the function list the checks that read ``handle_message``'s source share
  (``SLACK_TURN_FUNCTIONS``) is ``handle_message`` and its phases in run order.
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from source_corpus import repo_root

from kiro_crew.slack import handler

_FACADE_PATH = Path(handler.__file__).resolve()
_RUNTIME_DIR = _FACADE_PATH.parent / "handler_runtime"

#: The phases in the order ``handle_message`` runs them, each as (owner stem, function).
#: The facade's own ``handle_message`` builds the turn first.
_PHASES = (
    ("dispatch", "_dispatch_before_turn"),
    ("turn", "_run_turn"),
    ("delivery", "_deliver_answer"),
)
#: The permission ladder's two functions, called from ``_run_turn``'s stream loop.
_LADDER = (
    ("permission_ladder", "_screen_tool_call"),
    ("permission_ladder", "_answer_permission_request"),
)
#: The call in ``_run_turn`` the ladder's reads are measured against: a ladder value
#: must be on the turn before the stream loop that calls it opens.
_LADDER_CALL = "await _screen_tool_call(turn, event)"

#: ``handle_message`` and the functions its turn runs in, in run order (the ladder runs
#: in ``_run_turn``'s stream loop). A check that reads ``handle_message``'s source reads
#: these as one text (``test_options_click_validation``, ``test_slack_options_lifecycle``).
SLACK_TURN_FUNCTIONS = (
    "handle_message",
    "_dispatch_before_turn",
    "_run_turn",
    "_screen_tool_call",
    "_answer_permission_request",
    "_deliver_answer",
)
#: Where an entry point's turn takes its session, read as one source text: for the Slack
#: handler, ``handle_message`` with the dispatch before the turn and the turn body.
TURN_ENTRY_FUNCTIONS = {
    ("kiro_crew.slack.handler", "handle_message"): SLACK_TURN_FUNCTIONS[:3],
}


def _source(stem: str) -> str:
    return (_RUNTIME_DIR / f"{stem}.py").read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_the_shared_turn_functions_are_handle_message_and_its_phases_in_run_order() -> None:
    (_, dispatch), (_, run), (_, deliver) = _PHASES
    ladder = tuple(name for _, name in _LADDER)
    assert SLACK_TURN_FUNCTIONS == ("handle_message", dispatch, run, *ladder, deliver)
    assert TURN_ENTRY_FUNCTIONS == {
        ("kiro_crew.slack.handler", "handle_message"): ("handle_message", dispatch, run)
    }
    facade = _FACADE_PATH.read_text(encoding="utf-8")
    entry = ast.get_source_segment(facade, _function(facade, "handle_message")) or ""
    calls = [entry.find(f"await {name}(turn)") for name in (dispatch, run, deliver)]
    assert -1 not in calls and calls == sorted(calls)
    body = _source("turn")
    loop = [body.find(f"await {name}(turn, ") for name in ladder]
    assert -1 not in loop and loop == sorted(loop)


# ── the turn object ───────────────────────────────────────────────────────────


def _bare_turn() -> handler._SlackTurn:
    """A turn with no attribute set at all, so each read reaches ``__getattr__``."""
    return handler._SlackTurn.__new__(handler._SlackTurn)


def test_a_declared_value_read_before_it_is_set_raises_unbound_local() -> None:
    turn = _bare_turn()
    with pytest.raises(UnboundLocalError, match="'_stop_reason'"):
        turn._stop_reason
    # A handler written for the unbound local's own NameError still catches it.
    assert issubclass(UnboundLocalError, NameError)


def test_an_undeclared_name_is_an_attribute_error() -> None:
    turn = _bare_turn()
    with pytest.raises(AttributeError):
        turn.not_a_turn_value
    with pytest.raises(AttributeError):
        getattr(turn, "_stop_reasons")


def test_a_set_value_reads_back() -> None:
    turn = _bare_turn()
    turn._stop_reason = "end_turn"
    assert turn._stop_reason == "end_turn"


def test_the_turn_class_keeps_its_owner_module_and_runs_on_the_facade() -> None:
    """``compose`` picks an owner's classes by ``__module__ == owner.__name__``: the turn
    class must keep ``turn.py``'s module, or its methods would keep their own globals."""
    assert handler._SlackTurn.__module__ == "kiro_crew.slack.handler_runtime.turn"
    for name in ("__init__", "release_permit", "book_success", "book_failure", "__getattr__"):
        assert vars(handler._SlackTurn)[name].__globals__ is vars(handler), name


class _Recorder:
    """Ordered record of the session-manager, stats and telemetry calls a method makes."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def stats(self) -> SimpleNamespace:
        return SimpleNamespace(
            inc_message_success=lambda: self.calls.append(("stats.inc_message_success",)),
            inc_message_failed=lambda: self.calls.append(("stats.inc_message_failed",)),
        )


class _Sessions:
    def __init__(self, recorder: _Recorder, *, with_gap: bool = True) -> None:
        self._recorder = recorder
        if with_gap:
            self.close_replay_gap = lambda key: recorder.calls.append(("close_replay_gap", key))

    def release(self, key: str) -> None:
        self._recorder.calls.append(("release", key))

    def record_success(self, key: str) -> None:
        self._recorder.calls.append(("record_success", key))

    async def record_failure(self, key: str) -> bool:
        self._recorder.calls.append(("record_failure", key))
        return False


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(handler, "Stats", recorder.stats)
    monkeypatch.setattr(
        handler,
        "runtime_death",
        SimpleNamespace(
            clear_shared_deaths=lambda key: recorder.calls.append(("clear_shared_deaths", key))
        ),
    )
    monkeypatch.setattr(
        handler,
        "record_interaction_event",
        lambda client, key, source: recorder.calls.append(("interaction", client, key, source)),
    )
    return recorder


def _turn(recorder: _Recorder, **values: object) -> handler._SlackTurn:
    turn = _bare_turn()
    turn.sessions = values.pop("sessions", _Sessions(recorder))
    turn.session_key = "slack:1.2"
    for name, value in values.items():
        setattr(turn, name, value)
    return turn


def test_release_permit_releases_then_closes_the_gap_once(recorded: _Recorder) -> None:
    turn = _turn(recorded, _acquired=True)
    turn.release_permit()
    turn.release_permit()
    assert recorded.calls == [("release", "slack:1.2"), ("close_replay_gap", "slack:1.2")]
    assert turn._acquired is False


def test_release_permit_without_the_permit_does_nothing(recorded: _Recorder) -> None:
    turn = _turn(recorded, _acquired=False)
    turn.release_permit()
    assert recorded.calls == []


def test_release_permit_tolerates_a_manager_without_the_gap(recorded: _Recorder) -> None:
    turn = _turn(recorded, sessions=_Sessions(recorded, with_gap=False), _acquired=True)
    turn.release_permit()
    assert recorded.calls == [("release", "slack:1.2")]


def test_book_success_books_once_in_the_closures_order(recorded: _Recorder) -> None:
    client = object()
    turn = _turn(recorded, _verdict_booked=False, client=client)
    turn.book_success()
    turn.book_success()
    assert recorded.calls == [
        ("record_success", "slack:1.2"),
        ("clear_shared_deaths", "slack:1.2"),
        ("stats.inc_message_success",),
        ("interaction", client, "slack:1.2", "slack"),
    ]
    assert turn._verdict_booked is True


def test_book_success_without_a_client_records_no_interaction(recorded: _Recorder) -> None:
    turn = _turn(recorded, _verdict_booked=False, client=None)
    turn.book_success()
    assert [call[0] for call in recorded.calls] == [
        "record_success",
        "clear_shared_deaths",
        "stats.inc_message_success",
    ]


@pytest.mark.asyncio
async def test_book_failure_books_once_and_marks_the_error(recorded: _Recorder) -> None:
    turn = _turn(recorded, _verdict_booked=False, _had_error=False)
    await turn.book_failure()
    await turn.book_failure()
    assert recorded.calls == [("record_failure", "slack:1.2"), ("stats.inc_message_failed",)]
    assert turn._verdict_booked is True and turn._had_error is True


@pytest.mark.asyncio
async def test_a_booked_verdict_is_never_booked_again(recorded: _Recorder) -> None:
    turn = _turn(recorded, _verdict_booked=False, _had_error=False, client=None)
    turn.book_success()
    await turn.book_failure()
    assert [call[0] for call in recorded.calls] == [
        "record_success",
        "clear_shared_deaths",
        "stats.inc_message_success",
    ]
    assert turn._had_error is False


# ── the plumbing between the phases ───────────────────────────────────────────


def _turn_accesses(fn: ast.AST) -> list[tuple[int, int, str, str]]:
    """``(line, col, "read" | "write", name)`` for every ``turn.<name>`` in *fn*."""
    found = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "turn"
        ):
            kind = "write" if isinstance(node.ctx, ast.Store) else "read"
            found.append((node.lineno, node.col_offset, kind, node.attr))
    return sorted(found)


def _plumbing_faults(
    init: set[str],
    phases: list[ast.AST],
    ladder: list[ast.AST],
    ladder_call_line: int,
    ladder_host: int,
    methods: dict[str, set[str]],
) -> list[str]:
    """What breaks the hand-off rule, as readable lines (empty when it holds).

    *init* is what the turn's constructor sets, *phases* the phase functions in run
    order, *ladder* the functions phase *ladder_host* calls from the line
    *ladder_call_line*, and *methods* each turn method with the attributes it reads.
    A read must follow a write of the same name: in the constructor, an earlier phase,
    or earlier in the same phase; a ladder read must follow a write that precedes the
    ladder's call in its host phase. A write must be read by a later phase, the
    ladder, or a method the delivery calls; otherwise it hands on nothing.
    """
    faults: list[str] = []
    set_before = set(init)
    method_reads = set().union(*methods.values()) if methods else set()
    accesses = [_turn_accesses(fn) for fn in phases]
    for index, fn in enumerate(phases):
        written_here = set()
        for line, _col, kind, name in accesses[index]:
            if name in methods:
                continue  # a method call, not a value
            if kind == "read" and name not in set_before | written_here:
                faults.append(f"{fn.name}:{line} reads turn.{name} before anything sets it")
            if kind == "write":
                written_here.add(name)
        if index == ladder_host:
            before_call = set_before | {
                name
                for line, _col, kind, name in accesses[index]
                if kind == "write" and line < ladder_call_line
            }
            for lfn in ladder:
                for line, _col, kind, name in _turn_accesses(lfn):
                    if kind == "read" and name not in before_call and name not in methods:
                        faults.append(
                            f"{lfn.name}:{line} reads turn.{name} before {fn.name} sets it"
                        )
        set_before |= written_here
    for index, fn in enumerate(phases):
        later_reads = {
            name
            for later in [*accesses[index + 1 :], *(_turn_accesses(lfn) for lfn in ladder)]
            for _line, _col, kind, name in later
            if kind == "read"
        } | method_reads
        for line, _col, kind, name in accesses[index]:
            if kind == "write" and name not in later_reads and name not in methods:
                # A value the phase itself reads back after writing it is a turn
                # attribute only because a method or a later phase needs it.
                faults.append(f"{fn.name}:{line} hands on turn.{name}, which nothing reads")
    return faults


def _init_names(cls: ast.ClassDef) -> set[str]:
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }


def _method_reads(cls: ast.ClassDef) -> dict[str, set[str]]:
    return {
        node.name: {
            sub.attr
            for sub in ast.walk(node)
            if isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
        }
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("__")
    }


def _turn_class() -> ast.ClassDef:
    return next(
        node
        for node in ast.parse(_source("turn")).body
        if isinstance(node, ast.ClassDef) and node.name == "_SlackTurn"
    )


def _real_plumbing_faults() -> list[str]:
    cls = _turn_class()
    phases = [_function(_source(stem), name) for stem, name in _PHASES]
    ladder = [_function(_source(stem), name) for stem, name in _LADDER]
    run_source = _source("turn")
    call_line = next(
        number for number, text in enumerate(run_source.splitlines(), 1) if _LADDER_CALL in text
    )
    return _plumbing_faults(_init_names(cls), phases, ladder, call_line, 1, _method_reads(cls))


def test_every_turn_value_is_set_before_a_phase_reads_it() -> None:
    assert _real_plumbing_faults() == []


def test_the_plumbing_scan_reads_every_phase_and_the_ladder() -> None:
    """Non-vacuity on the real source: the scan sees reads and writes in each phase."""
    for stem, name in (*_PHASES, *_LADDER):
        kinds = {kind for _l, _c, kind, _n in _turn_accesses(_function(_source(stem), name))}
        assert "read" in kinds, name
    for stem, name in _PHASES[:2]:
        kinds = {kind for _l, _c, kind, _n in _turn_accesses(_function(_source(stem), name))}
        assert "write" in kinds, name


def _synthetic(source: str) -> list[str]:
    tree = ast.parse(textwrap.dedent(source))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}
    ladder_call = next(
        number for number, text in enumerate(source.splitlines(), 1) if "await ladder(turn)" in text
    )
    return _plumbing_faults(
        {"text"},
        [functions["first"], functions["second"]],
        [functions["ladder"]],
        ladder_call,
        0,
        {"book": {"flag"}},
    )


_GOOD = """
async def first(turn):
    text = turn.text
    turn.agent = text
    await ladder(turn)
    turn.flag = True

async def second(turn):
    agent = turn.agent
    turn.book()

async def ladder(turn):
    return turn.agent
"""


def test_the_plumbing_scan_passes_a_sound_hand_off() -> None:
    assert _synthetic(_GOOD) == []


def test_the_plumbing_scan_reports_a_read_nothing_set() -> None:
    bad = _GOOD.replace("    agent = turn.agent\n", "    agent = turn.never_set\n")
    assert _synthetic(bad) == ["second:9 reads turn.never_set before anything sets it"]


def test_the_plumbing_scan_reports_a_read_before_its_own_phase_sets_it() -> None:
    bad = _GOOD.replace(
        "    text = turn.text\n    turn.agent = text\n",
        "    text = turn.agent\n    turn.agent = text\n",
    )
    assert _synthetic(bad) == ["first:3 reads turn.agent before anything sets it"]


def test_the_plumbing_scan_reports_a_ladder_read_its_host_sets_too_late() -> None:
    bad = _GOOD.replace("    return turn.agent\n", "    return turn.flag\n")
    assert _synthetic(bad) == ["ladder:13 reads turn.flag before first sets it"]


def test_the_plumbing_scan_reports_a_hand_off_nothing_reads() -> None:
    bad = _GOOD.replace("    turn.flag = True\n", "    turn.flag = True\n    turn.orphan = 1\n")
    assert _synthetic(bad) == ["first:7 hands on turn.orphan, which nothing reads"]


# ── every facade import is read (what F401 stands for on the facade) ──────────


def _string_names(tree: ast.AST) -> set[str]:
    """Names inside string constants that parse as expressions: string annotations and
    ``cast("T | None", ...)`` targets, which a type checker reads."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval")
            except SyntaxError:
                continue
            names |= {n.id for n in ast.walk(parsed) if isinstance(n, ast.Name)}
    return names


def _names_read(tree: ast.AST) -> set[str]:
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return names | _string_names(tree)


def _module_level_imports(tree: ast.Module) -> dict[str, int]:
    """Names the module binds by an import at its top level or in a top-level
    ``if TYPE_CHECKING:`` block, with their line."""
    bound: dict[str, int] = {}
    for node in tree.body:
        group = node.body if isinstance(node, ast.If) else [node]
        for stmt in group:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    bound[(alias.asname or alias.name).split(".")[0]] = stmt.lineno
    return bound


#: Owner names the facade re-exports for OTHER modules to read (the gateway, the
#: dashboard, the messaging transport, the tests): no facade or owner function reads them,
#: so they are the facade's surface, exported at the base. Frozen: a re-export added
#: later is read by the facade or an owner, or it is dead, and this test says which.
_SURFACE_ONLY_REEXPORTS = frozenset("""
    _get_agent_for_session _review_drafts_get _review_drafts_pop adopt_default_agent
    adopt_slack_config cancel_background_tasks enable_yolo_with_ttl get_orch_cfg
    handle_interaction is_open_channel is_slack_session_trusted is_tracked_channel
    post_linked_approval refresh_phase_emojis resolve_linked_approval set_allowed_users
    set_dashboard_state set_open_channels set_orch_cfg set_owner_id set_tracking_channels
    set_yolo_mode
    """.split())


def _local_import_faults(tree: ast.Module, label: str) -> list[str]:
    """A function-local import that its own function never reads."""
    faults = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.Module(body=fn.body, type_ignores=[])
        direct = [
            node
            for stmt in fn.body
            for node in ast.walk(stmt)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        reads = _names_read(body)
        for node in direct:
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                if name not in reads:
                    faults.append(
                        f"{label}:{node.lineno} {fn.name} imports {name} and never reads it"
                    )
    return faults


def _unread_imports(
    facade_text: str, owner_texts: list[str], surface: frozenset[str] = frozenset()
) -> list[str]:
    facade = ast.parse(facade_text)
    owners = [ast.parse(text) for text in owner_texts]
    read = _names_read(
        ast.Module(
            body=[n for n in facade.body if not isinstance(n, (ast.Import, ast.ImportFrom))],
            type_ignores=[],
        )
    ).union(*(_names_read(tree) for tree in owners))
    faults = [
        f"facade:{line} imports {name} and nothing reads it"
        for name, line in sorted(_module_level_imports(facade).items())
        if name not in read and name not in surface and name != "annotations"
    ]
    faults += _local_import_faults(facade, "facade")
    for index, tree in enumerate(owners):
        faults += _local_import_faults(tree, f"owner{index}")
    return faults


def test_the_import_check_reports_a_dead_module_level_import() -> None:
    facade = "import json\nimport os\nfrom kiro_crew import x as y\n\n\ndef f():\n    return os\n"
    owner = "def g(turn):\n    return y\n"
    assert _unread_imports(facade, [owner]) == ["facade:1 imports json and nothing reads it"]


def test_the_import_check_reports_a_dead_function_local_import() -> None:
    facade = "def f():\n    from kiro_crew import x\n\n    return 1\n"
    assert _unread_imports(facade, []) == ["facade:2 f imports x and never reads it"]
    # Read in its own function: sound. Read only by another function: still dead.
    assert _unread_imports("def f():\n    from a import x\n\n    return x\n", []) == []
    elsewhere = "def f():\n    from a import x\n\n    return 1\n\n\ndef g():\n    return x\n"
    assert _unread_imports(elsewhere, []) == ["facade:2 f imports x and never reads it"]


def test_the_import_check_reports_a_dead_owner_reexport() -> None:
    facade = "from kiro_crew.slack.handler_runtime.x import a, b\n\n\ndef f():\n    return a\n"
    assert _unread_imports(facade, []) == ["facade:1 imports b and nothing reads it"]
    assert _unread_imports(facade, [], frozenset({"b"})) == []


def test_every_facade_import_is_read_by_the_facade_or_an_owner() -> None:
    owner_texts = [
        path.read_text(encoding="utf-8")
        for path in sorted(_RUNTIME_DIR.glob("*.py"))
        if path.name != "__init__.py"
    ]
    facade_text = _FACADE_PATH.read_text(encoding="utf-8")
    assert _unread_imports(facade_text, owner_texts, _SURFACE_ONLY_REEXPORTS) == []
    # Each surface-only name is still a facade binding, so the list cannot go stale.
    bound = _module_level_imports(ast.parse(facade_text))
    assert sorted(_SURFACE_ONLY_REEXPORTS - set(bound)) == []


# ── the sync-io gate reads every owner coroutine ─────────────────────────────


def _sync_io_gate() -> types.ModuleType:
    """``scripts/check_sync_io_in_async.py``, loaded by path (``scripts/`` is no package)."""
    path = repo_root() / "scripts" / "check_sync_io_in_async.py"
    spec = importlib.util.spec_from_file_location("_handler_sync_io_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Each new owner's coroutines with the moved code in them, and a call the gate names on
#: any receiver (``urlopen``) planted in each: the gate parses and walks that coroutine.
_OWNER_COROUTINES = (
    ("dispatch", "_dispatch_before_turn"),
    ("dispatch", "maybe_handle_keyword_command"),
    ("dispatch", "_handle_sessions_command"),
    ("turn", "_run_turn"),
    ("turn", "_replay_abandoned"),
    ("turn", "book_failure"),
    ("permission_ladder", "_screen_tool_call"),
    ("permission_ladder", "_answer_permission_request"),
    ("permission_ladder", "_reject_orphaned_tool"),
    ("permission_ladder", "_steer_host_deny"),
    ("permission_ladder", "_request_approval"),
    ("permission_ladder", "handle_interaction"),
    ("delivery", "_deliver_answer"),
)


def _coroutine(source: str, name: str) -> ast.AsyncFunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )


def _plant_in(source: str, name: str, statement: str) -> str:
    fn = _coroutine(source, name)
    first = fn.body[1] if isinstance(fn.body[0], ast.Expr) and fn.body[1:] else fn.body[0]
    lines = source.splitlines(keepends=True)
    lines.insert(first.lineno - 1, " " * first.col_offset + statement + "\n")
    return "".join(lines)


@pytest.mark.parametrize(("owner", "coroutine"), _OWNER_COROUTINES)
def test_the_sync_io_gate_flags_a_call_planted_in_an_owner_coroutine(
    owner: str, coroutine: str
) -> None:
    gate = _sync_io_gate()
    source = _source(owner)
    assert gate._violations_in_source(source) == []
    planted = _plant_in(source, coroutine, 'urlopen("http://127.0.0.1:1")')
    assert "http" in {family for _line, family, _why in gate._violations_in_source(planted)}


def test_every_owner_coroutine_is_in_the_planted_set() -> None:
    """A coroutine added to a new owner joins the plant list, or this fails."""
    listed = set(_OWNER_COROUTINES)
    for stem in {owner for owner, _ in _OWNER_COROUTINES}:
        tree = ast.parse(_source(stem))
        found = {
            (stem, node.name) for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
        }
        assert found <= listed, sorted(found - listed)


def test_the_planted_sleep_needs_the_owners_plain_import() -> None:
    """``delivery.py`` names ``time`` by its own import, so a ``time.sleep`` planted in its
    coroutine is seen; with ``time`` taken from the facade instead it would not be."""
    gate = _sync_io_gate()
    source = _source("delivery")
    assert "\nimport time\n" in source
    planted = _plant_in(source, "_deliver_answer", "time.sleep(1)")
    assert "sleep" in {family for _line, family, _why in gate._violations_in_source(planted)}
    hidden = source.replace("\nimport time\n", "\nfrom kiro_crew.slack.handler import time\n")
    planted = _plant_in(hidden, "_deliver_answer", "time.sleep(1)")
    assert "sleep" not in {family for _line, family, _why in gate._violations_in_source(planted)}
