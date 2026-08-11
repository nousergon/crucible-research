"""Verify flow-doctor wiring in research-module entrypoints.

Asserts the canonical alpha-engine-lib pattern (module-top setup_logging
+ exclude_patterns plumbed + yaml resolvable from the entrypoint
location) is in place for both research Lambdas:

- ``lambda/handler.py``         — main research pipeline (Saturday SF)
- ``lambda/alerts_handler.py``  — intraday price alerts (every 30 min)

Also locks in the deletion of the dead ``state["flow_doctor"]`` LangGraph
threading: 4 injection sites with zero downstream consumers were
removed in this PR; a regression check prevents quiet re-introduction.

Runs without firing any LLM diagnosis: ``setup_logging`` is exercised
with FLOW_DOCTOR_ENABLED=1 + stub env vars + a redirected yaml store
path, but no ERROR records are emitted (so flow-doctor's report() /
diagnose() pipeline is never triggered — no Anthropic calls, no email,
no GitHub issue).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stub_flow_doctor_env(monkeypatch):
    """Populate the env vars that flow-doctor.yaml's ${VAR} refs resolve.

    flow_doctor.init() substitutes these at load time. Stubs are non-empty
    strings; nothing actually contacts SMTP/GitHub since no report() fires.
    """
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    monkeypatch.setenv("FLOW_DOCTOR_SKIP_PREFLIGHT", "1")
    monkeypatch.setenv("EMAIL_SENDER", "test@example.com")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "test@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "stub-password")
    monkeypatch.setenv("FLOW_DOCTOR_GITHUB_TOKEN", "stub-token")
    # T3 flow-doctor.yaml telegram forum-topic notifiers (config#1749).
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:stub-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100stub")
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL", "1")
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH", "2")


@pytest.fixture
def reset_root_logger():
    """Snapshot + restore root logger handlers around each test."""
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    root.handlers = saved


@pytest.fixture
def temp_flow_doctor_yaml(tmp_path):
    """Write a copy of the production flow-doctor.yaml with its store
    block forced to a local sqlite file under tmp_path.

    Production now points store.type at the shared DynamoDB dedup table
    (alpha-engine-config#2418) so dedup_cooldown_minutes survives across
    separate process/Lambda invocations. Wiring tests only need to verify
    that setup_logging() attaches a FlowDoctorHandler and plumbs
    exclude_patterns — they must never touch live AWS credentials/tables,
    so the store type is unconditionally overridden here regardless of
    what the real flow-doctor.yaml declares.
    """
    import yaml as yamllib
    with open(REPO_ROOT / "flow-doctor.yaml") as f:
        cfg = yamllib.safe_load(f)
    cfg["store"] = {
        "type": "sqlite",
        "path": str(tmp_path / "flow_doctor_test.db"),
    }
    yaml_path = tmp_path / "flow-doctor.yaml"
    with open(yaml_path, "w") as f:
        yamllib.safe_dump(cfg, f)
    return str(yaml_path)


def _flow_doctor_available() -> bool:
    try:
        import flow_doctor  # noqa: F401
        return True
    except ImportError:
        return False


flow_doctor_required = pytest.mark.skipif(
    not _flow_doctor_available(),
    reason="flow-doctor not installed (pip install alpha-engine-lib[flow_doctor])",
)


class TestFlowDoctorYamlPresence:
    """The yaml file each entrypoint resolves must exist at that path."""

    def test_yaml_at_repo_root_exists(self):
        assert (REPO_ROOT / "flow-doctor.yaml").is_file()

    def test_yaml_path_resolved_by_handler_exists(self):
        # Mirrors lambda/handler.py's path computation:
        #   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        handler_path = REPO_ROOT / "lambda" / "handler.py"
        resolved = Path(os.path.dirname(os.path.dirname(os.path.abspath(handler_path)))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"handler.py resolves to {resolved}"

    def test_yaml_path_resolved_by_alerts_handler_exists(self):
        ah_path = REPO_ROOT / "lambda" / "alerts_handler.py"
        resolved = Path(os.path.dirname(os.path.dirname(os.path.abspath(ah_path)))) / "flow-doctor.yaml"
        assert resolved.is_file(), f"alerts_handler.py resolves to {resolved}"


class TestFlowDoctorYamlSchema:
    """flow-doctor.yaml must declare keys consistent with the lib contract."""

    def test_yaml_has_required_top_level_keys(self):
        import yaml
        with open(REPO_ROOT / "flow-doctor.yaml") as f:
            cfg = yaml.safe_load(f)
        for key in ("flow_name", "repo", "notify", "store", "rate_limits"):
            assert key in cfg, f"missing top-level key: {key}"
        assert cfg["repo"] == "nousergon/crucible-research"

    def test_yaml_has_email_notify_channel(self):
        import yaml
        with open(REPO_ROOT / "flow-doctor.yaml") as f:
            cfg = yaml.safe_load(f)
        types = {n.get("type") for n in cfg.get("notify", [])}
        assert "email" in types, "email channel required for ops alerts"


@flow_doctor_required
class TestSetupLoggingAttach:
    """setup_logging() should attach FlowDoctorHandler when ENABLED=1.

    Does NOT fire any ERROR records, so flow-doctor's diagnose() / Anthropic
    calls are never invoked. Verifies wiring shape only.
    """

    def test_disabled_attaches_no_flow_doctor_handler(self, monkeypatch, reset_root_logger):
        monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "0")
        from nousergon_lib.logging import setup_logging
        setup_logging(
            "research-test-disabled",
            flow_doctor_yaml=str(REPO_ROOT / "flow-doctor.yaml"),
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert attached == [], "FlowDoctorHandler should NOT attach when DISABLED"

    def test_enabled_attaches_flow_doctor_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from nousergon_lib.logging import get_flow_doctor, setup_logging
        setup_logging(
            "research-test-enabled",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=[],
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        assert get_flow_doctor() is not None

    def test_exclude_patterns_plumbed_to_handler(
        self, stub_flow_doctor_env, reset_root_logger, temp_flow_doctor_yaml
    ):
        from nousergon_lib.logging import setup_logging
        patterns = [r"langgraph retry exhausted", r"anthropic 5\d\d transient"]
        setup_logging(
            "research-test-patterns",
            flow_doctor_yaml=temp_flow_doctor_yaml,
            exclude_patterns=patterns,
        )
        import flow_doctor
        attached = [h for h in logging.getLogger().handlers
                    if isinstance(h, flow_doctor.FlowDoctorHandler)]
        assert len(attached) == 1
        compiled = attached[0]._exclude_re
        assert [p.pattern for p in compiled] == patterns


class TestEntrypointModuleTopWiring:
    """Each entrypoint must call setup_logging at MODULE-TOP, not inside a
    function. Source-text checks; no flow_doctor.init() side effects.
    """

    @staticmethod
    def _index_of(needle: str, text: str) -> int:
        idx = text.find(needle)
        assert idx != -1, f"missing required text: {needle!r}"
        return idx

    def test_handler_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "lambda" / "handler.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        handler_def_idx = self._index_of("def handler(", text)
        assert setup_idx < handler_def_idx, (
            "setup_logging must be called at module-top, before def handler()"
        )
        assert "exclude_patterns=" in text[setup_idx:handler_def_idx]

    def test_alerts_handler_calls_setup_logging_at_module_top(self):
        text = (REPO_ROOT / "lambda" / "alerts_handler.py").read_text()
        setup_idx = self._index_of("setup_logging(", text)
        handler_def_idx = self._index_of("def handler(", text)
        assert setup_idx < handler_def_idx
        assert "exclude_patterns=" in text[setup_idx:handler_def_idx]


class TestNoBarePrintsInHandlers:
    """Lock in the migration of bare ``print()`` calls to ``logger``.

    Audit 2026-05-01 found 21 prints in handler.py + 4 in
    alerts_handler.py — all bypassed setup_logging and never reached
    flow-doctor's ERROR escalation. Each was migrated to the
    appropriate logger.{info,warning,error}() call. Re-introductions
    silently re-open the bypass class, so this regression check fails
    the suite if a `print(` reappears outside comments/docstrings.
    """

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        import re
        # Remove triple-quoted blocks (docstrings + multi-line strings).
        stripped = re.sub(r'"""[\s\S]*?"""', "", text)
        # Remove full-line comments.
        stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
        return stripped

    def test_handler_has_no_bare_print(self):
        text = (REPO_ROOT / "lambda" / "handler.py").read_text()
        stripped = self._strip_comments_and_docstrings(text)
        assert "print(" not in stripped, (
            "bare print() found in lambda/handler.py — convert to "
            "logger.info/warning/error so the record propagates through "
            "flow-doctor's root handler"
        )

    def test_alerts_handler_has_no_bare_print(self):
        text = (REPO_ROOT / "lambda" / "alerts_handler.py").read_text()
        stripped = self._strip_comments_and_docstrings(text)
        assert "print(" not in stripped, (
            "bare print() found in lambda/alerts_handler.py — convert to "
            "logger.info/warning/error"
        )


class TestAlertsHandlerHasLogger:
    """alerts_handler.py defines its own logger.

    Without this, the print()-to-logger migration above would silently
    NameError at runtime. Catches the case where a future refactor
    drops the ``logger = logging.getLogger(__name__)`` declaration.
    """

    def test_alerts_handler_defines_logger(self):
        text = (REPO_ROOT / "lambda" / "alerts_handler.py").read_text()
        assert "logger = logging.getLogger(__name__)" in text


class TestNoDeadFlowDoctorPlumbing:
    """Lock in the deletion of the dead ``state["flow_doctor"]`` injections.

    Audit 2026-05-01 found 4 injection sites at handler.py lines 287/316/347/373
    with zero downstream consumers — pure cargo cult. Removed in this PR.
    These tests prevent quiet re-introduction.

    If a graph node ever LEGITIMATELY needs flow-doctor as an explicit
    consumer, prefer ``from alpha_engine_lib.logging import get_flow_doctor``
    in the node module itself rather than threading via state — keeps the
    dependency local and the test trivial to update.
    """

    def test_handler_does_not_thread_flow_doctor_via_state(self):
        text = (REPO_ROOT / "lambda" / "handler.py").read_text()
        assert 'state["flow_doctor"]' not in text
        assert "state['flow_doctor']" not in text
        # The get_flow_doctor import is dropped too — handler.py only needs
        # setup_logging now. (alerts_handler.py likewise.)
        assert "from alpha_engine_lib.logging import setup_logging, get_flow_doctor" not in text
        assert "from alpha_engine_lib.logging import get_flow_doctor" not in text

    def test_no_graph_node_consumes_state_flow_doctor(self):
        """If a node ever introduces state["flow_doctor"] as a real consumer,
        update this test + handler.py to thread it back in. Today there are
        zero consumers."""
        graph_dir = REPO_ROOT / "graph"
        if not graph_dir.is_dir():
            pytest.skip("graph/ directory not present in this checkout")
        for py in graph_dir.rglob("*.py"):
            content = py.read_text()
            assert 'state["flow_doctor"]' not in content, (
                f"new state['flow_doctor'] consumer in {py.relative_to(REPO_ROOT)}; "
                "if intentional, restore the injection in handler.py + update this test"
            )
            assert "state['flow_doctor']" not in content, (
                f"new state['flow_doctor'] consumer in {py.relative_to(REPO_ROOT)}"
            )


class TestThinkTankNamesItsOwnComponent:
    """An alert names the COMPONENT, not the repo (alpha-engine-config-I6910).

    `flow-doctor.yaml` declares `flow_name: research-lambda` for the whole
    repo. The Think Tank stopped being a Lambda on 2026-07-29 when it moved to
    a self-terminating EC2 spot box (ARCHITECTURE §47) — the old
    `alpha-engine-research-thinktank` function last logged on 2026-07-30. For
    the twelve days after, every Think Tank abort paged under the name of a
    component with no logs to go and read, and the operator went looking at
    Lambdas for a failure that had none.
    """

    def _handler_source(self) -> str:
        return (REPO_ROOT / "lambda" / "thinktank_handler.py").read_text()

    def test_setup_logging_is_called_with_the_component_name(self):
        assert 'flow_name="thinktank-spot"' in self._handler_source(), (
            "thinktank_handler must file its alerts under `thinktank-spot`; "
            "without the override it inherits flow-doctor.yaml's repo-wide "
            "`research-lambda`, which names a compute substrate it does not "
            "run on"
        )

    def test_the_yaml_still_declares_the_repo_wide_default(self):
        """Pins WHY the override is needed. If the yaml is ever changed to a
        per-component name, this test failing is the prompt to re-examine
        whether the override is still the right mechanism — not a reason to
        delete the assertion above."""
        yaml_text = (REPO_ROOT / "flow-doctor.yaml").read_text()
        assert "flow_name: research-lambda" in yaml_text

    def test_krepis_accepts_the_kwarg(self):
        """The CAPABILITY, not the version string — mirroring the krepis-floor
        contract tests requirements.txt points at. An older krepis raises
        TypeError on this kwarg rather than degrading, so a stale cached layer
        would break the handler at import, not merely lose the label."""
        import inspect

        from nousergon_lib.logging import setup_logging

        assert "flow_name" in inspect.signature(setup_logging).parameters

    def test_the_log_prefix_is_unchanged(self):
        """`name` and `flow_name` are separate knobs. Nothing about correcting
        the alert label should rewrite twelve days of log-grep habits."""
        assert '"thinktank",' in self._handler_source()


class TestEveryEntryPointNamesItsOwnComponent:
    """`flow-doctor.yaml` declares ONE `flow_name` for the whole repo; this repo
    ships fourteen entry points (alpha-engine-config-I6910 D3).

    Two things follow from sharing it, and both bit:

    - **The label misdirects.** `research-lambda` sent an operator to Lambda
      logs for a Think Tank failure that had none, twelve days after that
      component moved to an EC2 spot box.
    - **The budget is shared.** As of flow-doctor 0.10.0 the daily alert budget
      is scoped per `flow_name` (alpha-engine-config-I6921); components sharing
      one name share one budget, so a noisy neighbour spends everyone's.

    This test is the reason a fourteenth handler cannot quietly inherit the
    default: adding one without a `flow_name` fails here.
    """

    def _handlers(self):
        return sorted((REPO_ROOT / "lambda").glob("*_handler.py")) + [
            REPO_ROOT / "lambda" / "handler.py"
        ]

    def _flow_name(self, path):
        m = re.search(r'flow_name="([^"]+)"', path.read_text())
        return m.group(1) if m else None

    def test_every_handler_declares_a_flow_name(self):
        missing = [
            p.name for p in self._handlers()
            if "setup_logging(" in p.read_text() and self._flow_name(p) is None
        ]
        assert not missing, (
            f"{missing} call setup_logging without flow_name, so they file "
            f"alerts under flow-doctor.yaml's repo-wide `research-lambda` — a "
            f"name that identifies neither the component nor, since the Think "
            f"Tank migration, its compute substrate"
        )

    def test_no_two_handlers_share_a_flow_name(self):
        """A shared name is a shared alert budget and an ambiguous page."""
        seen = {}
        for p in self._handlers():
            flow = self._flow_name(p)
            if flow is None:
                continue
            seen.setdefault(flow, []).append(p.name)
        dupes = {f: n for f, n in seen.items() if len(n) > 1}
        assert not dupes, f"flow_name collisions: {dupes}"

    def test_no_handler_reintroduces_the_repo_wide_default(self):
        offenders = [
            p.name for p in self._handlers()
            if self._flow_name(p) == "research-lambda"
        ]
        assert not offenders, (
            f"{offenders} pin the repo-wide default explicitly. If a handler "
            f"genuinely wants it, delete the kwarg — restating it defeats the "
            f"two tests above without saying so."
        )
