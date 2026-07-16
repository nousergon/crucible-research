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
        from nousergon_lib.logging import setup_logging, get_flow_doctor
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
