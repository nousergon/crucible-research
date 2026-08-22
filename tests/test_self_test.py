"""Tests for `scoring/self_test.py` — the research module's known-answer
self-test (alpha-engine-config-I7262).

Four layers, and the last three are the load-bearing ones:

1. **The battery agrees on THIS runner.** Every case passes here too, so a CI
   failure and an in-Lambda failure mean the same thing and can be compared.
2. **The expectations are re-derived here, independently.** Each closed form is
   recomputed from the function's definition. If the module's own arithmetic were
   ever quietly changed to match the implementation, this layer is what notices.
3. **The runner's outcome taxonomy holds.** Disagreed => FAIL, could-not-run =>
   UNKNOWN, over-budget => FAIL (Brian ruling 2026-08-13). This is the part that
   decides whether a harness fault gets reported as a correctness regression.
4. **The battery can actually FAIL.** A self-test never shown to fail is not
   evidence. `test_a_perturbed_implementation_is_caught` is the standing,
   automated form of the manual perturbation recorded in the PR body.
"""

from __future__ import annotations

import json
import math

import pytest
from nousergon_lib.quant.selftest_perturbation import assert_perturbation_caught

from scoring import self_test as st

# ── layer 1: the real battery ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def body():
    return st.run_self_test(run_date="2026-08-15")


def test_every_case_passes_on_this_runner(body):
    failures = [c for c in body["cases"] if c["verdict"] != st.PASS]
    assert not failures, json.dumps(failures, indent=2, default=str)
    assert body["verdict"] == st.PASS
    assert body["n_cases"] == len(st.build_cases())


def test_all_four_case_classes_are_represented(body):
    """A case class silently dropped in a refactor is a coverage regression
    nothing else would notice: the artifact would still say PASS, on fewer
    questions."""
    names = {c["case"] for c in body["cases"]}
    assert any(n.endswith("_closed_form") for n in names)
    assert any(n.endswith("_metamorphic") for n in names)
    assert any(n.endswith("_degenerate") for n in names)
    assert any(n.endswith("_convention") for n in names)


def test_the_named_numeric_surfaces_are_all_covered(body):
    names = {c["case"] for c in body["cases"]}
    assert {
        "attractiveness_zblend_closed_form",
        "attractiveness_percentile_closed_form",
        "attractiveness_winsorization_closed_form",
        "composite_final_score_closed_form",
        "cross_sectional_rank_closed_form",
        "spearman_ic_closed_form",
        "date_clustered_se_closed_form",
        "rankdata_average_ties_closed_form",
    } <= names


def test_every_closed_form_case_asserts_to_1e_9(body):
    for case in body["cases"]:
        if case["case"].endswith("_closed_form"):
            assert case["tolerance"] <= 1e-9, case["case"]
            assert case["abs_error"] <= case["tolerance"], case["case"]


def test_no_case_buys_itself_a_loose_tolerance(body):
    """A tolerance wide enough to hide a convention change is a case that cannot
    fail. Only the one case measuring a divergence across a 6dp-rounded
    production value is allowed past 1e-9, and it is named here explicitly."""
    for case in body["cases"]:
        assert case["tolerance"] <= 1e-9, (
            f"{case['case']} carries tolerance {case['tolerance']} — justify it "
            "here or tighten it"
        )


def test_the_battery_completes_well_inside_its_budget(body):
    """It runs beside the weekly signal build; a battery that costs minutes would
    be the reason someone disables it."""
    assert body["wall_clock_seconds"] < 30.0


# ── layer 2: the expectations, re-derived from first principles ─────────────

_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_expected_zscore_uses_the_population_denominator():
    """The load-bearing convention: ddof=0, not ddof=1. The sample-sd alternative
    differs by sqrt(6/5) = 1.0954, so this assertion is what stops the
    expectation drifting to match a changed library."""
    n = len(_VALUES)
    mean = sum(_VALUES) / n
    pop = math.sqrt(sum((v - mean) ** 2 for v in _VALUES) / n)
    sample = math.sqrt(sum((v - mean) ** 2 for v in _VALUES) / (n - 1))
    assert st._expected_zscore(6.0) == pytest.approx((6.0 - mean) / pop, rel=0, abs=1e-15)
    assert st._expected_zscore(6.0) != pytest.approx((6.0 - mean) / sample, rel=1e-6)


def test_expected_spearman_is_the_rank_correlation_definition():
    """rho = 1 - 6*sum(d^2)/(n(n^2-1)); one adjacent transposition => sum(d^2)=2."""
    n = st._IC_N
    assert st._expected_spearman_ic() == pytest.approx(
        1.0 - 12.0 / (n * (n * n - 1)), rel=0, abs=1e-15)


def test_expected_clustered_se_uses_the_sample_denominator():
    """ddof=1, not ddof=0. The population alternative differs by sqrt(5/4) =
    1.118 and would overstate every t-stat the cutover gate reads."""
    assert st._expected_clustered_se() != pytest.approx(
        st._expected_clustered_se_population(), rel=1e-6)


def test_expected_macro_shift_is_the_overlay_definition():
    assert st._expected_macro_shift() == pytest.approx(5.0, rel=0, abs=1e-12)


def test_expected_composite_final_is_base_plus_shift_plus_boost():
    assert st._expected_composite_final() == pytest.approx(80.0, rel=0, abs=1e-12)


def test_the_ic_fixture_is_not_a_perfect_correlation():
    """A rho of exactly 1.0 passes under any implementation that merely preserves
    order, including a broken one."""
    assert 0.9 < st._expected_spearman_ic() < 1.0


def test_the_rank_fixture_contains_a_tie():
    """A tie is the ONLY input that distinguishes min-rank from dense-rank and
    ordinal. A fixture of distinct scores would pass under all three."""
    assert len(set(st._RANK_SCORES)) < len(st._RANK_SCORES)


def test_the_clip_fixture_is_large_enough_to_reach_the_bound():
    """With a POPULATION sd the largest attainable z in an n-name cross-section
    is (n-1)/sqrt(n) — 2.27 at n=7, first crossing 3.0 at n=11. A small fixture
    cannot exercise the +3.0 clip AT ALL, so it would pass while measuring
    nothing. This guards the fixture against shrinking back below that bound."""
    n = st._CLIP_FIXTURE_N
    assert (n - 1) / math.sqrt(n) > 3.0
    assert st._unclipped_outlier_z() > 3.0


# ── layer 3: the artifact and the outcome taxonomy ─────────────────────────

def test_artifact_carries_the_provenance_header(body):
    """The resolved library versions are the point — they are what makes this an
    instrument check rather than a code check."""
    assert body["schema"] == st.SCHEMA
    assert body["component"] == "research"
    assert body["run_date"] == "2026-08-15"
    assert body["python"]
    assert body["code_sha"]
    for dist in st._TRACKED_DISTRIBUTIONS:
        assert dist in body["libraries"], dist


def test_every_case_publishes_enough_to_re_derive_it(body):
    for case in body["cases"]:
        assert case["description"]
        assert case["inputs"]
        assert "units" in case["inputs"], case["case"]
        assert case["expected"] is not None


def test_known_gap_cases_say_so_in_words(body):
    """A pinned-wrong case must never read as an endorsement. The artifact has to
    carry that in words, not leave a reader to infer it from a green row.

    config-I7272 closed the last of them: _zmap and _pearson were already
    honest, and the nousergon-lib v0.124.3 → v0.124.70 pin bump delivered
    ``attractiveness._zscore``'s fix to this image. So the battery carries ZERO
    known gaps today, and ``n_known_gaps == 0`` is asserted rather than assumed
    — a count nobody checks is not a zero, it is an unread field. The per-case
    shape contract below still binds the moment a gap is re-introduced."""
    gaps = [c for c in body["cases"] if c.get("known_gap")]
    assert len(gaps) == body["n_known_gaps"] == 0, gaps
    for case in gaps:  # pragma: no cover — binds only once a gap returns
        assert case["gap_issue"].startswith("alpha-engine-config-I")
        assert "NOT" in case["known_gap_note"]
        assert "PINNED NOT FIXED" in case["description"] or "I7272" in case["description"]


def test_the_undefined_representation_finding_is_fixed_at_all_three_sites(body):
    """alpha-engine-config-I7272, ARRIVED. Three degenerate sites, all three now
    reporting undefined honestly ON THIS IMAGE:
    ``leaderboard_scoring._pearson`` (always was),
    ``attractiveness_trajectory._zmap`` (crucible-research#628), and
    ``nousergon_lib.quant.attractiveness._zscore`` — the last of the three,
    fixed at source in another repo and delivered here only now, by the
    nousergon-lib v0.124.3 → v0.124.70 pin bump in this same change.

    This case was deliberately pinned at 2.0 so that the day the pin moved it
    would go red and force this update in the SAME change. It did exactly that:
    the assertion below is what replaces it, and 3.0 is now the value that
    breaks if a future bump regresses any of the three."""
    case = next(c for c in body["cases"]
                if c["case"] == "undefined_representation_divergence_convention")
    assert case["expected"] == 3.0
    assert case["actual"] == 3.0
    assert case["inputs"]["total_sites"] == 3
    assert case.get("known_gap") is None


def test_the_zero_variance_attractiveness_leg_is_dropped_not_fabricated(body):
    """The scoring half of I7272 arriving: a pillar with zero cross-sectional
    spread is UNDEFINED, so its leg is dropped and the surviving weights
    renormalize. A name whose only pillar is degenerate has no measured position
    at all and carries ``attractiveness_raw`` None — where the old lib returned
    a fabricated 0.0 that then VOTED in the blend."""
    case = next(c for c in body["cases"]
                if c["case"] == "attractiveness_zero_variance_degenerate")
    assert case["expected"] == 1.0
    assert case["actual"] == 1.0
    assert case.get("known_gap") is None
    assert st._zero_variance_attractiveness() is None


def test_the_artifact_is_strict_json(body):
    """alpha-engine-config-I7237's third finding was a metrics.json that was not
    strict JSON. NaN/Infinity are not JSON, so a consumer using a strict parser
    would fail on the artifact rather than on the numbers."""
    text = json.dumps(body, default=str)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict by default


def test_a_disagreeing_case_is_FAIL_not_UNKNOWN():
    def cases():
        return [st.Case(name="wrong", description="d", inputs={"units": "u"},
                        expected=1.0, compute=lambda: 2.0)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["verdict"] == st.FAIL


def _raise():
    raise RuntimeError("the instrument could not be measured")


def test_a_case_that_could_not_run_is_UNKNOWN_not_FAIL():
    """Collapsing these two would make a broken image read as a correctness
    regression."""
    def cases():
        return [st.Case(name="boom", description="d", inputs={"units": "u"},
                        expected=1.0, compute=_raise)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["cases"][0]["verdict"] == st.UNKNOWN
    assert out["verdict"] == st.UNKNOWN


def test_a_battery_that_could_not_be_built_is_UNKNOWN_and_never_raises():
    def cases():
        raise ImportError("no numpy in this image")

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["verdict"] == st.UNKNOWN
    assert out["status"] == "error"
    assert out["n_cases"] == 0


def test_an_over_budget_case_is_FAIL():
    """Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN."""
    def slow():
        import time
        time.sleep(0.05)
        return 1.0

    def cases():
        return [st.Case(name="slow", description="d", inputs={"units": "u"},
                        expected=1.0, compute=slow)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases,
                           case_timeout_seconds=0.001)
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["cases"][0]["timed_out"] is True


def test_run_self_test_never_raises_whatever_the_provider_does():
    """The contract that keeps an accuracy instrument from taking down the
    pipeline it measures. `lambda: [1, 2, 3]` is the case that matters: a
    well-formed LIST of malformed items gets past the materialisation guard."""
    for provider in (lambda: None, lambda: [1, 2, 3], _raise):
        out = st.run_self_test(run_date="2026-08-15", case_provider=provider)
        assert out["verdict"] in (st.PASS, st.FAIL, st.UNKNOWN)


def test_a_malformed_case_is_UNKNOWN_and_named():
    out = st.run_self_test(run_date="2026-08-15", case_provider=lambda: [1, 2])
    assert out["verdict"] == st.UNKNOWN
    assert out["n_cases"] == 2
    assert all("malformed" in c["case"] for c in out["cases"])


def test_verdict_is_pass_is_strict():
    assert st.verdict_is_pass("PASS")
    for other in (None, "ok", "pass", "OK", "UNKNOWN", ""):
        assert not st.verdict_is_pass(other)


# ── layer 4: the battery can FAIL — the acceptance criterion for I7262 ──────

def test_a_perturbed_winsorization_clip_is_caught(monkeypatch):
    """THE acceptance criterion (alpha-engine-config-I7262): a self-test never
    shown to fail is not evidence.

    Perturbs the winsorization bound by 1e-4 relative — far below any realistic
    edit — and asserts the battery goes FAIL. This fleet has shipped several
    detectors that could not fail; this is the standing guard against adding
    another.

    Delegates to the LIFTED helper (alpha-engine-config-I7238/I7262) — the same
    monkeypatch/rerun/assert-FAIL shape crucible-predictor's test suite carries
    independently; both now import one proven-correct implementation from
    ``nousergon_lib.quant.selftest_perturbation`` instead of each keeping its
    own copy.
    """
    assert_perturbation_caught(
        monkeypatch, module_path="nousergon_lib.quant.attractiveness",
        attr="_ZSCORE_CLIP", perturbed=3.0001,
        run=lambda: st.run_self_test(run_date="2026-08-15"),
        case_name="zscore_clip_convention",
    )


def test_a_perturbed_composite_weight_is_caught(monkeypatch):
    assert_perturbation_caught(
        monkeypatch, module_path="scoring.composite", attr="DEFAULT_W_QUANT",
        perturbed=0.5001, run=lambda: st.run_self_test(run_date="2026-08-15"),
        case_name="composite_weights_convention",
    )


def test_a_perturbed_horizon_is_caught(monkeypatch):
    assert_perturbation_caught(
        monkeypatch, module_path="scoring.leaderboard_scoring",
        attr="DEFAULT_HORIZON_DAYS", perturbed=22,
        run=lambda: st.run_self_test(run_date="2026-08-15"),
        case_name="horizon_days_convention",
    )


def test_perturbing_the_clustered_se_denominator_is_caught(monkeypatch):
    """The ddof convention, perturbed at the production function rather than at a
    constant — the shape the sqrt(365)-vs-sqrt(252) defect (I7236) actually took.

    Also asserts the METAMORPHIC cases do NOT fire: a ddof change preserves every
    invariance relation, which is exactly why closed-form and metamorphic cases
    are both needed and neither substitutes for the other.
    """
    import math as _math

    def population_se(per_date):
        vals = [float(v) for v in per_date]
        n = len(vals)
        if n == 0:
            return None
        mean = sum(vals) / n
        if n == 1:
            return {"mean": round(mean, 6), "se": None, "t_stat": None, "n_dates": 1}
        sd = _math.sqrt(sum((v - mean) ** 2 for v in vals) / n)  # the perturbation
        se = sd / _math.sqrt(n)
        return {"mean": round(mean, 6), "se": round(se, 6),
                "t_stat": round(mean / se, 4) if se > 0 else None, "n_dates": n}

    out = assert_perturbation_caught(
        monkeypatch, module_path="scoring.leaderboard_scoring",
        attr="date_clustered_stats", perturbed=population_se,
        run=lambda: st.run_self_test(run_date="2026-08-15"),
        case_name="date_clustered_se_closed_form",
    )
    failed = {c["case"] for c in out["cases"] if c["verdict"] == st.FAIL}
    assert "date_clustered_ddof_convention" in failed
    assert not any(n.endswith("_metamorphic") for n in failed)


def test_perturbing_the_rank_tie_rule_is_caught(monkeypatch):
    """Min-rank -> dense-rank. A tie-rule change is invisible to every case
    except the one built on a tie, which is why the fixture carries one."""
    from scoring import aggregator

    original = aggregator.assign_cross_sectional_ranks

    def dense(results):
        original(results)
        # Collapse [1, 2, 2, 4] to [1, 2, 2, 3].
        ordered = sorted({r["cross_sectional_rank"] for r in results.values()})
        remap = {old: i + 1 for i, old in enumerate(ordered)}
        for r in results.values():
            r["cross_sectional_rank"] = remap[r["cross_sectional_rank"]]

    assert_perturbation_caught(
        monkeypatch, module_path="scoring.aggregator",
        attr="assign_cross_sectional_ranks", perturbed=dense,
        run=lambda: st.run_self_test(run_date="2026-08-15"),
        case_name="cross_sectional_rank_closed_form",
    )


# ── the console surface ─────────────────────────────────────────────────────

def test_console_envelope_matches_the_published_contract(body):
    """FORK-DETECTION BACKSTOP (shared-code-policy.md §5.1).

    The envelope is built here rather than imported from
    `nousergon_lib.fleet_check_result` because that module first ships in lib
    v0.124.29 and this repo pins v0.124.3 (see the module docstring). This test
    is what holds the interim duplication legitimate: every field the console's
    checks-envelope adapter reads must be present and correctly typed.
    Migration is tracked in alpha-engine-config-I7274.
    """
    env = st.console_envelope(body)
    for field in ("schema_version", "check_id", "label", "ran_at", "status",
                  "summary", "cadence_minutes", "deep_link", "findings"):
        assert field in env, field
    assert env["schema_version"] == 1
    assert env["check_id"] == st.CHECK_ID
    assert "/" not in env["check_id"], "check_id must be a single path segment"
    assert env["status"] in ("ok", "attention", "error")
    assert isinstance(env["cadence_minutes"], int) and env["cadence_minutes"] > 0
    assert isinstance(env["findings"], list)
    json.loads(json.dumps(env, default=str))


def test_console_envelope_key_is_where_the_adapter_looks():
    assert st.console_envelope_key() == f"ops/checks/{st.CHECK_ID}/latest.json"


def test_console_status_never_renders_unknown_as_green():
    """`principles.md` §2.7: no data is never rendered as green."""
    assert st.console_envelope({"verdict": st.PASS})["status"] == "ok"
    assert st.console_envelope({"verdict": st.FAIL})["status"] == "error"
    assert st.console_envelope({"verdict": st.UNKNOWN})["status"] == "attention"
    assert st.console_envelope({})["status"] == "attention"


def test_console_envelope_lists_every_non_passing_case():
    def cases():
        return [
            st.Case(name="ok", description="d", inputs={"units": "u"},
                    expected=1.0, compute=lambda: 1.0),
            st.Case(name="bad", description="d", inputs={"units": "u"},
                    expected=1.0, compute=lambda: 9.0),
        ]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    env = st.console_envelope(out)
    assert [f["key"] for f in env["findings"]] == ["bad"]


def test_publish_console_row_never_raises_on_a_broken_client(body):
    """A check must not go red because its telemetry did."""
    class Boom:
        def put_object(self, **kwargs):
            raise RuntimeError("s3 is down")

    assert st.publish_console_row(body, s3_client=Boom()) is None


def test_publish_console_row_writes_the_expected_key(body):
    captured = {}

    class Capture:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    uri = st.publish_console_row(body, s3_client=Capture())
    assert uri == f"s3://{st.RESEARCH_BUCKET}/ops/checks/{st.CHECK_ID}/latest.json"
    assert captured["Bucket"] == st.RESEARCH_BUCKET
    assert captured["ContentType"] == "application/json"
    assert json.loads(captured["Body"])["check_id"] == st.CHECK_ID


def test_publish_console_row_dry_run_writes_nothing(body):
    assert st.publish_console_row(body, dry_run=True) is None


# ── the S3 artifact ─────────────────────────────────────────────────────────

def test_self_test_key_is_beside_the_run():
    assert st.self_test_key("2026-08-15") == "research/2026-08-15/self_test.json"


def test_write_self_test_writes_the_body_verbatim(body):
    captured = {}

    class Capture:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    key = st.write_self_test("bucket", "2026-08-15", body, s3_client=Capture())
    assert key == "research/2026-08-15/self_test.json"
    assert json.loads(captured["Body"])["verdict"] == body["verdict"]


# ── the handler call site ───────────────────────────────────────────────────

def _handler_module():
    import importlib
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root / "lambda") not in sys.path:
        sys.path.insert(0, str(root / "lambda"))
    return importlib.import_module("handler")


def test_handler_emitter_publishes_the_verdict(monkeypatch):
    import datetime

    h = _handler_module()
    captured = {}
    monkeypatch.setattr(st, "write_self_test",
                        lambda b, d, r, **k: captured.update(body=r) or "key")
    monkeypatch.setattr(st, "publish_console_row",
                        lambda r, **k: captured.update(console=True))

    h._maybe_emit_self_test(datetime.date(2026, 8, 15))
    assert captured["body"]["verdict"] == st.PASS
    assert captured["console"] is True


def test_handler_emitter_never_raises_when_everything_fails(monkeypatch):
    """A §2.3a verdict stage that dies must not kill the stages that do not
    depend on it — the weekly briefing is the primary deliverable."""
    import datetime

    h = _handler_module()

    def boom(*a, **k):
        raise RuntimeError("s3 is down")

    monkeypatch.setattr(st, "run_self_test", boom)
    h._maybe_emit_self_test(datetime.date(2026, 8, 15))  # must not raise

    monkeypatch.undo()
    monkeypatch.setattr(st, "write_self_test", boom)
    monkeypatch.setattr(st, "publish_console_row", boom)
    h._maybe_emit_self_test(datetime.date(2026, 8, 15))  # must not raise
