# Scanner output contract

**Status:** normative. `schema_version 1`.
**Owner:** `crucible-research`. **Tracker:** `alpha-engine-config-I7809`.

This document is the single statement of *what the scanner is supposed to
produce*. Before it existed, the answer was the emergent sum of four code sites
— `data/scanner.py`, `data/scanner_orchestrator.py`, `data/scanner_specs.py`
and `scoring/universe_membership.py` — which drifted independently, so a
ranking cutover on 2026-07-22 left three separate labels asserting a ranking
the live path had stopped using. Nothing failed, because nothing had ever
written down what the labels were supposed to mean.

Any change to a cut's name, width, ranking basis or consumer changes this file
in the same commit. `tests/test_scanner_contract.py` asserts the emitted
artifacts against the tables below and fails when they disagree.

---

## 1. The two groups — a count-matched champion/challenger pair

**Brian's ruling 2026-08-20.** The scanner ranks the same ~900-name universe two
ways, at the same width, and the two compete. **The better performer is promoted
weekly, and the champion is what the sector teams research.**

| | **Group A — attractiveness** | **Group B — technical** |
|---|---|---|
| cut name | `attractiveness_top_60` | `tech_score_top_60` |
| ranks over | the full scored universe | the momentum-path-eligible universe |
| ranking basis | `attractiveness_rank` — the 6-pillar composite (quality, value, momentum, growth, stewardship, defensiveness) | `tech_score_rank` — RSI / MACD / MA50 / MA200 / 20-day momentum, equally weighted |
| momentum | **excluded.** The momentum pillar's weight lives in `s3://{bucket}/config/factor_attractiveness_weights.json` and is zero, per Brian's ruling 2026-08-17 (`alpha-engine-config-I7580`): the top-60 must capture ~1-year attractive names, not short-term movers. | **included.** `momentum_20d` is one of the five equally-weighted `tech_score` terms. |
| width | 60 | 60 — count-matched, so a win is never confounded with breadth |
| graded on | `topn_alpha_vs_population` at 21 / 126 / 252 sessions, against the population it narrowed — never against SPY (`alpha-engine-config-I7576`) | same |

**The champion pointer.** `s3://{bucket}/config/scanner_cut_champion.json` names
which arm is live. `universe_membership.live_cut_champion()` reads it;
`resolve_feed_cut()` returns that cut's tickers, and
`graph/research_graph.py::_resolve_agent_input_set` feeds them to the sector
teams. Absent pointer ⇒ `DEFAULT_CUT_CHAMPION` = `attractiveness_top_60`, the
standing champion. A pointer naming anything outside `PROMOTABLE_CUTS` **raises**
— a promotion engine that believes one arm is live while the funnel serves
another is the exact drift this contract exists to prevent.

**Fixed consumers, not subject to promotion:** `attractiveness_top_20` is the
predictor's daily universe (`PREDICTOR_UNIVERSE_CUT`, explicitly ruled unchanged
2026-08-20); `attractiveness_top_60` is the RAG corpus scope and Think Tank's
coverage window regardless of which arm holds the feed.

**Promotion refuses immature evidence.** The horizons that match a ~1-year
objective are 126 and 252 sessions. `tech_score_top_60` was first emitted
2026-08-20, so neither is measurable for months. Promoting on the 21-day block
selects for exactly the short-horizon behaviour the momentum-zero ruling removed
— the error recorded in `alpha-engine-config-I7580`. Until a horizon matures the
pointer holds the standing champion, and the engine says so rather than flipping
the feed on a number it has already called wrong.

### The promotion engine

`scoring/cut_promotion.py::run_cut_promotion`, invoked from
`lambda/scanner_handler.py` on every scanner run, immediately after
`build_cuts_leaderboard` — the moment the board it reads exists. Slot registry:
`CUT_PROMOTION_SLOT` (champion-challenger-policy.md §10).

| | value | why |
|---|---|---|
| evidence | `research/cuts_leaderboard/{date}.json`, `topn_alpha_vs_population` | the arm against the population it narrowed (`alpha-engine-config-I7576`) |
| decision horizon | **126 sessions** | the shorter of the two horizons matching the ~1-year objective, so it matures first |
| corroborating horizon | 252 sessions, **veto only** | a longer horizon may block a promotion the shorter one proposes, never propose one — the asymmetry is `I7580` |
| forbidden horizon | **21 sessions**, at any confidence | asserted at import; it is the block that produced `I7580` |
| evidence floor | both arms `n_dates_scored ≥ min_dates_for_inference` (5) | below it a per-date mean is an anecdote (`alpha-engine-config-I7542`) |
| hysteresis (§5.2) | margin `0.005` of mean lift, cooldown `28` days, symmetric on demotion | implemented, not waived — the §9.3 winner-take-all delta is scoped to the selection-producer slot and does not transfer |

**A decision is written on every evaluation, promote or hold**, to three keys
carrying one v1 document (`contracts/scanner_cut_champion.schema.json`):

| key | role |
|---|---|
| `config/scanner_cut_champion.json` | the live pointer `live_cut_champion()` reads |
| `config/apply_audit/scanner_cut_champion/{date}.json` | immutable dated history — the promote/hold series |
| `config/apply_audit/scanner_cut_champion/latest.json` | liveness proxy: a dead engine must not read as an engine that held |

Every record carries `decision`, a machine-readable `reason_code`, a prose
`reason`, and **both arms' `n_dates_scored` even when zero** — that count is the
number saying how far off a real decision is, and it is the measurability
surface for this slot. An evidence-shaped hold (immature, thin, board absent) is
the expected steady state and does not alert. A structural defect on the board
(duplicate rows, a missing horizon block) is recorded as a hold **and then
raised**: the defect is durable before the process is allowed to fail.

The engine is **cadence-agnostic** — its hysteresis is measured in calendar
days, not invocations — so it behaves identically before and after the weekly
cadence change below.

### Cadence

**Target: the scanner runs weekly** (Brian, 2026-08-20), one run producing the
week's cuts. Today it runs every weekday — a `Scanner` stage in both
`step_function.json` (weekly) and `step_function_daily.json` (weekday preopen) —
and `SCANNER_CUT_REFRESH_CADENCE` is unset, so the cut re-derives daily.

Closing that gap is step 3 of `alpha-engine-config-I7823`, and the order is
load-bearing: every consumer keyed on `{today}` fails loud the first morning the
scanner does not run. This document's step 1 — the feed resolving from
`universe_membership/latest.json` rather than a dated artifact — is what makes
the weekly cadence shippable at all.

## 2. Every cut emitted, and what its `basis` means

`universe_membership/{date}/membership.json::cuts`. Every entry carries a
`basis` string naming *how* membership was decided. A `basis` is a claim about
the producing function, and `tests/test_scanner_contract.py::test_every_cut_basis_matches_its_producer`
holds it to that.

| cut | width | `basis` | produced by | feeds |
|---|---|---|---|---|
| `attractiveness_top_20` | 20 | `attractiveness_rank` | `_rank_table(attractiveness)` | predictor universe |
| `attractiveness_top_25` | 25 | `attractiveness_rank` | same | historical series continuity |
| `attractiveness_top_60` | 60 | `attractiveness_rank` | same | RAG corpus scope, Think Tank window, and the sector teams whenever it holds the champion pointer |
| `scanner_gate_baseline_60` | 60 | `scanner_champion_rank` | the scanner slot's champion ARM, verbatim from `candidates.json::scanner_tickers` | **nothing** — a candidate-generation experiment, scored on the scanner leaderboard |
| `tech_score_top_60` | 60 | `tech_score_rank` | `_rank_tech_score` over `scan_path == "momentum"` rows | the sector teams whenever it holds the champion pointer (§1) |
| `scanner_top_20` | 20 | `tech_score_rank_within_cut` | top 20 by `tech_score` **of the champion's 60** | nothing live — churn diagnostics |
| `attractiveness_momzero_top_{20,60}` | 20/60 | `attractiveness_rank_momzero` | `momzero_attractiveness_for_run` | nothing — observe-only arm |
| `attractiveness_mom121_top_{20,60}` | 20/60 | `attractiveness_rank_mom121` | `challenger_attractiveness_for_run` | nothing — observe-only arm |

**The name `scanner_gate_baseline_60` is dated and deliberately not changed
here.** It predates the 2026-07-22 cutover and reads as a `tech_score` gate,
which it has not been since. It was already renamed once this month (from
`scanner_candidates`, `alpha-engine-config-I7578`, whose alias is still
emitted for the deprecation window; known live reader
`crucible-dashboard/loaders/universe_churn.py`). Renaming a load-bearing key
twice in four weeks costs consumers more than the stale word costs readers, so
`basis` and `role` carry the truth and the rename is bundled with the I7578
alias removal.

**`scanner_top_20` is scoped within the champion's cut, not over the universe.**
It answers "which 20 of the champion's 60 does `tech_score` like best?", not
"what is the universe's `tech_score` top 20?". The universe-wide question is
`tech_score_top_60`. These two were conflated before this contract existed.

---

## 3. The champion/challenger register for Group B

`data/scanner_specs.py::SCANNER_SPECS`. Exactly one entry is `kind="champion"`
and it is named by `LIVE_CHAMPION`. **The orchestrator applies
`SCANNER_SPECS[LIVE_CHAMPION].rank` — it does not import a ranking function
directly.** That indirection is the structural fix for the drift this contract
documents: before it, the live path called `_rank_momentum_sleeve` while the
register still declared a different champion, and no test could see the
disagreement.

| arm | kind | ranks on |
|---|---|---|
| `momentum_sleeve` | **champion** | `mean(z(momentum_20d), z(return_60d))` over the liquidity-eligible universe. Live since the 2026-07-22 `config#1186` cutover, promoted on measured lift over the displaced incumbent. |
| `tech_score_gate` | challenger | `tech_score` (RSI / MACD / MA50 / MA200 / 20-day momentum, equally weighted) over `scan_path == "momentum"` rows. **The displaced incumbent**, restored as a scored arm so the 2026-07-22 promotion stays measurable instead of being asserted from a one-off backtest. |
| `mom_12_1_sleeve` | challenger | `z(mom_12_1_pct)` — 12-1 skip-month momentum. Isolates the horizon question (`alpha-engine-config-I7544`). |

Every arm holds eligibility, width and clock constant and varies only the
ranking signal. Widths are count-matched to `momentum_top_n`.

### The champion is never also a challenger

`momentum_sleeve` was registered as a challenger *while it was already the live
ranking*, so the scanner leaderboard scored an arm against itself for four
weeks and alerted daily (`alpha-engine-config-I7808`). Two guards now make that
state unreachable:

1. `assert_registry_matches_live_path()` — the champion entry's `rank` must be
   the callable the orchestrator applies. A cutover that forgets the register
   fails at import, not four weeks later on a leaderboard.
2. `_vacuous_membership_collisions()` reports **near**-identity, not only exact
   identity, and an arm sharing the champion's ranking function is recorded
   `inapplicable` rather than scored — mirroring
   `universe_membership.momentum_arms_applicable()`, which already does this
   for the attractiveness arms.

---

## 4. Ordering

`candidates.json::scanner_tickers` is **rank-ordered by the live champion arm**
— `ScannerSpec.rank` sorts descending and slices, so list position is the
champion's own ranking by construction. Consumers may rely on this.

`membership.json::cuts.*.tickers` is **sorted alphabetically** — set semantics.
Ordering is not recoverable from a cut; read `candidates.json` for it.

---

## 5. Fail-loud boundaries

| producer | posture | why |
|---|---|---|
| live candidate cut (§4b) | falls back to `tech_score_gate`'s ranking if factor loadings are unavailable, and logs it | the trading day must not stop for a missing shadow input |
| `universe_membership` | **raises** on empty inputs or a violated cut invariant | the predictor resolves its universe from it; a silently-empty membership is indistinguishable from a real empty |
| shadow challenger arms | fail-soft per arm, alarmed, and record an explicit `scanner_shadow_status.v1` miss | one broken arm must not take out the live cut or the other arms |
| scanner leaderboard | fail-soft, never raises into the live path | observe-only |

---

## 6. Related

- `champion-challenger-policy.md` §3 (absent-is-a-miss), §4 (vacuity)
- `alpha-engine-config-I7808` — the vacuous leaderboard this contract's absence produced
- `alpha-engine-config-I7809` — this contract
- `alpha-engine-config-I7580` — momentum weight zero in Group A
- `alpha-engine-config-I4983` — the predictor's move off the gate cut
