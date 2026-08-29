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
| momentum | **included, at 1/6.** The momentum pillar's weight lives in `s3://{bucket}/config/factor_attractiveness_weights.json`. It was set to ZERO on 2026-08-17 (`alpha-engine-config-I7580`) and Brian REVERSED that on 2026-08-21 (`I7988`): equal weight is the champion again and the momentum-zero composite runs as the `attractiveness_momzero_top_60` shadow arm, so the question is measured before it is adopted rather than after. | **included.** `momentum_20d` is one of the five equally-weighted `tech_score` terms. |
| width | 60 | 60 — count-matched, so a win is never confounded with breadth |
| graded on | **decided** on the paired weekly difference vs the champion, chained (`research/cuts_weekly_ledger/ledger.parquet`, Brian's ruling 2026-08-24 / `alpha-engine-config-I8261`); still SCORED at 21 / 126 / 252 sessions on `topn_alpha_vs_population`, against the population it narrowed and never against SPY (`alpha-engine-config-I7576`) — 126 and 252 hold a veto, 21 holds nothing | same |

**The champion pointer.** `s3://{bucket}/config/scanner_cut_champion.json` names
which arm is live. `universe_membership.live_cut_champion()` reads it;
`resolve_feed_cut()` returns that cut's tickers, and the live producers
(`producers/no_agent.py`, `producers/single_agent.py`) read them through
`live_cut_champion()`. (The multi-agent sector teams read the same pointer via
`_resolve_agent_input_set` until that graph was retired 2026-07-12 and deleted
2026-08-20, alpha-engine-config-I7827.) Absent pointer ⇒ `DEFAULT_CUT_CHAMPION` = `attractiveness_top_60`, the
standing champion. A pointer naming anything outside `PROMOTABLE_CUTS` **raises**
— a promotion engine that believes one arm is live while the funnel serves
another is the exact drift this contract exists to prevent.

**`tech_score_top_60` is OBSERVE-ONLY as of 2026-08-21** (Brian ruling,
`alpha-engine-config-I8060`). `PROMOTABLE_CUTS` is `("attractiveness_top_60",)`
and `OBSERVE_ONLY_CUTS` carries `tech_score_top_60` plus the two attractiveness
challengers. All four are scored every cycle — on the weekly ledger and at
21 / 126 / 252 on the leaderboard; non-promotable is not non-measured — but only
the promotable set may hold the pointer, and `live_cut_champion` refuses the
rest regardless of what the promotion engine believes. With one promotable arm
there is nothing to decide, so the engine writes
`reason_code: "no_promotable_challenger"` every cycle rather than falling
through to a comparison it did not make. The arm was made promotable on
2026-08-20 and first emitted the same day, so it has never had a scored cohort;
Brian declined to restore it in the I8261 cutover for that reason — arming an
automatic pointer write before evidence exists re-creates the I8060 condition.
It returns to `PROMOTABLE_CUTS` on a ruling once it has weeks of measured
performance, which is a one-line registry edit.

**Fixed consumers, not subject to promotion:** `attractiveness_top_20` is the
predictor's daily universe (`PREDICTOR_UNIVERSE_CUT`, explicitly ruled unchanged
2026-08-20); `attractiveness_top_60` is the RAG corpus scope and Think Tank's
coverage window regardless of which arm holds the feed.

**Promotion decides on the chained weekly series** (Brian's ruling
2026-08-24, `alpha-engine-config-I8261`). The cut is re-formed weekly, so a
forward-window return from a cohort date measures a hold the re-cut guarantees
never happens, and consecutive cohort dates' windows overlap. The decision
metric is therefore the **paired weekly difference vs the champion** — net of
transaction cost, both legs read from `research/cuts_weekly_ledger/ledger.parquet`
and joined on the week — aggregated over the chained series. 126 and 252 are
demoted from decision basis to **corroborating vetoes**: a mature block may
block a promotion the weekly series proposes and may never propose one, and an
immature one is recorded non-blocking (§5.1 — you cannot gate on a statistic you
did not measure). 21 sessions is excluded from both roles: it is the block that
produced `alpha-engine-config-I7580`. `forbidden_horizons_days` retired with the
old basis; the property it held is now held by three import-time invariants in
`cut_promotion.py` (decision source pinned to the ledger; excluded horizons
disjoint from veto horizons; every veto horizon ≥ 126).

### The promotion engine

`scoring/cut_promotion.py::run_cut_promotion`, invoked from
`lambda/scanner_handler.py` on every scanner run, immediately after
`build_cuts_leaderboard` — the moment the board it reads exists. Slot registry:
`CUT_PROMOTION_SLOT` (champion-challenger-policy.md §10).

| | value | why |
|---|---|---|
| evidence | `research/cuts_weekly_ledger/ledger.parquet`, `net_log_return` | the return the slot actually earns over the week it is held; net because at weekly rebalance turnover is first-order and these arms' churn differs 42% vs 76% |
| decision metric | **`paired_weekly_net_log_return_vs_champion`**, chained | the same-week champion leg cancels the common market factor, which is the only thing that makes ~52 observations a year enough (`champion-challenger-policy.md` §4) |
| corroborating horizons | 126 and 252 sessions, **veto only** | a long horizon may block a promotion the weekly series proposes, never propose one — the asymmetry is `I7580`; immature ⇒ non-blocking (§5.1) |
| excluded horizon | **21 sessions** — neither decides nor vetoes | asserted disjoint from the veto set at import; it is the block that produced `I7580` |
| evidence floor | every arm `n_weeks_paired ≥ min_weeks_for_inference` (5) | below it a mean of paired weekly differences is an anecdote (`alpha-engine-config-I7542`) |
| earliest possible decision | **2026-09-25** | `FIRST_COHORT_DATE` 2026-08-20 (`I8255`) + 5 weekly holding periods. The v1 basis put it at 2027-02-22 |
| hysteresis (§5.2) | margin `0.0002` **per week**, cooldown `28` days, symmetric on demotion | implemented, not waived. The margin is the retired `0.005`-per-126-sessions bar converted to weekly units (0.005 / 25.2 weeks) — the cutover changes the BASIS, not the BAR |

**A decision is written on every evaluation, promote or hold**, to three keys
carrying one v2 document (`contracts/scanner_cut_champion.schema.json`):

| key | role |
|---|---|
| `config/scanner_cut_champion.json` | the live pointer `live_cut_champion()` reads |
| `config/apply_audit/scanner_cut_champion/{date}.json` | immutable dated history — the promote/hold series |
| `config/apply_audit/scanner_cut_champion/latest.json` | liveness proxy: a dead engine must not read as an engine that held |

Every record carries `decision`, a machine-readable `reason_code`, a prose
`reason`, and **every arm's `n_weeks_paired` even when zero** — that count is the
number saying how far off a real decision is, and it is the measurability
surface for this slot. An evidence-shaped hold (ledger absent, series thin, no
promotable challenger) is the expected steady state and does not alert. A
CORRUPT board — duplicate arm rows — is recorded as a hold **and then raised**:
the defect is durable before the process is allowed to fail. An ABSENT board is
neither, post-I8261: it makes every veto unmeasured and therefore non-blocking,
and a decision the ledger fully supports still goes through.

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
| `scanner_champion_60` | 60 | `scanner_champion_rank` | the scanner slot's champion ARM, verbatim from `candidates.json::scanner_tickers` | **nothing** — a candidate-generation experiment, scored on the scanner leaderboard |
| `tech_score_top_60` | 60 | `tech_score_rank` | the head of `tech_score_ranks`, over `scan_path == "momentum"` rows (§2a) | the sector teams whenever it holds the champion pointer (§1) |
| `scanner_top_20` | 20 | `tech_score_rank_within_cut` | top 20 by `tech_score` **of the champion's 60** | nothing live — churn diagnostics |
| `attractiveness_momzero_top_{20,60}` | 20/60 | `attractiveness_rank_momzero` | `momzero_attractiveness_for_run` | nothing — observe-only arm |
| `attractiveness_mom121_top_{20,60}` | 20/60 | `attractiveness_rank_mom121` | `challenger_attractiveness_for_run` | nothing — observe-only arm |

**The name is `scanner_champion_60` as of `alpha-engine-config-I7818`.** It was
`scanner_gate_baseline_60` before that — a name that predates the 2026-07-22
cutover and reads as a `tech_score` gate, which it has not been since — and
`scanner_candidates` before that (`alpha-engine-config-I7578`). I7578 deferred
this second rename deliberately: renaming a load-bearing key twice in four
weeks costs consumers more than the stale word costs readers. That reasoning
expired once the I7578 alias's deprecation window forced a consumer update
anyway, so I7818 did both moves in one change instead of two.
`scanner_gate_baseline_60` is now the deprecated alias, emitted for one
window; `scanner_candidates` is retired outright and no longer emitted. Known
live reader migrated in the same change:
`crucible-dashboard/loaders/universe_churn.py`.

**`scanner_top_20` is scoped within the champion's cut, not over the universe.**
It answers "which 20 of the champion's 60 does `tech_score` like best?", not
"what is the universe's `tech_score` top 20?". The universe-wide question is
`tech_score_top_60`. These two were conflated before this contract existed.

### 2a. One full-universe rank table per promotable basis

`alpha-engine-config-I7843`. A consumer resolves its rank ceiling in the basis
of whichever arm is **champion** — so every promotable arm needs a table wide
enough to answer it, or the arm is unpromotable in practice. Until this landed,
the only `tech_score` table emitted was `scanner_ranks` (60 names, ranked
*within* the champion's cut), so a `tech_score_top_60` champion could not say who
was rank 150 and `rank_table_for_cut()` refused rather than substituting the
attractiveness table.

| basis | field | ranks over | width (2026-08-20) |
|---|---|---|---|
| `attractiveness_rank` | `ranks` | the scanned universe with a rankable score | 902 |
| `tech_score_rank` | `tech_score_ranks` | scanned rows the momentum path admitted (`scan_path == "momentum"`) | 818 |

`rank_tables` is the **index**: `basis -> {field, rank_key, score_key, size,
population, serves_rank_ceiling, eligibility}`. A consumer resolves the field
from the artifact, not from a constant on its own side — adding a basis is then
a producer-side change with no matching edit in every reader, the same reason
`funnel.advances_to` is declared here. The tech table is legitimately narrower
than the universe (that gate is the incumbent rule's own), so its `size` is
declared rather than left for a consumer to discover by being refused.

`scanner_ranks` is unchanged and still emitted: the two tables answer two
questions, and collapsing them would answer one with the other.

### 2b. The ranked population is the SCANNED universe

`alpha-engine-config-I7844`. `ranks` and `scanner/universe/latest.json` are two
views of ONE Scanner invocation and were built from two different sources: the
board iterates `candidates.json::scanner_eval_log`, while the ranks came from
every key in `factors/profiles/{date}/by_ticker.json` — a file that legitimately
carries Metron-supplemental and fundamental-only rows the scanner never
evaluated. Measured 2026-08-20 on the live artifacts: **906 ranked vs 903 board
rows**, with `EQR` at rank 98 — inside Think Tank's `rank_ceiling: 150` — scored
on four of six pillars with no technical data at all (`momentum_n: 0`,
`low_vol_n: 0`, `sector: "Unknown"`), and the 3-name population difference
shifted the percentile of 860 of the 902 common names, first changing the
ordering at rank 26.

The profiles are now restricted to the scanned universe **before** the
cross-sectional scoring chokepoint — after would leave every surviving score
computed against a population including names nobody scanned. Every arm
(champion, `momzero`, `mom121`) is restricted identically, so no arm can differ
from another by which names it scored.

`population_reconciliation` states both sides and the difference. A ranked name
outside the scanned universe **raises**; scanned names with no rankable score
are listed in `unrankable` (the board carries them with
`attractiveness_score: null`) and raise only past a 5% allowance.

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

**Every registered arm is promotion-eligible.** Brian's ruling 2026-08-29,
verbatim: *"for the research arm, we should make all arms promote eligible,
including think tank"* — the principle, stated twice that day, being that **an
arm that is scored must be able to win**. So `PROMOTABLE_SPECS` is *derived
from* `SCANNER_SPECS` rather than restated: registering an arm makes it
promotable, with no second list to forget. A genuine exclusion is a declared
property carrying its reason (`DECLARED_INELIGIBLE_SPECS`, today empty), copied
onto every decision record — never an absence from a list.

### The promotion engine

`scoring/spec_promotion.py::run_spec_promotion`, invoked from
`lambda/scanner_handler.py` immediately after `build_scanner_leaderboard` — the
moment the board it reads exists. Slot registry: `SPEC_PROMOTION_SLOT`
(`champion-challenger-policy.md` §10). It shares nothing with the cut slot's
`CUT_PROMOTION_SLOT`: §2, separate axes, never conflated.

| | value | why |
|---|---|---|
| evidence source | `scanner/leaderboard/{date}.json`, the `horizons` block at 21 sessions | read from `horizons` and never from the artifact's top level — the two carry the same rows, but only the block carries the `status` saying whether the horizon is decidable |
| decision metric | **`topn_alpha_vs_champion`** — long-only top-N realized alpha minus the champion's, per shared cohort date, date-clustered | already differenced against the champion **on the cohort intersection**: a date enters only when both sides have a realized top-N return, so `n_dates` on this metric *is* the intersection size (§4, same cohort dates) |
| reported, never decided on | `topn_alpha_vs_population` | measured on each arm's OWN dates; ranking two arms by it compares one arm's month to another's quarter. The champion's cohort comes from the LIVE `candidates/` prefix and the challengers' from `candidates_shadow/`, so their cohorts always differ (`alpha-engine-config-I9274`) |
| count-matching | 50 on the board, and `momentum_top_n` at generation | no arm can win on breadth at either stage |
| evidence floor | `n_dates_paired ≥ 5` | below it a per-date mean is an anecdote (`alpha-engine-config-I7542`) |
| hysteresis (§5.2) | margin `0.00083` per 21-session window, cooldown `28` days, **symmetric on demotion** | implemented, not waived. Derived twice and the two agree: the fleet's ~1%/yr bar over 1/12 of a trading year is 0.01/12; the cut slot's 0.005-per-126-sessions scaled to 21 is 0.005 × 21/126 |
| earliest possible decision | **2026-09-28** | `FIRST_COHORT_DATE` 2026-08-21 — the first date more than one arm emitted a shadow set — plus 21 forward sessions plus 4 further cohort dates. A CEILING, not a promise: `tech_score_gate` has written shadows on 2026-08-21 and 2026-08-28 only, and that gap pushes the real date out |

**Demotion is symmetric with promotion.** A promotion moves the pointer away
from `DEFAULT_SPEC_CHAMPION` when a challenger clears the margin; a demotion
moves it back when the standing default clears the same margin in reverse, with
the same cooldown. It is recorded as its own `decision` value rather than as a
promotion of the default, because *a new arm won* and *the experiment was
reversed* are different facts about the slot and the series must be countable
both ways.

**A decision is written on every evaluation**, to three keys carrying one v1
document (`contracts/scanner_spec_champion.schema.json`):

| key | role |
|---|---|
| `config/scanner_spec_champion.json` | the live pointer `live_champion_spec()` reads |
| `config/apply_audit/scanner_spec_champion/{date}.json` | immutable dated history — the promote/demote/hold series |
| `config/apply_audit/scanner_spec_champion/latest.json` | liveness proxy: a dead engine must not read as an engine that held |

**`LIVE_CHAMPION` is no longer the mechanism by which the champion changes.**
`live_champion_spec()` resolves the WRITTEN pointer with the register entry as
the default, so a promotion moves the live ranking with no code edit — the same
shape `universe_membership.live_cut_champion()` already has, and the structural
fix for the I7808 drift. A pointer naming an arm outside `SCANNER_SPECS`
**raises** rather than serving the default. Before the pointer moves,
`register_for_champion()` constructs the target register and re-asserts its
coherence, so an incoherent target is refused rather than served.

**Three conditions are kept apart and must never be collapsed**, because
collapsing them is the misread this engine exists to prevent:

| `reason_code` | means |
|---|---|
| `no_eligible_challenger` | no challenger has been scored on ANY cohort date. No comparison was possible — **unanswerable, not a champion win** |
| `no_common_cohort` | a challenger IS scored, but on no date the champion also scored. Measured and incomparable |
| `champion_already_leads` | a real paired comparison happened and the incumbent won it |

On the measured board `scanner/leaderboard/2026-08-28.json` — champion
`momentum_sleeve` 7 dates at +0.031348 (t 11.41), both challengers **0** — the
engine writes `no_eligible_challenger`. The same distinction is carried per arm
as `eligible_for_promotion: false` plus an `ineligible_reason_code`, so an arm's
own block says why it could not win. An evidence-shaped hold is the expected
steady state and does not alert. A CORRUPT board — duplicate arm rows — is
recorded as a hold **and then raised**: the defect is durable before the process
is allowed to fail.

### The champion is never also a challenger

`momentum_sleeve` was registered as a challenger *while it was already the live
ranking*, so the scanner leaderboard scored an arm against itself for four
weeks and alerted daily (`alpha-engine-config-I7808`). Two guards now make that
state unreachable:

1. `assert_registry_coherent()` — runs at import: exactly one champion, named
   by `LIVE_CHAMPION`, carrying a `rank` the orchestrator can apply, and no
   challenger sharing that callable. A cutover that forgets the register fails
   at import, not four weeks later on a leaderboard. Its generalised form
   `assert_register_coherent(register, champion)` holds a register CONSTRUCTED
   for a promoted champion to the identical invariants, so a promotion cannot
   reach a state the import-time guard would have refused.
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
| `universe_membership` | **raises** on empty inputs, a violated cut invariant, a promotable basis with no full-universe rank table (§2a), or a ranked name outside the scanned universe (§2b) | the predictor resolves its universe from it; a silently-empty or silently-divergent membership is indistinguishable from a real one |
| shadow challenger arms | fail-soft per arm, alarmed, and record an explicit `scanner_shadow_status.v1` miss | one broken arm must not take out the live cut or the other arms |
| scanner leaderboard | fail-soft, never raises into the live path | observe-only |

---

## 6. Related

- `champion-challenger-policy.md` §3 (absent-is-a-miss), §4 (vacuity)
- `alpha-engine-config-I7808` — the vacuous leaderboard this contract's absence produced
- `alpha-engine-config-I7809` — this contract
- `alpha-engine-config-I7580` — momentum weight zero in Group A
- `alpha-engine-config-I4983` — the predictor's move off the gate cut
