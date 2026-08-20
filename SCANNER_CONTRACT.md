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

## 1. The two groups

The scanner emits **two independently-ranked cuts of the same ~900-name
scanned universe**, at the same width. They are not variants of one selection;
they answer different questions and feed different consumers.

| | **Group A — attractiveness** | **Group B — technical candidates** |
|---|---|---|
| cut name | `attractiveness_top_60` | `scanner_gate_baseline_60` |
| ranks over | the full scored universe | the momentum-path-eligible universe |
| ranking basis | `attractiveness_rank` — the 6-pillar composite (quality, value, momentum, growth, stewardship, defensiveness) | the scanner slot's **live champion arm** (§3) |
| momentum | **excluded.** The momentum pillar's weight is set by `s3://{bucket}/config/factor_attractiveness_weights.json` and is currently zero, per Brian's ruling 2026-08-17 (`alpha-engine-config-I7580`): the top-60 must capture ~1-year attractive names, not short-term movers. | **included, and it is the whole signal.** Every arm registered in this group ranks on some definition of price momentum. |
| horizon it serves | ~1 year | weeks to months |
| consumers | RAG corpus scope, Think Tank coverage window; its head `attractiveness_top_20` is the predictor's daily scoring universe (`PREDICTOR_UNIVERSE_CUT`) | the sector teams' input set (`graph/research_graph.py::_resolve_agent_input_set`, via `candidates.json::scanner_tickers`) |
| produced by | `scoring/universe_membership.py` from `scanner/universe/{date}/universe.json` | `data/scanner_orchestrator.py` §4b from the live champion arm |

**Why both exist.** Group A is a selection *rule* — a stable, slow-moving
ranking whose membership should persist across cycles. Group B is a *gate* — a
technical screen whose membership is expected to turn over. Conflating them is
what `alpha-engine-config-I4983` corrected on the predictor side; keeping them
separate is what this contract exists to hold.

**Momentum appears in exactly one of the two.** If it is ever restored to a
non-zero weight in Group A, `momentum_arms_applicable()` re-enables the
`momzero` and `mom121` challenger arms automatically and this table is wrong
until updated.

---

## 2. Every cut emitted, and what its `basis` means

`universe_membership/{date}/membership.json::cuts`. Every entry carries a
`basis` string naming *how* membership was decided. A `basis` is a claim about
the producing function, and `tests/test_scanner_contract.py::test_every_cut_basis_matches_its_producer`
holds it to that.

| cut | width | `basis` | produced by | feeds |
|---|---|---|---|---|
| `attractiveness_top_20` | 20 | `attractiveness_rank` | `_rank_table(attractiveness)` | predictor universe |
| `attractiveness_top_25` | 25 | `attractiveness_rank` | same | historical series continuity |
| `attractiveness_top_60` | 60 | `attractiveness_rank` | same | RAG corpus scope, Think Tank window |
| `scanner_gate_baseline_60` | 60 | `scanner_champion_rank` | the live champion arm, verbatim from `candidates.json::scanner_tickers` | sector teams |
| `tech_score_top_60` | 60 | `tech_score_rank` | `_rank_tech_score` over `scan_path == "momentum"` rows | nothing live — the displaced-incumbent baseline (§3) |
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
