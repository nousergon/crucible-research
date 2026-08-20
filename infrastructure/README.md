## Relocated to nous-ergon-ops (private)

The following operational files were relocated to the private
`nousergon/nous-ergon-ops` repo (mirrored layout) in the Phase-2 scoped
ops migration (alpha-engine-config#636, 2026-06-11). Each was verified
consumer-free (no workflow/test/SF-literal/box-runtime path) before
removal. Operators: find them at `nous-ergon-ops/<this-repo>/<same-path>`.

- `iam-policy.json`, `trust-policy.json` (Lambda role definitions)

## Retired: this repo owns NO spot launcher

`spot_research_weekly.sh` and its box entrypoint `weekly_box_runner.py` were
deleted 2026-08-20 (`alpha-engine-config-I7856`). They drove
`handler(weekly_run=True)` — the champion graph pass deleted in
`crucible-research-PR685` / `alpha-engine-config-I7827`, which now raises
`RetiredResearchPathError`. Verified consumer-free: no invoker in
`crucible-research`, `nousergon-data`, `nous-ergon-ops` or
`alpha-engine-config`, and EventBridge `alpha-research-weekly` is DISABLED.

`thinktank_spot_bootstrap.sh` is NOT a replacement launcher. It is the ON-BOX
entrypoint the `alpha-engine-thinktank-spot-dispatcher` Lambda runs via SSM
after it has already launched the instance — it never calls `krepis.ec2_spot`,
never resolves `LIB_PYTHON`, and has no relaunch or staging-teardown decision.
So **this repo currently owns no spot launcher at all**, and that is a
statement, not an oversight.

What that did to the five launcher-invariant tests, each named with where its
subject now lives:

| invariant | disposition |
|---|---|
| `test_launchers_resolve_the_declared_krepis_guard` | **KEPT, re-pointed at the launcher SURFACE.** `KNOWN_LAUNCHERS` is now empty and the membership assertion is bidirectional: a `spot_*.sh` added here fails until declared. Live subjects for the guard itself: `nousergon-data`, `crucible-predictor`, `crucible-dashboard` copies. |
| `test_spot_bootstrap_invariants` | **PARTIALLY KEPT.** The three repo-WIDE negative sweeps (no inline bootstrap heredoc, no silent `python3.12`→`python3` fallback, no heredoc `git clone` into `/home/ec2-user/`) keep a live subject — they scan every `.sh` under `infrastructure/`, `scripts/` and `bin/`, `thinktank_spot_bootstrap.sh` included. The `krepis.spot_bootstrap render` dispatch-shape assertions were subject-specific and are deleted; live copies in `nousergon-data`, `crucible-predictor`. |
| `test_spot_research_weekly_krepis_cli_executes` | **DELETED — no live subject here.** Fleet copies: `nousergon-data/tests/test_spot_launchers_krepis_cli_executes.py`, `crucible-predictor/tests/test_spot_train_krepis_cli_executes.py`, `crucible-backtester/tests/test_spot_backtest_krepis_cli_executes.py`. |
| `test_failure_staging_is_retained` | **DELETED — no live subject here.** The invariant is a property of `_teardown_staging`, which this launcher carried inline. Fleet copies over live launchers: `nousergon-data/tests/test_failure_staging_is_retained.py`, `crucible-predictor/tests/test_failure_staging_is_retained.py`. |
| `test_spot_relaunch_decision_survives_errexit` | **DELETED — no live subject here.** Fleet copies: `nousergon-data/tests/test_spot_relaunch_decision_json_contract.py`, `crucible-predictor/tests/test_spot_train_reclaim_relaunch.py`, `crucible-backtester/tests/test_spot_backtest_reclaim_relaunch.py`. |

No invariant was dropped fleet-wide. Three lost their subject **in this repo
only**, because the subject was deleted — and the first row is what makes a
future launcher here re-acquire coverage instead of arriving unguarded.
