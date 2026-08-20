## Relocated to nous-ergon-ops (private)

The following operational files were relocated to the private
`nousergon/nous-ergon-ops` repo (mirrored layout) in the Phase-2 scoped
ops migration (alpha-engine-config#636, 2026-06-11). Each was verified
consumer-free (no workflow/test/SF-literal/box-runtime path) before
removal. Operators: find them at `nous-ergon-ops/<this-repo>/<same-path>`.

- `spot_research_weekly.sh` (DEAD launcher — it drove `weekly_box_runner.py` →
  `handler(weekly_run=True)`, the champion graph pass deleted in
  alpha-engine-config-I7827. It has no invoker in any fleet repo and that entry
  point now raises `RetiredResearchPathError`. Retained only because five
  cross-cutting launcher-invariant tests take it as their subject; its own
  retirement is tracked separately.)
- `iam-policy.json`, `trust-policy.json` (Lambda role definitions)
