## Relocated to nous-ergon-ops (private)

The following operational files were relocated to the private
`nousergon/nous-ergon-ops` repo (mirrored layout) in the Phase-2 scoped
ops migration (alpha-engine-config#636, 2026-06-11). Each was verified
consumer-free (no workflow/test/SF-literal/box-runtime path) before
removal. Operators: find them at `nous-ergon-ops/<this-repo>/<same-path>`.

- `spot_research_weekly.sh` (legacy spot launcher; Research now runs via the alpha-engine-research-runner Lambda)
- `iam-policy.json`, `trust-policy.json` (Lambda role definitions)
