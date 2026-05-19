"""Producer-substrate reader — read-only consumer side of the
institutional data-revamp arc (Wave 1).

The producer side lives in alpha-engine-data. Canonical-shape S3
layout (post-#234; legacy ``{date}``-keyed fallback retired 2026-05-19):

  - news_aggregates/{run_id}_result.parquet  +  news_aggregates/latest.json
  - insider_transactions/{run_id}_result.parquet  +  insider_transactions/latest.json
  - analyst_revisions/{run_id}_result.parquet  +  analyst_revisions/latest.json
  - analyst_snapshots/{ticker}/{run_id}.json  +  analyst_snapshots/{ticker}/latest.json

This package reads those parquets and exposes typed per-ticker
accessors that ``fetch_data`` joins onto ``input_data_snapshot`` for
downstream agents.

Architectural pattern: research is the CONSUMER. We never write to
the substrate; the producer (alpha-engine-data Saturday SF + daily
news/analyst cron) owns the write side.

See ``~/Development/alpha-engine-docs/private/data-revamp-260513.md``
for the full arc context.
"""

from data.substrate.reader import (
    SubstrateReader,
    SubstrateSnapshot,
    read_substrate_for_population,
)

__all__ = [
    "SubstrateReader",
    "SubstrateSnapshot",
    "read_substrate_for_population",
]
