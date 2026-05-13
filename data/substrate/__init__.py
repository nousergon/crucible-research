"""Producer-substrate reader — read-only consumer side of the
institutional data-revamp arc (Wave 1).

The producer side lives in alpha-engine-data:

  - news_aggregates/{date}.parquet           (PR A.2 #228)
  - insider_transactions/{date}.parquet      (PR B #230)
  - analyst_revisions/{date}.parquet         (PR C #231)
  - analyst_snapshots/{ticker}/{date}.json   (PR C #231)

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
