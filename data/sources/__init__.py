"""Multi-source data substrate — Protocol-based adapters for news,
filings, analyst data, and alt data.

Vendor-agnostic by construction: agents and downstream consumers see a
normalized Pydantic shape; the source adapter handles vendor-specific
transport, schema mapping, and rate-limit compliance.

Free-tier adapters live alongside paid stubs that drop in trivially
when subscriptions are upgraded (Phase 4 per
``~/Development/alpha-engine-docs/private/data-revamp-260513.md``).

See ``data/sources/protocols.py`` for the canonical shapes + Protocols.
"""
