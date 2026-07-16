"""RAG namespace package.

The shared retrieval / db / embeddings / schema code now lives in
``alpha_engine_lib.rag`` (since lib v0.3.0). This folder retains only the
ingestion ``pipelines/`` subpackage. Consumers wanting the retrieval
surface should import from the lib:

    from nousergon_lib.rag import retrieve, is_available

The lib's own ``__init__`` auto-loads ``.env`` for ``RAG_DATABASE_URL`` and
``VOYAGE_API_KEY``, so no duplication is needed here.
"""
