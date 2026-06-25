"""Research producer champion/challenger OBSERVE substrate (config#1221 / M3).

The R slot (research → signals.json) is the second module after the scanner to
become a champion/challenger observe substrate (ARCHITECTURE §37): the live
agentic LangGraph producer is the champion; alternative producers (a no-agent
pure-quant baseline, a single-agent baseline) run in shadow on the SAME scanner
candidate set, emit a conforming signals.json to an isolated path, and are
scored on realized outcomes. Promotion is manual + evidence-gated; nothing
trades on a challenger until promoted.

This realizes the M3 "second-implementation proof" — does the multi-agent
orchestration earn its keep over a single agent / no agent on ranking alpha?
"""
