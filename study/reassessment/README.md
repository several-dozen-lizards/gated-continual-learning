# Continuing the selected Qwen3.5-2B experiment

This phase resumes the three saved gated adapters from `../qwen35/`. It tests
reconsideration after new evidence and replay of an accepted-fact ledger.
The source histories and lineage metadata are synthetic assumptions. The gate
can admit corroborated falsehoods; their downstream effects are measured.

Read `PROTOCOL.md` for the frozen design. `plan.json` hashes source files,
fixtures and prior checkpoint receipts. `RUN.ps1` runs all three seeds and
refuses existing output directories. `summarize.py` audits completed runs and
produces `RESULTS.md` and `SUMMARY.json`; `plot.py` renders `OUTCOMES.png`.

The sham condition leaves the prior checkpoint unchanged. Replay comparisons
have their own matched random controls because replay uses additional training.
No prior result, base model, resident or JNSQ service is modified.
