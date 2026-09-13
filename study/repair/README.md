# Repair continuation

Completed all three seeds. Corrected-ledger replay reached 100% on both prompt
sets; unchanged replay remained at 91.7%, and correction-only averaged 58.3%.
See `RESULTS.md` for collateral retention, matched budgets and scope limits.

This phase starts from the completed `../reassessment/` corrected-evidence replay
adapters, each of which learned one false admission. Explicit source withdrawals
and new reports revise the accepted ledger. Three arms compare correction-only,
corrected ledger replay, and unchanged ledger replay, plus a frozen baseline.

`PROTOCOL.md` documents the frozen design; `plan.json` pins inputs and sources.
`RUN.ps1` runs seeds 17,29,43 without overwriting previous output. `summarize.py`
audits completed runs and produces `RESULTS.md` and `SUMMARY.json`.

This tests behavioral correction under supplied evidence. It does not discover
falsehood autonomously, demonstrate literal erasure, or alter resident state.
