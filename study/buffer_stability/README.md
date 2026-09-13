# Bounded replay and later learning

Completed all three seeds. Final mean accuracy on the original evaluation prompts:
full replay 100%, no replay 97.5%, random four-fact replay 89.2%. See `RESULTS.md`
for additional phrasings, trajectories, target stability, budgets and limitations.

Resume the completed repaired Qwen3.5-2B checkpoints. Two waves add four facts
each. Compare no replay, a four-fact random replay buffer, and the full ledger.
See `PROTOCOL.md` and frozen `plan.json`; run `RUN.ps1` for all three seeds.
Outputs are exclusive: existing trial directories are never overwritten.
`summarize.py` audits completed runs and writes `RESULTS.md` and `SUMMARY.json`.

The buffer limits facts rehearsed during training. The full external accepted
ledger remains available for sampling; this is not a total-memory-size claim.
