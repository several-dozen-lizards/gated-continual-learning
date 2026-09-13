# Measured-forgetting replay

Completed three trials. Targeted replay recovered every prior incorrect answer,
but created new errors: 96.7% mean overall versus 98.3% for matched random replay
on the original evaluation prompts. See `RESULTS.md` for both prompt sets and costs.

Recovery comparison using the before/after small-buffer checkpoints from the
previous phase. Rank lost confidence with separate operational probes, then
compare four selected facts against token-matched random replay.

See frozen `PROTOCOL.md` and `plan.json`. `RUN.ps1` runs all three seeds without
overwriting receipts. `summarize.py` audits completed runs and produces results.
Probe and training costs are separate; accepted-ledger values guide the probes.
