# Candidate promotion gate

Completed three retrospective trials. The gate rejected both damaging targeted
candidates; fallback retained all previously correct evaluation answers and
averaged 98.3% on original prompts (99.6% on additional phrasings). These match
always-random replay here. See `RESULTS.md` for the rejection/utility tradeoff.

Evaluate operational probes before admitting saved candidate weights. Strictly
preserve every previously correct probe answer and require a recovered answer.
Fallback tries the targeted candidate, then random, then retains the baseline.

`PROTOCOL.md` and `plan.json` freeze the experiment. `RUN.ps1` runs three seeds;
`summarize.py` audits outputs and writes results. Each completed run contains
`active_adapter.json`: an immutable checkpoint path and hashes, committed only
after the selected adapter was freshly loaded and verified. A consumer can read
`checkpoint.path` from this manifest to load that experiment's selected adapter.
All prior weights remain available. This does not modify any JNSQ resident.
