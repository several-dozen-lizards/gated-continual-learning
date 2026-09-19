# Scale comparison v3

This is the confirmation study for the corrected gating hypothesis. It tests
performance, variability, and capability preservation as separate outcomes.

Read [PROTOCOL.md](PROTOCOL.md) before running. The full campaign is 90 trained
cells: three models by three stream arms by ten paired seeds, plus three frozen
references. A bare invocation intentionally runs only a cheap plumbing cell:

```powershell
.\RUN.ps1
```

That command performs real GPU training. CPU-only measurement contracts can be
checked with:

```powershell
python -m unittest test_measurement.py -v
```

After inspecting the smoke receipt, launch the frozen full protocol explicitly:

```powershell
.\RUN.ps1 -FullCampaign
```

Do not combine v1, v2, and v3 summaries. Their measurement contracts differ.
