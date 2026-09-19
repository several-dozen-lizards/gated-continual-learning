# Invalidated smoke receipt

This bounded plumbing run exposed a state-transition bug in the new v3 runner:
canary evaluation switched the model to evaluation mode, and the next epoch did
not explicitly restore training mode. Gradients still flowed and LoRA dropout
was configured to zero, but the run does not satisfy the frozen v3 protocol.

The receipt and adapter are retained as diagnostic evidence. `aggregate.py`
excludes any receipt with a sibling `INVALIDATED.md` file. The runner now calls
`model.train()` at every epoch boundary.

