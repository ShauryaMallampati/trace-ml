# Integration guide

TRACE-ML fits between an experiment tracker and whatever consumes or displays aggregate metrics.

A typical flow is:

1. export completed run records;
2. normalize a metric statement into a claim record;
3. select the run IDs that the claim is supposed to summarize;
4. call `verify(claim, runs)`;
5. accept, block, or route the result for review based on the verdict.

For CI or release automation, a useful policy is:

- `supported`: continue;
- `violation`: fail the check;
- `insufficient_evidence`: require review or additional evidence.

That distinction prevents missing or ambiguous data from being mislabeled as a confirmed error.

TRACE-ML intentionally does not authenticate a tracker, resolve aliases, or discover candidate runs. Keep those responsibilities in the surrounding system and pass the verifier an explicit, auditable input set.
