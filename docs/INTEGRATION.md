# Integration guide

TRACE-ML fits between an experiment tracker and whatever consumes or displays aggregate metrics.

A typical flow is:

1. export completed run records;
2. normalize a metric statement into a claim record;
3. select the run IDs that the claim is supposed to summarize and record them in `candidate_run_ids`;
4. for comparison claims, also select independent baseline runs and record them in `baseline_run_ids`;
5. pass exactly those referenced records to `verify(claim, runs)`;
6. accept, block, or route the result for review based on the verdict.

The verifier checks that the supplied run IDs match the claim manifests exactly. Missing references or extra unreferenced records return `insufficient_evidence` rather than being silently ignored.

For CI or release automation, `trace-ml verify ... --require-supported` implements this policy directly:

- `supported`: continue;
- `violation`: fail the check;
- `insufficient_evidence`: require review or additional evidence.

That distinction prevents missing or ambiguous data from being mislabeled as a confirmed error.

TRACE-ML intentionally does not authenticate a tracker, resolve aliases, or discover candidate runs. Keep those responsibilities in the surrounding system and pass the verifier an explicit, auditable input set.
