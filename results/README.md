# Results

Generated samples, execution traces, and model checkpoints are intentionally not committed.

Each evaluation writes:

- `samples.jsonl`: raw model completions;
- `scored_samples.jsonl`: reward components and execution status;
- `summary.json`: pass@k and aggregate status counts.

Oracle smoke-test output is a harness check, not a model result.
