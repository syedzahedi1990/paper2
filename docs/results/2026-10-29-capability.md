# Base-model capability on the frozen development stories

All three untrained models completed the same 120 clean development stories. Values below are accuracy with 95% whole-story bootstrap intervals (20,000 resamples). Seven unrewarded answers are averaged within each story before resampling; they are not 840 independent observations.

| Model | Immediate direct | Native reasoning direct | Seven unrewarded views |
|---|---:|---:|---:|
| Qwen2.5 3B | 23.3% [15.8%, 30.8%] | 13.3% [7.5%, 20.0%] | 35.0% [31.5%, 38.5%] |
| Qwen2.5 7B | 38.3% [30.0%, 46.7%] | 35.8% [27.5%, 44.2%] | 42.5% [39.0%, 46.0%] |
| Mistral Small 24B | 46.7% [37.5%, 55.8%] | 20.0% [13.3%, 27.5%] | 59.2% [55.7%, 62.6%] |

## Depth and event type

| Model | Depth | Event | Immediate direct | Native direct | Unrewarded views |
|---|---:|---|---:|---:|---:|
| Qwen2.5 3B | 1 | move | 26.7% [13.3%, 43.3%] | 10.0% [0.0%, 23.3%] | 42.4% [36.7%, 48.1%] |
| Qwen2.5 3B | 1 | tell | 30.0% [13.3%, 46.7%] | 13.3% [3.3%, 26.7%] | 35.7% [28.6%, 43.3%] |
| Qwen2.5 3B | 2 | move | 16.7% [3.3%, 30.0%] | 10.0% [0.0%, 23.3%] | 30.0% [23.3%, 37.1%] |
| Qwen2.5 3B | 2 | tell | 20.0% [6.7%, 33.3%] | 20.0% [6.7%, 36.7%] | 31.9% [25.7%, 38.6%] |
| Qwen2.5 7B | 1 | move | 33.3% [16.7%, 50.0%] | 60.0% [43.3%, 76.7%] | 48.6% [42.9%, 54.3%] |
| Qwen2.5 7B | 1 | tell | 46.7% [30.0%, 63.3%] | 40.0% [23.3%, 56.7%] | 47.1% [40.5%, 53.8%] |
| Qwen2.5 7B | 2 | move | 36.7% [20.0%, 53.3%] | 23.3% [10.0%, 40.0%] | 38.1% [30.0%, 46.7%] |
| Qwen2.5 7B | 2 | tell | 36.7% [20.0%, 53.3%] | 20.0% [6.7%, 36.7%] | 36.2% [31.0%, 41.4%] |
| Mistral Small 24B | 1 | move | 66.7% [50.0%, 83.3%] | 20.0% [6.7%, 33.3%] | 68.1% [62.4%, 73.3%] |
| Mistral Small 24B | 1 | tell | 80.0% [63.3%, 93.3%] | 33.3% [16.7%, 50.0%] | 71.0% [64.8%, 77.1%] |
| Mistral Small 24B | 2 | move | 13.3% [3.3%, 26.7%] | 16.7% [3.3%, 30.0%] | 50.0% [43.8%, 56.7%] |
| Mistral Small 24B | 2 | tell | 26.7% [13.3%, 43.3%] | 10.0% [0.0%, 23.3%] | 47.6% [42.9%, 52.4%] |

## Format and trace diagnostics

| Model | Invalid immediate direct | Invalid native | Forced delimiter | Truncated native | Trace tokens, mean / max |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 3B | 0/120 | 37/120 | 20/120 | 0/120 | 203.2 / 384 |
| Qwen2.5 7B | 0/120 | 2/120 | 0/120 | 0/120 | 171.0 / 302 |
| Mistral Small 24B | 0/120 | 75/120 | 14/120 | 0/120 | 160.9 / 384 |

Native accuracy uses the training/G1 reward rule. Missing EOS at the budget boundary is a truncation diagnostic, not automatically an error when that rule accepts the completion. Parsed-answer accuracy and parsed-correct/reward-invalid counts are retained in the JSON. A format failure alone does not establish inability to learn.

## Measured execution costs

| Model | Process time, min | Immediate / native evaluation, min | Generated tokens | Input tokens: forward / generation | Model-load time, s |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 3B | 7.55 | 0.67 / 6.76 | 24,819 | 498,260 / 61,855 | 7.05 |
| Qwen2.5 7B | 7.66 | 0.99 / 6.49 | 20,996 | 498,260 / 61,855 | 10.23 |
| Mistral Small 24B | 15.64 | 2.21 / 13.09 | 19,883 | 497,889 / 61,804 | 18.91 |

The recorded process time is 0.514 hours. At the plan's reported $1.326389/hour rental rate, this corresponds to approximately $0.682. This is **not an invoice**: preparation, download/idle time and separate network charges are excluded. The JSON retains raw per-session counters, repeat-work counters and peak-memory receipts.

## Interpretation and provenance

Development-only, untrained models; no gate decision. Mistral 24B changes model family as well as size. Nine-event stories and A–F answers differ from Paper 1, so these are not replications of its scores. Intervals concern stories, not training seeds or a population of models.

No pass/fail threshold or gate approval is applied. These measurements inform the capability review; baseline performance alone does not establish what training can learn.

Frozen corpus manifest: `01b4e8106d9a178b707a8702ad0724167e4ada03921dec503a8779be4195562c`. Execution tree: `071f930cb6d4570bd18c45ede42e259f1f1732fc`. Artifact hashes and complete model revisions are in the JSON sidecar.

## Subsequent decision

The [dated model and resource-profiling amendment](../amendments/2026-10-29-model-and-profiling.md) selects the exact Mistral 24B checkpoint for both stages, subject to measured resource feasibility. A posthoc diagnostic found that all 75 invalid Mistral completions end in an extra terminal period. Accepting only that period gives 65/120 = 54.2% [45.0%, 63.3%], but the primary 20.0% score and reward rule remain unchanged. Depth-two weakness remains a limitation. No confirmation outcomes or research training informed this decision.
