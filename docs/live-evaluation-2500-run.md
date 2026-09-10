# Step-2500 live evaluation — 2026-09-10

Data preparation and inference validation are complete. The full evaluation jobs
started at 14:23:50 UTC and remain running as of 15:00 UTC. The results below are
completed QA conditions, not a claim that all reconstruction runs have finished.

## Test sets

| Suite | Cases |
|---|---:|
| Reconstruction, 2 images | 249 |
| Reconstruction, 4 images | 124 |
| Reconstruction, 8 images | 61 |
| Reconstruction, 80 original source lines | 395 |
| LongCodeQA, complete Qwen text prompt below 32,768 tokens | 110 |

There are 939 cases and 1,626 unique PNGs in the final manifest. The QA cohort
covers 26 repositories, 3,160–30,191 text prompt tokens, and 5–50 images per
question. All 443 LongCodeQA examples were inspected; 333 exceed the strict
token limit. The 80-source-line images have 80–303 displayed lines after wrapping.
The longest reconstruction reference is 20,443 tokens, covered by the 32,768-token
generation budget.

The original reconstruction pool contains one snippet whose visual text also
appears in training. One case containing it was removed from each reconstruction
suite. The raw preparation-stage counts in `reconstruction.json` precede this
exclusion; `manifest.json` and `audit.json` contain the final cohort.

Server artifact root: `/workspace/optical-adaptor/outputs/benchmarks/live-2500`.

| Artifact | SHA-256 / revision |
|---|---|
| Final `manifest.json` | `f365be54cbd3089f868ac6cfcdb25a2601de88e4dff425cedff848f9665d6922` |
| `LongCodeQA.zip` | `0b31cf93a744d659e059056e6cd03bc7bb8d2601bad11e28d6b97dbfddf8d41a` |
| LCB revision | `989d5eff750d65a72c522e47f8f745ef2e22906b` |
| Step-2500 adapter weights | `bc0c59b28c357ceebf58b83383b63014edfb3a285a7606fd5536e2aec3423867` |

The checkpoint is `outputs/adapter-v1/runs/mlp-10epochs/step-002500`; its state
records step 2500. Model revisions and processor settings are in the canonical
configs and each backend's result metadata.

## Validation

- Five focused tests passed on the server. Local Ruff and Git whitespace checks
  passed; no Python or GPU experiments were run locally.
- The pinned Qwen template produces adjacent, separate vision blocks for 2, 4
  and 8 images in one user message. Each adapted image contributes 111 tokens.
- The selected live DeepSeek tensor exactly matches its training-cache tensor:
  maximum, mean and relative L2 differences are all zero with encoder batch size
  eight. Padding the final live microbatch preserves this geometry.
- A 64-token greedy reconstruction is identical between the live vLLM backend
  and Transformers when supplied the same adapted embedding sequence.
- A real `/v1/chat/completions` request with eight native images succeeded:
  14,480 image tokens and 14,554 prompt tokens. The temporary test endpoint was
  stopped before launching the full GPU-9 evaluation.

Detailed checks are in `validation/live-backend.json` and
`validation/native-endpoint.json` under the artifact root.

## Completed QA results

| Condition | LCB parser correct | Accuracy | Strict letter correct |
|---|---:|---:|---:|
| Step-2500 adapter, live images | 1 / 110 | 0.9% | 0 / 110 |
| Native Qwen, full text | 78 / 110 | 70.9% | 78 / 110 |
| Native Qwen, no repository input | 72 / 110 | 65.5% | 72 / 110 |
| Native Qwen, live images | Running | — | — |

The adapter's output generally transcribes or continues code instead of answering
the multiple-choice question. The upstream parser cannot extract an answer from
108/110 responses, and 94/110 reach the 32-token QA generation limit. This result
therefore includes a substantial instruction-following and output-budget failure;
it does not isolate repository comprehension. All conditions use the same
greedy, no-thinking policy and QA budget. Both raw responses and strict scores
are retained, and the upstream comparison is a separate postprocessing artifact.

Reconstruction is still running. Some adapter generations repeat until the
32,768-token limit; their truncation flags and full raw outputs are retained.

## Running jobs and outputs

| GPU | Job ID | Conditions |
|---|---|---|
| 8 | `20260910-142350-evaluate-2500-live-adapt-c853a8` | Adapter image QA and all reconstruction suites |
| 9 | `20260910-142350-evaluate-2500-native-bas-f35de6` | Native image QA/reconstruction, text QA, no-image QA |

Per-case outputs, `summary.json`, and eventual `complete.json` are under
`results/vllm-adapter` and `results/vllm-native`. The jobs are durable remote tmux
jobs. Re-run the same evaluator command to resume only missing outputs after a
failure; do not start a duplicate while the original job is active.

Refresh the QA comparison without inference:

```bash
uv run python -m optical_adaptor.benchmark.compare
```

Refresh either saved-result summary without a GPU:

```bash
uv run evaluate-live-benchmark --backend vllm-adapter --report-only
uv run evaluate-live-benchmark --backend vllm-native --report-only
```

The evaluation jobs started from commit `7aa7bd2`. Commit `45a0947` added the
separate upstream QA parser and comparison command without changing the running
inference or its strict scoring. See [live-evaluation.md](live-evaluation.md) for
data construction, inference design and endpoint usage.
