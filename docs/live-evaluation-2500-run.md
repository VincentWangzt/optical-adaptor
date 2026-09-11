# Step-2500 live evaluation — 2026-09-10

Results snapshot: **2026-09-11 10:32:43 UTC (18:32:43 China time)**.
All QA conditions are complete. Reconstruction is partially complete, and both
GPU jobs remain running approximately 20 hours after their 2026-09-10 14:23:50 UTC
start. The tables below distinguish completed suites from partial results.

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
| Native Qwen, live images | 80 / 110 | 72.7% | 80 / 110 |

The adapter's output generally transcribes or continues code instead of answering
the multiple-choice question. The upstream parser cannot extract an answer from
108/110 responses, and 94/110 reach the 32-token QA generation limit. This result
therefore includes a substantial instruction-following and output-budget failure;
it does not isolate repository comprehension. All conditions use the same
greedy, no-thinking policy and QA budget. Both raw responses and strict scores
are retained, and the upstream comparison is a separate postprocessing artifact.
Native image QA has seven unparseable, length-limited responses; the text and
no-image conditions have none.

Native images improve accuracy by 1.8 percentage points over text and 7.3 points
over no-image; text improves by 5.5 points over no-image. On the same questions,
images win over text nine times and lose seven times. Images win over no-image
21 times and lose 13 times; text wins over no-image 19 times and loses 13 times.
These small net differences on 110 questions do not establish an image advantage.
The high no-image accuracy also limits what this cohort can establish about
repository-dependent reasoning; it does not itself demonstrate contamination.

The 32K eligibility threshold applies to the full **text** prompt. Image prompts
can exceed 32K: mean native image prompt length is 87,827 tokens, versus 21,086
for text and 3,684 for adapted images. This is the same question cohort, not a
matched token-budget comparison. All encoders run live.

## Reconstruction results

CER and WER are corpus edit distance divided by the number of reference units;
lower is better. Median CER is the median per-case character error rate. Scores
compare raw outputs with references, including whitespace and any extra output.
Partial rows describe only completed cases and are not final suite estimates.

| Suite | Model | Completed / expected | Corpus CER | Median CER | Corpus WER | Length stops |
|---|---|---:|---:|---:|---:|---:|
| 2 images | Adapter | 249 / 249 | 853.7% | 64.1% | 504.4% | 53 |
| 2 images | Native Qwen | 249 / 249 | 600.9% | 4.8% | 102.5% | 16 |
| 4 images | Adapter, partial | 73 / 124 | 423.9% | 83.4% | 410.1% | 22 |
| 4 images | Native Qwen | 124 / 124 | 358.9% | 7.6% | 153.2% | 16 |
| 8 images | Adapter | 0 / 61 | — | — | — | — |
| 8 images | Native Qwen | 61 / 61 | 334.6% | 7.6% | 72.5% | 10 |
| 80 original lines | Adapter | 0 / 395 | — | — | — | — |
| 80 original lines | Native Qwen, partial | 236 / 395 | 175.4% | 4.0% | 186.0% | 13 |

There are no exact matches in the multi-image suites so far. The native
80-original-line subset has one exact match out of 236 completed cases.

The enormous corpus errors are real insertion penalties from degenerate
generations. One adapter response is 2,097,152 hyphens against a 2,264-character
reference; one native response grows to 2,068,327 characters, ending in repeated
asterisks. Both models sometimes repeat until the 32,768-token generation cap.
CER/WER can exceed 100%; median CER makes the typical case more visible without
discarding these failures from the primary corpus metrics.

For the complete two-image suite, the adapter reaches the cap in 21.3% of cases
versus 6.4% for native Qwen. Restricting separately to non-truncated outputs gives
58.5% versus 8.3% corpus CER, but this is a selected-subset diagnostic, not an
unbiased replacement score.

The four-image rows have different completion counts. On the **same 73 cases**,
adapter versus native corpus CER is 423.9% versus 546.0%, median CER is 83.4%
versus 7.6%, and length stops are 22 versus 13. Extreme insertion failures reverse
the corpus-CER ranking here; the partial aggregate does not support a blanket
claim that either model wins on every reconstruction metric.

The adapter is substantially worse on the complete two-image suite and fails QA
instruction following under this policy. Conclusions about its eight-image and
80-source-line performance must wait for those suites to run.

## Running jobs and outputs

| GPU | Job ID | Conditions |
|---|---|---|
| 8 | `20260910-142350-evaluate-2500-live-adapt-c853a8` | Adapter image QA and all reconstruction suites |
| 9 | `20260910-142350-evaluate-2500-native-bas-f35de6` | Native image QA/reconstruction, text QA, no-image QA |

Per-case outputs, `summary.json`, and eventual `complete.json` are under
`results/vllm-adapter` and `results/vllm-native`. The jobs are durable remote tmux
jobs. Re-run the same evaluator command to resume only missing outputs after a
failure; do not start a duplicate while the original job is active.

The fixed snapshot, including per-suite distributions, matched-subset results,
and paired QA counts, is saved as `results-report-2026-09-11.json` under the
server artifact root. `qa-comparison.json` was refreshed from all 110 responses
per condition. The running jobs continue to update their own summaries.

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
