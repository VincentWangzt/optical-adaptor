# Delayed reconstruction learning in v1

The MLP's reconstruction improvement is consistent with a delayed optimization
transition, but its mechanism is not established. The evidence comes from both
complete 939-update W&B histories, saved evaluation metrics, and a CPU probe of
the adapters at checkpoints 250, 500, 750, and 939. No training was extended.

## What the histories show

The plateau is mainly in reconstruction. MLP front-continuation KL already falls
from 1.233 at initialization to 0.849 at update 100 and 0.781 at update 500.
Thus a nearly flat reconstruction curve does not mean the adapter is unchanged.

| Updates | MLP training reconstruction CE | Transformer training reconstruction CE | MLP mean gradient norm | Transformer mean gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 451–500 | 1.0826 | 1.0921 | 0.1665 | 0.1144 |
| 501–550 | 1.0509 | 1.0627 | 0.1613 | 0.1126 |
| 551–600 | 1.0482 | 1.0871 | 0.2176 | 0.1060 |
| 601–650 | 0.7596 | 1.0906 | 0.4355 | 0.1040 |
| 651–700 | 0.5971 | 1.0674 | 0.4374 | 0.1086 |

Training CE here divides summed token losses by summed token counts across the
window. It is a moving training-set measurement, not a fixed-set generalization
gap estimate. Both models used the same record schedule, helping distinguish
shared batch difficulty from the growing difference between models.
Raw gradient norms depend on parameterization; their cross-architecture
magnitudes are not direct measures of learning quality.

In ten-update windows, MLP's gradient norm rises from 0.177 at 571–580 to 0.243
at 581–590, 0.336 at 591–600, and 0.420 at 601–610. Reconstruction CE reaches
0.887 at 601–610 and 0.803 at 611–620. Transformer does not make the same
transition. The sustained change is closer to 580–610 than exactly 500.

Evaluation every 100 updates makes the transition look coarser: MLP CE is 1.066
at 500, 0.900 at 600, and 0.579 at 700. All 30 reconstruction language strata
improve between 500 and 700, with CE decreases ranging from 0.050 to 0.833.
Training and evaluation improve together; this is not evidence of generalization
emerging only after the training set was already memorized. That distinction
matters when comparing this behavior with
[grokking](https://arxiv.org/abs/2201.02177).

## Schedule and implementation checks

- Learning rate is constant at `2e-4` throughout the transition. Warmup occupies
  the first 29 updates.
- Epoch boundaries are 313 and 626. The change is visible before 626. There is
  no curriculum, unfreezing event, or loss-weight change near 500.
- No gradients are clipped in updates 101–939 for either model. The larger MLP
  norms during the transition remain below the clipping threshold of 1.
- Previously completed audits verified FP32 adapter parameters and optimizer
  states, full data coverage, fixed evaluation membership, and identical
  configurations and schedules between models.
- At 500, the trainer performs evaluation, its first generation check, and a
  checkpoint save. The code restores adapter training mode on every update and
  sets the frozen decoder's mode and `use_cache=False` explicitly for training.
  No parameter or optimizer replacement occurs in the generation/checkpoint
  branches. A direct full-Qwen comparison before and after generation remains
  untested; the idle GPU became occupied before that probe could start.

The timing and code inspection argue against a step-500 schedule switch. They do
not rule out every runtime issue. The subsequent temperature-1 sampling change
does not alter teacher-forced evaluation loss and cannot explain this historical
curve.

## Leading explanation and an architectural concern

Teacher forcing supplies the correct preceding output tokens. Frozen Qwen can
therefore achieve substantial accuracy before the visual prefix is useful for
transcription. A plausible learning sequence is that the adapter first becomes
compatible with Qwen's input representations and fits easier continuation cues;
later, more useful visual alignment gives larger reconstruction gains and
stronger gradients. This resembles the long periods of reliance on one modality
studied in [multimodal learning](https://arxiv.org/abs/2312.00935), but that theory
does not establish the mechanism in this frozen-Qwen system.

A separate probe ran the saved adapters in FP32 on CPU using the first 16 records
of the fixed reconstruction subset. It measured how much output energy varies
between the 111 tokens within each image:

`R = mean((z - mean_over_tokens(z))²) / mean(z²)`.

The inner mean is computed separately for each image and feature dimension;
the outer means include images, tokens, and features. A low value means a large
component is shared across token positions within an image.

| Checkpoint | MLP output RMS | Transformer output RMS | MLP within-image varying fraction | Transformer within-image varying fraction |
| --- | ---: | ---: | ---: | ---: |
| 250 | 0.457 | 1.519 | 56.8% | 1.3% |
| 500 | 0.479 | 1.694 | 67.6% | 4.7% |
| 750 | 0.458 | 1.679 | 69.8% | 10.4% |
| 939 | 0.452 | 1.671 | 71.1% | 13.2% |

Transformer outputs have a much larger component shared across positions.
At step 500, their RMS after subtracting each image's token mean is 0.367,
close to MLP's 0.394, despite the very different varying fractions. Thus the
fraction alone does not show that token-specific information disappeared;
the common component dominates the relative variation. Input-scale mismatch
and excessive spatial mixing are hypotheses to distinguish, not established
causes. The shared component can also carry information that differs between
images. This is a small CPU probe, not a reproduction of Qwen's BF16 computation.
MLP's RMS does not show a sharp growth between 500 and 750 that would by itself
explain its loss drop.

The earlier final-checkpoint image-mismatch test supports a difference in how
the visual input is used: reconstruction CE increases by 0.799 for MLP and 0.081
for Transformer on 50 matched records. It does not establish when this difference
emerged.

## Tests that would distinguish the hypotheses

1. Repeat correct-image versus mismatched-image reconstruction loss at checkpoints
   250, 500, and 750 on fixed records. A growing image benefit would directly
   test the delayed-use hypothesis.
2. Measure continuation and reconstruction gradient norms and their cosine
   similarity separately. Equal scalar loss weights do not imply equal gradient
   contributions; competition between the objectives remains unmeasured.
3. Check the effect of the generation call on a fixed training forward/backward
   pass, and compare Transformer token similarity before and after its attention
   blocks. These target runtime state and spatial mixing independently.
4. Repeat with another seed before treating the knee location as stable, or the
   three-epoch ranking as an inherent architectural advantage. A reconstruction
   warm-start or a Transformer initialized closer to the per-token MLP are
   possible controlled interventions after the diagnostic checks.

Server-side aggregates:
`outputs/adapter-v1/diagnostics/learning-dynamics-history.json` and
`outputs/adapter-v1/diagnostics/adapter-representation-dynamics.json`.
The latter records the 16-image selection fingerprint. Primary experiment
scores remain those in [training-results.md](training-results.md).
