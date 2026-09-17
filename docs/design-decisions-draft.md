# Optical-adaptor discussion and decision log (temporary)

Updated: 2026-09-17, Asia/Shanghai.
Baseline inspected: optical-adaptor `f6ba90f`; AutoModel `2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a`.

This preserves the design discussion at the inspected baseline. On September 17,
the user requested implementation of the confirmed decisions. The migration is
implemented and synchronized through GitHub. Server validation passed 94 focused
tests and completed nine GPU optimizer updates, including shared/separate
placement and checkpoint continuation. The subsequent review/fix now integrates
the existing native FSDP2/TP/CP strategy and resolves the observed repeated-update
drift from DeepSeek's scripted activation warm-up. See
[parallel validation](automodel-parallel-validation.md) for current evidence and
limits, and [migration validation](automodel-migration-validation.md) for the
initial results. Historical observations below describe the baseline unless
explicitly marked otherwise.

## Decision protocol

The user requested that each change receive at least one confirmation before it is recorded as to be carried out.

- **Confirmed — to implement:** the user explicitly accepted a proposal already discussed. The table records the confirming statement; implementation status is tracked separately.
- **Confirmed — retain:** an existing behavior the user explicitly accepted.
- **Pending:** a newly proposed design, open question, or unresolved implementation scope. Neither the assistant's recommendation nor silence is confirmation.
- **Deferred:** explicitly excluded from the initial implementation. Revisit only through a later discussion; do not treat it as unfinished required work for the initial migration.
- **Superseded proposal:** an earlier suggestion that the latest discussion replaces. It is not on the implementation list.

Confirmation of a direction does not confirm every new detail attached to it. For example, accepting an editable checkout does not select a fork location or a packing strategy. Promote individual pending IDs only after an explicit user reply; record its substance and date. An explicit correction also confirms the revised scope. Do not ask again for choices already confirmed, or treat confirming this document's creation as confirming all pending changes.

Discussion references:

- **R0:** original task, “Build optical-adaptor training” (`01a0a678-50c3-71b0-9efc-8f98a0201a01`).
- **R1:** user's eight questions about framework reuse, annotations, bins, mixture, batching, evaluation, legacy code, and speed.
- **R2:** assistant's code/log audit and proposals in response to R1.
- **R3:** user's eight-point response and request for this decision log, recorded September 16.
- **R4:** user's seven-point follow-up on September 17: confirms ordinary AutoModel CLI/YAML and recipe reuse, primary bins and logical sources, equal leaves/accounting, runtime evaluation defaults; proposes inherited relative training weights; clarifies adaptive batching; replaces repository-held-out evaluation with a per-leaf sample fraction; asks for batch-statistic definitions.
- **R5:** user's four-point follow-up on September 17: requires logits at the teacher/student boundary, proposes selecting hidden positions before projection, asks about chunking and independent mesh/microbatch/cap settings, accepts a unified image interface if it fits AutoModel, and selects batch-size logging without per-example means. Requests a list of remaining decisions.
- **R6:** user's eight-point follow-up on September 17: confirms removing position-axis projection/loss chunk loops, asks which axis official chunking uses and how official packing/DP routing works, settles image batching, confirms inherited weights and warning/use-available behavior with absolute counts plus percentages, retains teacher CE/agreement/target-token accuracy, and approves the proposed historical-runtime disposition. Batching and memory policy remain under discussion.
- **R7:** user reaffirms “disable chunking for now” and asks whether official packing primarily controls token counts rather than balancing compute. This confirms the current unchunked loss direction, not a new adoption decision for packing.
- **R8:** user rejects stock dataset-level packing as the desired policy and proposes selecting a fixed global batch first, grouping its examples under configured limits, making the number of groups a multiple of the destination mesh's DP size, and dispatching them. Asks whether this is easy/moderate/difficult and appropriate for research code. The proposal is under discussion, not yet confirmed for implementation.
- **R9:** user asks what global-batch grouping adds, whether official packing concatenates sequences, how a multi-example group reaches a DP rank, and how fixed local batch size behaves. This is clarification of P08; it does not select padded grouping over concatenation.
- **R10:** user questions whether concatenated packing becomes too complicated when selective loss positions are included. This expresses a complexity concern; it does not yet confirm abandoning packing or choosing a replacement batching policy.
- **R11:** user confirms “right, defer packing then,” accepting the preceding recommendation: ordinary padded microbatches with manually configured local batch size and existing AutoModel accumulation for the initial migration. Concatenated packing and adaptive global-batch grouping are deferred.

## Confirmed register

| ID | Status | Confirmed scope | Confirmation evidence | Still unresolved |
| --- | --- | --- | --- | --- |
| C01 | Confirmed — to implement | Install our AutoModel dependency from a pinned editable checkout. | R3.1: “install AutoModel via a local pinned editable checkout is good,” accepting R2.1; reaffirmed by R4.1. | Exact fork location and Git integration remain to be selected. CLI direction is now confirmed under P01. |
| C02 | Confirmed — to implement | Use readable inline annotations as the stored visual-area representation instead of the explicit sidecar. | R3.2: “yeah I think an inline tag is better,” accepting R2.2. | Example spelling is `<visual_AREA_start>`; a corresponding end marker and compiler details have not been finalized. User does not want an elaborate collision mechanism. |
| C03 | Confirmed — to implement | Remove the image-bin × turn-bin cross-product and its diagnostic reporting. Use task-appropriate primary grouping; retain current bin boundaries. | R3.3: “full cross product is unnecessary, and no need for diagnostics as well”; R4.2: “the current binning looks good.” | Logical-source hierarchy is confirmed under P03. Raw image/turn counts remain metadata without becoming cross-product metrics. |
| C04 | Confirmed — retain | Keep image-encoder microbatch control separate from conversation batching; rely on its own batching logic for image costs. | R3.5 accepts separate image batch control; R6.3: “image costs can be settled then, we can rely on its own batching logic.” | P16 confirms a unified interface conditional on AutoModel compatibility. Numeric image batch size is a configuration choice. Do not reopen a separate image-cost scheduler as an undecided architecture item. |
| C05 | Confirmed — to implement | Add top-level source/task aggregates to eval-core; keep evaluation quality reporting focused and omit token-count curves from evaluation metrics. | R3.6 accepts the R2.6 top-level metric proposal and says reporting tokens as evaluation metrics “does not look good.” | Exact metric list, names, and counts are P12/P14. Internal token denominators remain necessary for correct aggregation. |
| C06 | Confirmed — to implement | Retire the old training implementation after extracting or replacing dependencies still used by retained tools. | R3.7 agrees with R2.7's shared-module extraction and removal direction. | “Only config.py is needed” is an assumption contradicted by current imports; dependency disposition is P13. |
| C07 | Confirmed — to implement | Measure teacher-input and student-input throughput alongside supervised-target throughput. | R3.8 accepts the throughput measurement proposed in R2.8. | Input versus prompt terminology, denominators, timing windows, additional batch statistics, and live ratios are P14/P15. |
| C08 | Confirmed — to implement | Report sample filtering and empty leaves with warnings, absolute counts, and percentages. | R4.3 accepts filtering warnings; R6.5 explicitly requests absolute numbers as well as percentages. | Include before/after counts, rejection reasons, and per-leaf effects. Malformed data/configuration errors remain errors. P06 now confirms empty-leaf and evaluation-shortfall handling. |
| P01 | Confirmed — to implement | Use ordinary AutoModel CLI/YAML and reuse the official recipe and infrastructure. Modify existing framework functions in our research fork; keep optical-specific components/configs importable from this addon. | R4.1: “use the normal cli and yaml” and “reuse the official recipe”; explicitly asks to modify/reuse existing AutoModel infrastructure. | Exact checkout arrangement and individual extension boundaries are implementation details to make concrete. No upstream contribution or unrelated upstream test suite is required. |
| P02 | Confirmed — architectural requirement | Preserve the official recipe's ability to use different teacher/student meshes. Use separate logical teacher/student instances where independent placement is required. | R4.1 confirms recipe reuse “so different meshes are supported,” following the separate-instance explanation. | Initial mesh topology is unspecified. An optional shared physical instance on the same placement is not required or confirmed. Model-specific parallel support still needs validation; the pinned VLM KD recipe does not support pipeline parallelism. |
| P03 | Confirmed — to implement | Use `<logical_source>/<task>/<size_bin>`; remove the view level. Stack front/middle and SWE window/full become distinct logical sources; simple SWE observation needs no view suffix. Use image bins for simple tasks and turn bins for next action. | R4.2 and R4.5 explicitly confirm removing view and using logical sources; R3.6 specifies the source distinctions. | Concrete spelling examples appear below. Preserve provenance separately. Boundaries remain `1`, `2`, `3-4`, `5-8`, `9-16`, `17+`. |
| P04 | Confirmed — to implement | Each leaf inherits the nearest configured ancestor's relative weight unless overridden. Copy the weight to every leaf, then normalize globally over eligible nonempty leaves. Overrides replace defaults; weights are not multiplied along the path or divided as parent probability shares. | R6.4: “weight inheritance confirmed,” accepting the interpretation explained after R4/R5. | `p_leaf = weight_leaf / sum(weight_leaf)`; uniform row draws within a leaf have `p_row = p_leaf / n_leaf`. More leaves imply more aggregate parent mass. Exact YAML spelling is an implementation detail. |
| P05 | Confirmed — to implement | Equal leaf defaults and per-leaf intended ratios, consumed samples, and equivalent epochs. Log `data/ratio/<slice>`, `data/training_samples/<slice>`, and `data/training_epochs/<slice>`. | R4.3: “equal leaves look good, and the accounting is good,” accepting the earlier accounting proposal. | Relative-weight resolution is P04. Count globally consumed examples, not prefetch; checkpoint counters. Epoch denominator is eligible distinct rows, so this is equivalent exposure rather than unique coverage. |
| P06 | Confirmed — to implement | Warn on emptied training leaves with absolute counts and percentages, omit them and normalize surviving leaf weights; evaluate available rows when quotas cannot be met and report the shortfall. | R6.5 accepts warning/use-available behavior and adds absolute counts. | Still fail for invalid configured names, no eligible training data, or no positive total weight. Define percentage denominators and include per-leaf before/after counts. |
| P09 | Confirmed — to implement; revised | At curation, extract a deterministic fraction of samples from each leaf into a fixed held-out pool. At training setup, select the configured subset from that pool and keep it fixed within the run. | R4.6 explicitly corrects the earlier repository proposal: “not reserve evaluation repositories, just extract a fraction of samples from each leaf.” | Fraction, seed, and small-leaf rounding defaults remain unspecified. This is sample-level validation, not repository-held-out evaluation. |
| P10 | Confirmed — to implement; revised | Configure evaluation diversity and scale at training time. Inherit absolute per-leaf sample counts from the nearest configured ancestor, with leaf overrides. Curation uses the fraction in P09, not fixed pool counts. | R4.6: training should configure evaluation diversity/scale; “inherited counts as hierarchical defaults works for me.” | Exact YAML/default counts remain unspecified. Parent count defaults apply to each descendant leaf; they are not a shared quota divided among children. |
| P12 | Confirmed — retain and aggregate | Retain teacher CE, teacher/student argmax agreement, and student target-token accuracy alongside student CE/KL; retain reconstruction CER/LER. Apply the accepted overall/source/task aggregation and omit evaluation token-count curves. | R6.7 asks to keep teacher CE/agreement/target-token accuracy, selecting retention from the alternatives discussed. | These are teacher-forced target-position metrics; agreement is not agreement between free-running generated trajectories. Generation budget policy is P11. Exact key spelling and bookkeeping loss fields are implementation details. |
| P13 | Confirmed — to implement | Migrate retained inference/benchmark entry points and shared helpers, then retire obsolete training/cache paths and old checkpoint compatibility unless historical replay is separately requested. | R6.8: “historical runtime approved,” accepting the immediately preceding recommendation. | This approves the proposed retirement/migration disposition; it does not request preservation of all historical runtimes. Moving only config.py is insufficient because of current imports. |
| P07 | Confirmed — to implement | The teacher returns logits to the student-side KD loss. Select supervised hidden-state positions on the teacher before LM-head projection, then return full-vocabulary logits for those positions. Hidden states stay an internal teacher detail. | R5.1: “One thing that I insist is to pass logits between teacher and student,” followed by hidden-position selection before calculating and returning logits; confirms the previously discussed selected-logit interface. | Preserve example/target alignment and reuse the official criterion/bridge. Exact projection/loss chunking is P17. Padding versus compact target-axis representation is an implementation detail; do not replace logits with hidden states or top-k probabilities at the boundary. |
| P08 | Confirmed — initial scope; packing deferred | Use ordinary padded microbatches, a manually configured local example batch size, a fixed global example batch, and the existing AutoModel accumulation/mesh bridge flow. Defer concatenated packing, adaptive global-batch grouping, and independent adaptive teacher/student regrouping. | R11: “right, defer packing then,” accepting the R10 simplification recommendation. | Preserve different teacher/student mesh support, paired inputs, selected logits, image batching, and disabled position-axis loss chunking. Numeric batch sizes/topology and memory validation are implementation/configuration work, not a requirement to design a new scheduler. |
| P14 | Confirmed — logging scope | Log batch size and the previously accepted batch workload totals. Omit per-example means for tokens, targets, and images. | R5.4: “simply logging the batch size is enough” and “the average per example is [not] needed for each quantity.” | Batch size counts original conversations in a global optimizer update. Suggested clear name: `batch/size`. Logging-window and live-ratio cadence can be routine implementation defaults; do not add duplicate normalized curves. |
| P15 | Confirmed — measurement scope | Measure step and meaningful substeps, including teacher forward, student forward, and student backward. | R3.8 requests these measurements; R4.7 says the proposed measurements look mostly good. | Timer implementation and overhead need focused validation. P14 resolves example-count/per-example fields. |
| P16 | Confirmed — to implement, subject to integration validation | Use a unified processor/model interface that fits AutoModel's normal VLM pipeline, retaining internal image-encoder chunking. CPU preprocessing belongs in the processor/data path; GPU vision execution belongs in the optical model. | R5.3 accepts the proposed unified interface if it fits AutoModel's pipeline. | Validate model forward/generation integration and sharding hooks. This is not authorization to move GPU model work into data-loader workers or build a separate pipeline framework. |
| P17 | Confirmed — to implement; revised | Remove our explicit position-axis projection/loss chunk loop and its per-chunk checkpointing. Disable the official position-axis KD chunking (`chunk_size: 0`). Preserve the selected-logit interface. Existing vocabulary sharding through TP may remain where supported. | R6.1 accepts removing chunking and permits retention only if official chunking is on vocabulary. Inspection establishes that official `KDLoss.chunk_size` chunks positions, so the requested condition means disabling it. | Do not invent a vocabulary-chunk loop: official TP vocabulary sharding is a distinct distributed path, not this chunk option. Image batching and normal framework accumulation are outside this loss-loop decision. Capacity, topology, and handling the resulting allocations remain P08. |

R3 establishes and R4 confirms the framework-reuse direction. Focused correctness, shape, loss, distributed, and resume checks remain relevant to the research experiment. Reuse the existing recipe/mesh bridge rather than implementing another training or communication framework.

## Generation-budget confirmation

| ID | Confirmed choice | Evidence |
| --- | --- | --- |
| P11 | Configure generation subset sizes and output caps separately from teacher-forced evaluation, using the same nearest-ancestor inheritance. | User reply in the implementation task, September 17: “Use inherited generation limits.” Numeric limits remain ordinary YAML configuration. |

Superseded proposals (not implementation requirements):

- R2's suggestion to switch off all slice balancing: equal-leaf defaults and accounting are now confirmed.
- R3-era P04 parent-relative probability partitioning and residual-mass allocation: replaced by inherited relative leaf weights, now confirmed in R6.
- R3-era P09 repository-held-out evaluation and fixed per-leaf curation pool counts: explicitly replaced by R4.6's sample fraction per leaf. Do not reserve whole repositories or silently restore repository grouping.
- R3-era P10 `curation_eval_pool_samples`: curation now uses a fraction; runtime selection still uses inherited per-leaf counts.
- R4-era flexibility about the teacher boundary: R5 requires logits and accepts hidden-position selection before projection. A hidden-state transfer interface is not an implementation option.
- R4-era optional per-example mean curves: R5 declines them. Retain batch size and workload totals instead.
- R5's recommendation to omit teacher CE/agreement/target-token accuracy: R6 retains them.
- R5's proposal to retain position-chunk/checkpoint controls pending benchmarks: R6 explicitly removes position-axis chunking. Memory planning must respect that decision rather than silently reintroducing it.
- Stock dataset-level packing as the proposed solution to automatic microbatch sizing: R8 rejects this policy and proposes packing within a preselected global batch. Existing framework utilities may still be reused.
- R8's global-batch-first packing/grouping proposal: R11 defers it for the initial migration. Retain the analysis below for a possible later optimization; it is not an initial implementation requirement.

## Verified current behavior and implications

### AutoModel integration and meshes

The current dependency is already pinned, but not installed as an editable checkout. The project subclasses `KnowledgeDistillationRecipeForVLM` and replaces setup, paired forward/backward, and validation. It inherits the optimizer-step/accumulation loop and surrounding training/checkpoint machinery.

The official recipe entry point is `nemo_automodel.recipes.vlm.kd`; its CLI accepts `--config`/`-c` and dotted overrides. A fork can retain that interface and instantiate optical-specific components from this addon. This does not mean the stock unmodified recipe can already consume optical YAML or separate teacher/student sequences.

The official recipe expects a separate teacher and student and normally forwards the same batch structure to each. Our text and optical branches have different sequence lengths; routing two branches and aligning supervision remain necessary. For separate meshes, the bridge must send the teacher branch and return target-aligned outputs. Just enabling separate_meshes on today's implementation will not work.

One reused PyTorch model instance has one parameter placement/sharding layout. Distinct teacher/student meshes cannot ordinarily share that same physical instance. Distinct logical copies may load the same checkpoint but have separate residency. Retaining adapter-only training means both Qwen parameter sets remain frozen; the student still needs activation gradients. Separate unsharded BF16 Qwen adds about 7.8 GiB of weight residency per duplicated rank, before activation/workspace costs. This is an estimate, not a memory benchmark.

R5 inspection of the pinned `KDMeshBridge` confirms independent student `distributed` and `teacher_distributed` mesh settings, including DP size. It routes one whole student-replica microbatch to a teacher replica per wave; the wave count is `ceil(student_replicas / teacher_replicas)`. For four student replicas and two teacher replicas, two waves serve the four requests. It does not split/merge sample axes or expose an independent teacher microbatch scheduler. Separate meshes currently require mesh-backed strategies; the plain DDP strategy is rejected.

Different per-forward example counts are feasible without changing the paired objective: a student request for A/B/C/D can be evaluated by the teacher as A/B then C/D, with selected logits returned in A/B/C/D order. This would be an extension to teacher execution under the existing bridge. Combining requests into a teacher batch larger than a student request requires additional scheduling. Any variable splitting must keep ranks sharing model/DP collectives on a compatible call schedule; independent Python loops per rank can hang. These details are P08, not existing capability claims.

R6 clarification of the official implementation:

- A bridge “replica” fixes one DP coordinate (including `dp_replicate`/`dp_shard` axes where present) and groups the corresponding TP/CP ranks. It denotes a logical data stream for routing, not necessarily a complete persistent weight copy on one GPU. Under FSDP, DP ranks can hold weight shards while consuming different data. With TP=CP=1 a bridge replica has one rank; with TP=2, CP=1 it has two ranks cooperating on the same microbatch. The VLM KD recipe does not support PP.
- Student-side `DistributedSampler` assigns dataset items by DP rank. `step_scheduler.local_batch_size` items form each local forward microbatch. TP ranks cooperate on model computation for the same items; CP partitions their sequence positions. Increasing TP or CP does not multiply the number of distinct examples in that microbatch.
- Teacher ranks have no independent training sampler. They receive the student batch request through the bridge, execute teacher forward, and return logits to the requesting student replica. The stock recipe sends the same branch input; the optical extension must route the corresponding teacher text instead.
- Official accumulation uses `global_batch_size = student_DP * local_batch_size * grad_acc_steps`. Teacher DP does not multiply this training sample count. Different teacher DP changes the number of request-serving waves, not the examples per teacher forward.

For student DP=4, teacher DP=2, TP=CP=1, local batch size=2, use student GPUs 0–3 and teacher GPUs 4–5 as an illustrative six-GPU mapping:

| Student replica / GPU | Its microbatch | Teacher replica / GPU | Teacher wave |
| --- | --- | --- | --- |
| S0 / 0 | A, B | T0 / 4 | 1 |
| S1 / 1 | C, D | T1 / 5 | 1 |
| S2 / 2 | E, F | T0 / 4 | 2 |
| S3 / 3 | G, H | T1 / 5 | 2 |

Each teacher forward still processes two items; each teacher serves two requests. Each student receives the corresponding logits and runs forward/loss/backward on its two items. Three accumulation rounds would consume 24 items per global optimizer update. This illustrates the mesh-backed strategy, not a request to launch six GPUs. In incomplete waves the pinned bridge feeds extra teacher replicas a duplicate request from S0 and discards those outputs; it does not subdivide another request to use that capacity.

Official sequence packing is a separate dataset transform: estimate lengths, assign examples to bounded `pack_size` groups, then materialize concatenated tensors with document boundary/attention metadata. The dataloader samples those packs with a fixed local batch size. With packing enabled, scheduler batch counts are counts of packs, while original-example counts can vary. The stock KD teacher receives the same pack as the student, not an independently packed teacher batch. Our paired input lengths and target alignment would require adaptation before this can be used. This pack-versus-example accounting is part of P08 if packing is selected.

R7 clarification: packing primarily fills a configured token capacity efficiently and reduces padding/underfilled sequences. Similar token loads can also reduce compute imbalance, so describing it as having no compute-balancing role would be too strong. The pinned implementation has optional `balance_media_tokens`, which orders packs by estimated visual-token workload; its effectiveness depends on subsequent sampling preserving useful order. It is a workload heuristic, not a guarantee of equal GPU execution times. Packing does not allocate GPUs between teacher/student meshes, tune their independent microbatch capacities, or match their execution times. For the optical branches, a teacher-full pack need not be equally full on the student side because their representations have different lengths.

### R8 assessment: pack within a fixed global batch (deferred by R11)

The following is retained design analysis for future reference. R11 selects ordinary padded microbatches and existing accumulation for the initial implementation; the planner described here is deferred.

Proposed step semantics: draw G sample occurrences from the confirmed mixture -> compute a deterministic grouping/dispatch plan from length metadata -> each mesh executes its assigned microbatches in coordinated rounds -> accumulate the entire global batch's student gradients -> one optimizer update. Reordering is confined to that global batch; it does not change its mixture draw or sample membership. Repeated weighted draws are distinct occurrences and require separate step/occurrence IDs for alignment and accounting.

Difficulty assessment:

| Part | Assessment |
| --- | --- |
| Length-based grouping within G and assigning groups to equal-length DP round schedules | Straightforward deterministic planning; a greedy method is adequate for research. It need not solve optimal bin packing. |
| Variable microbatch count per optimizer update in the existing AutoModel scheduler | Moderate integration. Current StepScheduler derives a fixed accumulation count from global/local example counts, while the KD optimizer step already accepts a microbatch list, uses its actual length, and computes a global target-token denominator. Extend the former and reuse the latter; do not build another optimizer/checkpoint loop. |
| Independently grouping teacher and student, possibly on different DP sizes | Moderate additional integration. The current bridge routes one teacher response per whole student request. Different group memberships require sample-occurrence routing, target-order reconstruction, and bounded temporary logits. |
| Concatenating sequences into one packed sequence | Additional model-specific work beyond grouping. Requires attention/recurrent-state isolation, position/media metadata, and validation of the actual Qwen backend. Simple padded microbatch grouping does not need that representation change. |

Correctness/feasibility details to preserve in the design:

- Pack-count divisibility supports equal numbers of forwards/backwards per DP rank. Use whole-example group splitting/rebalancing to obtain the needed count without dropping or duplicating sampled occurrences. A sufficient simple configuration condition is G divisible by each mesh's DP size (or their least common multiple), assuming every individual example fits its branch's limit. Otherwise rounding the number of packs upward may exceed the number of examples, making nonempty packs impossible without another tail policy. This condition is a proposal, not a confirmed new restriction.
- Equal pack counts are not sufficient by themselves: every rank sharing model/DP collectives must execute compatible round ordering and finish accumulation at the same boundary. Assignment can also group similarly costly packs into the same round to reduce waiting.
- If a “pack” is a conventional padded microbatch, its input work is approximately group size times maximum sequence length, not just the sum of raw lengths. If it is a true concatenated pack, summed lengths describe its input size more directly, but isolation requirements apply. Loss memory also depends on selected target positions times vocabulary; position-axis loss chunking remains disabled.
- Normalize CE/KD sums using the global supervised-target count, not an average of per-pack means. Thus changing group sizes does not intentionally change the objective. Resume/accounting tracks the global sampled batch and deterministic plan, not prefetched work.
- Do not implement the independent teacher plan by retaining all of the global batch's logits on one GPU. Those tensors can be much larger than model weights. Process a bounded set of teacher results and route/consume them according to the student dependencies. This is ordinary transient step data, not a persistent teacher cache.
- “No internal loops” in R6 concerns position-axis projection/loss chunking. Normal framework microbatch accumulation and the accepted image batching still require repeated forward calls; they are not the rejected loss optimization.

Historical R8 recommendation, now deferred: this would be reasonable research infrastructure if implemented as a synchronous per-step plan that reuses AutoModel's recipe, optimizer, loss normalization, mesh operations, and checkpointing. It would preserve a stable research global batch while adapting execution to heterogeneous lengths. A fully asynchronous scheduler, automatic memory probing, globally optimal packing, or custom packed-attention kernels would be a larger performance-engineering project. The gain over simpler grouping cannot be quantified without profiling.

R9 terminology clarification: “select the global batch first” determines optimizer-update membership. “Group examples into microbatches” determines which examples execute together. “Concatenate sequences into packs” determines the tensor representation. These are separate choices, and global-batch-first planning can be combined with either padded microbatches or true concatenation. The assistant's padded-grouping suggestion is a simpler alternative, not an interpretation already selected by the user.

| Execution policy | What one model forward receives | Meaning of a fixed local batch size |
| --- | --- | --- |
| Ordinary batching | Separate example rows, padded to a common length | Number of examples per forward per DP replica |
| Proposed variable-size padded grouping | The same batched tensor format, with example-row count chosen from lengths/capacity | Exact example count is variable; a fixed count would instead constrain the planner to length bucketing only |
| Official concatenated packing | Each row represents a concatenation of several examples, with document boundaries and isolated attention; THD represents boundaries through variable-length metadata | Number of packs per forward; original examples inside the packs can vary |

Illustration using processed sequence lengths 1,000/2,000/3,000, before optional alignment or fixed-length padding: ordinary batching gives `[3, 3000]` input IDs, comprising 6,000 valid and 3,000 padding positions. A single concatenated pack logically gives `[1, 6000]` plus document boundary metadata. Packing is not equivalent to joining conversations with unrestricted causal attention; the samples must remain isolated. These position counts do not imply a guaranteed attention-speed ratio, which depends on the backend.

A DP rank can execute multiple samples in one model call. Its collator constructs IDs/labels/masks with a batch dimension and accompanying image tensors/ownership metadata; these are moved to that rank's device and passed to the model. The sample rows are processed as a batched forward, not separate Python calls. TP/CP ranks within that DP replica cooperate on the same logical batch. The current optical backbone already pads multiple examples into a batch in `FrozenBackbone.hidden`.

The added benefit of the proposed variable grouping is automatic example-count adaptation and possible padding reduction: short examples can share a larger microbatch, long examples a smaller one, while all sampled occurrences still belong to the same global optimizer update. The grouping operation itself does not remove padding or guarantee balanced compute. Under an unchanged fixed local example count, it can reorder samples to reduce padding but cannot increase/decrease samples per forward. Under fixed packs-per-forward, global-batch-first concatenated packing is also possible; it requires an appropriate total pack count and dynamic accumulation rounds. No new packing choice is confirmed by this explanation.

R10 assessment: selective target positions add bookkeeping but are not the principal difficulty. Each branch already needs its own prediction-position map because teacher/student input lengths differ. With concatenation, a position becomes that branch's sample start offset plus its already-causally-aligned within-sample prediction position. Align returned logits by sampled occurrence and target index, not by equal absolute sequence positions. Attention/state isolation, media/position handling, independent mesh schedules, and bounded logit routing account for most of the additional complexity.

Simplification confirmed by R11: implement the initial AutoModel migration with ordinary padded microbatches and the existing accumulation/bridge flow, retaining separate teacher inputs and selected-logit outputs. Use manually configured local batch size and a fixed global example batch. Defer adaptive global-batch grouping, independent adaptive regrouping, and true concatenation. The adaptive-batching goal remains a possible later optimization; it is not required to finish the initial migration.

### Visual annotations and bins

Current data stores messages plus offset-addressed `visual_areas`. The compiler inserts model-visible vision markers and builds separate target-position maps. Inline storage annotations can compile to the same internal maps; accepting inline tags does not eliminate alignment.

Both current bin lists are `1`, `2`, `3-4`, `5-8`, `9-16`, `17+`. Stack requests 1/2/4/8 images. Each image covers at most 50 display rows by default, with long source lines wrapping at 100 columns. SWE windows request 1/2/4/8 visual-observation turns; full trajectories can occupy larger turn bins. A turn counts an assistant action owning visual observations; it is not every message. One turn can generate multiple images. Long enough tool observations (default >=256 characters) become visual.

The current leaf is `<source>/<task>/<view_family>/images-<bin>/turns-<bin>`. Families are front/middle/observation/window/full. View suffix details are folded into these families. Binning does not independently resplit train/eval in the current code, but it changes sampling mass and per-leaf evaluation selection.

Confirmed replacement examples:

```text
stack_front/reconstruction/images-1
stack_middle/continuation/images-3-4
swe_qwen38_27b/reconstruction/images-1
swe_qwen38_27b_window/next_action/turns-3-4
swe_qwen38_27b_full/next_action/turns-17+
```

The same simple/window/full source distinction applies to Qwen3.5-122B. These examples illustrate spelling; the confirmed distinction is logical source, then task, then one primary size bin. There is no view level or image-by-turn cross-product. Full trajectories remain recognizable through their logical source.

### Sampling audit

Current source weights are applied before task weights, which normalize over available tasks per source. With balance_slices enabled, populated leaves within each source/task receive equal mass and examples within a leaf share it. Draws use replacement and a deterministic global list partitioned across DP ranks.

The inspected smoke selection contains 1,690 training rows. Each original source has 1/3 draw mass. Its resolved task masses are reconstruction 44.44%, continuation 27.78%, next action 27.78%. Qwen3.8 continuation has no eligible rows, so its mass was redistributed within that source. Example probabilities range from 0.0000621427 to 0.0138889, approximately a 224-fold difference. One rare-leaf example expects 23.47 draws in a nominal epoch; a common-leaf example expects 0.105. These are consequences of the chosen balancing policy, not a normalization arithmetic failure.

For P04, the desired probability of a leaf is an example-draw share, not automatically an equal contribution to token-normalized loss. Ratios over a finite batch fluctuate. Preserve underlying source/repository identity as provenance, independently of the newly confirmed sample-level evaluation split.

Filtering is not necessarily whole-leaf removal:

- Intentional source/task/bin selection may exclude complete leaves. An image-count restriction can also remove only some examples within a leaf.
- Eligibility checks, such as teacher/student sequence limits or missing supervised targets, reject individual examples. Enough rejected examples can leave a leaf empty.
- Curation also has eligibility rules, such as which tool observations qualify for rendering. Data integrity failures and invalid configuration should remain explicit errors.

For eligibility filtering, report absolute counts before and after, percentage removed, rejection reasons, and effects on each leaf. The percentage denominator must be stated for each stage; do not sum overlapping reason percentages as if they were disjoint. Resolve the training mixture over the surviving eligible rows and log that resolved table. C08 and P06 confirm reporting and empty-leaf/shortfall handling. An emptied leaf can report, for example, `20 before, 0 retained, 20 removed (100%)`; evaluation can report `16 requested, 7 available, shortfall 9 (56.25%)`.

Under the confirmed leaf-weight semantics, filtering some rows out of a surviving leaf does not reduce that leaf's requested probability. Its remaining rows are therefore sampled more frequently. Removing an entire leaf changes the normalization denominator. Reporting per-leaf retained sizes makes both effects visible.

Confirmed relative-weight semantics (P04, accepted in R6): suppose parent A has two leaves and weight 3, and parent B has three leaves and weight 1. The five leaf weights are `3, 3, 1, 1, 1`. Their sum is 9, so A's leaves each get probability 1/3 and B's leaves each get 1/9. The parent totals are 2/3 and 1/3. Setting an ancestor weight provides a default to all descendant leaves; a nearer override replaces it. Adding/removing leaves changes parent totals, which is a deliberate consequence of this design rather than a normalization error. Root weight 1 yields equal leaf probabilities by default.

### Loss and batching

The current loss already invokes the official AutoModel KDLoss. Custom code gathers aligned hidden states and projects them in recomputed chunks; CE is directly computed with cross_entropy. Stock KDLoss requires matching teacher/student logit leading dimensions and a matching target mask.

The actual current path on each rank is:

```text
teacher inputs -> frozen Qwen, no_grad -> full final hidden states
               -> gather supervised prediction positions -> selected teacher hidden states
student inputs + adapted visual embeddings -> frozen Qwen, gradients to adapter
               -> gather corresponding prediction positions -> selected student hidden states
both selected hidden-state arrays -> frozen LM head, 64 targets per chunk
                                 -> teacher/student logits -> official KD criterion + CE
```

Teacher outputs are targets for the loss, not input embeddings fed into the student. The current implementation has no cross-mesh teacher transfer: it reuses one Qwen per rank. Selected states are flattened over examples as `[sum(target_counts), hidden_size]`, and projected chunks have shape `[chunk_targets, vocabulary]`; `[batch, target_positions, vocabulary]` is the equivalent padded conceptual representation. Each branch uses its own causal prediction-position map, with corresponding target IDs in the same order.

Confirmed R5 boundary: teacher forward -> select supervised hidden positions -> LM head -> detached full-vocabulary selected logits -> student-side official KD criterion. The model output must be logits; the current local helper returning hidden states is not the future teacher interface. Keep targets and example identity aligned across the two sequences.

There are two different kinds of chunking:

- Current optical code splits selected target positions into chunks of 64 before the LM head and computes projection, CE, and KD per chunk. A 2,048-target example uses 32 such chunks. This does not repeat the Qwen transformer forward or restrict its context. It adds Python iterations/kernel launches and uses activation checkpointing to recompute projection/loss intermediates during backward. The checkpoint covers the loss block, so teacher-head calculations can also be recomputed; the teacher transformer is not rerun by this particular checkpoint.
- Official `KDLoss(chunk_size=...)` receives already-created logits. Its optional loop divides valid-token probability calculations into chunks; the default is 0, and the TP path ignores this option in favor of distributed vocabulary calculations. It does not avoid the full incoming logit tensors, masking copies, or default FP32 upcasts.

Both preserve the full vocabulary and the intended token-summed objective, apart from floating-point reduction differences. Looping has overhead; checkpointing adds recomputation. Larger chunks generally reduce launch overhead but require more memory, and the best size has not been benchmarked. Simply looping over projections and concatenating their logits does not eliminate the final selected-logit allocation.

The official chunk loop also is not a guarantee that only one chunk's autograd state remains resident until backward. A CPU-only `saved_tensors_hooks` check of the pinned loss, using 16 tokens and vocabulary 128, found identical loss values and 18,432 bytes of saved vocabulary-shaped tensors for chunk size 0 versus 4; largest individual tensors fell from 8,192 to 2,048 bytes. This checks saved tensors in a tiny example, not full training peak memory. It establishes why the current checkpointed path and ordinary official loss chunking should not be described as equivalent memory optimizations.

The stock VLM recipe calls teacher/student models for sequence logits, then CE/KD, then backward. Separate-mesh transport materializes teacher logits across vocabulary/context shards before returning them. The confirmed extension changes which positions are projected and aligns targets, while retaining that logit interface and framework machinery.

R6 resolves P17: remove our position-chunk loop and its checkpoint wrapper, and set the official KD position-chunk option to 0. The official slice is `t_logits[start:end]` / `s_logits[start:end]` on `[valid_positions, vocabulary]`, so the first (position) axis is divided and every slice retains the entire vocabulary. TP vocabulary sharding instead distributes vocabulary columns across ranks and uses collective softmax/KL calculations; it is not a sequential chunk loop. Preserve supported framework TP behavior without adding another vocabulary loop. The confirmed image chunking is separate and remains. Memory-capacity planning must account for full selected-position logits and official loss intermediates under P08; do not silently restore the rejected position loops.

R7 reaffirms disabling this chunking for now. This is the confirmed implementation direction; it is not a claim that the current training code/configuration has already been changed, nor a permanent prohibition on discussing another memory policy if future measurements motivate one.

On September 17, a CPU-only server snippet read the pinned cached Qwen configuration: vocabulary 248,320; hidden width 2,560; 32 text layers. The frozen model has 4,205,751,296 parameters, corresponding to about 7.83 GiB of BF16 weights. Raw tensor arithmetic for batch size 1 and one branch:

| Positions projected | BF16 logits | FP32 logits | BF16 final hidden states |
| --- | ---: | ---: | ---: |
| 64 | 0.030 GiB | 0.059 GiB | 0.0003 GiB |
| 2,048 | 0.947 GiB | 1.895 GiB | 0.0098 GiB |
| 4,096 | 1.895 GiB | 3.789 GiB | 0.0195 GiB |
| 32,768 | 15.156 GiB | 30.313 GiB | 0.1563 GiB |

Formula: `batch * positions * vocabulary * bytes_per_element / 2**30`. Two simultaneously resident 32K BF16 teacher/student logit tensors would total about 30.31 GiB before FP32 conversions, gradients, and probability intermediates. Real teacher/student lengths differ. This is tensor-size arithmetic, not a measured peak-memory or speed result. Full training activations include all layers and depend on checkpointing/attention; the final hidden-state column is not their total. Selecting targets helps in proportion to the removed positions; chunked projection still helps if most positions are supervised. Separate-mesh communication also makes unnecessary dense outputs expensive.

Current training uses a fixed number of conversations per microbatch and optimizer update. Token limits reject whole examples; they are not a token-budget batch scheduler. AutoModel's pinned VLM loader supports optional neat/THD packing with a target pack_size and media balancing, while its optimizer step scheduler still uses local/global batch counts. The generic dataset loader also has custom batch-sampler extension points; there is no already-wired paired optical token-budget mode.

R4 clarifies that “packing” means allowing the number of examples per forward/backward to adapt, without manually tuning an example-count microbatch size. Sequence packing and adaptive batching are related but different: concatenating several short examples into a bounded pack makes the original-example count per pack vary, but still requires configuring pack capacity and packs per microbatch. The inspected framework does not automatically choose a GPU-memory-safe microbatch size.

In the deferred proposal, “paired token budgets” means keeping the same logical sample identities and targets for the update/request while allowing each branch its own capacity limit and microbatch partition. It does not require identical per-forward example counts, equal sequence lengths, or identical token counts. Teacher and student caps can differ because one is inference-only and their sequences/activations differ. Under padded batching, capacity estimation must include padding; raw token sums alone can underestimate cost. R11 defers this scheduling policy and selects manually configured ordinary microbatches initially.

Concatenated packing additionally needs attention and recurrent-state isolation for Qwen, correct media offsets, and preservation of paired target ordering. The current optical wrapper has not been validated for these packed formats. Some stock packing paths skip oversized samples or discard overflowing whole examples; any future reuse must expose those removals to filtering and sampling accounting. R11 defers concatenation and adaptive scheduling, so this validation is not required for the initial migration.

Native VLM processing: load conversations/media -> configurable Hugging Face AutoProcessor/chat template -> token IDs, pixel tensors and media metadata -> collator/padding/optional packing -> GPU VLM vision/projector/language forward -> loss. Processor work can happen in data-loader workers. GPU vision encoding belongs to the model; a combined processor API does not remove image batching. Our renderer is an additional text-to-image stage.

The confirmed encapsulation (P16, conditional on AutoModel compatibility) is a normal processor/model interface: the processor handles text rendering, image preprocessing, tokenization, and position metadata; the optical model handles GPU vision encoding in bounded chunks, adaptation, and language-model forward. The same implementation can be called by training and generation. GPU model execution should not move into CPU data-loader workers. Generation additionally needs model-side embedding and cache handling.

Putting the same GPU operations behind a method boundary does not itself improve or reduce their efficiency. Relevant differences are preprocessing parallelism/prefetch, host-to-device copies, chunk sizes, and opportunities to overlap CPU/GPU work. No processor-versus-separate-stage speed benchmark was run. A model-integrated vision stage can retain exactly the separate image-encoder chunk control accepted under C04.

Image orchestration for the current frozen, fixed-resolution encoder: collect images for the selected student microbatch in their original order, preprocess/encode groups up to `image_microbatch_size`, retain the compact features, and map them back into each conversation before the adapter/language forward. The current size is 4: ten images require encoder calls of 4/4/2. This grouping is independent of the number of conversations or teacher requests. The teacher text branch does not need image encoding.

Fixed image chunk size and `no_grad` bound encoder intermediates, while compact retained features still grow with image count. Each image contributes 111 student visual positions in the current configuration, so student sequence cost includes that contribution. This can make a separate image token-budget scheduler unnecessary initially, but does not make rendering/vision compute free or prove it negligible. Many-image, short-text batches may be vision-bound; use the accepted vision timing and image totals before making a speed claim. Integrate execution with the framework's model/pre-embedding hooks to respect TP/CP placement.

The native default collator enables truncation unless configured otherwise, and some preprocessing paths replace rejected examples. Reusing framework components must preserve the original experiment's explicit whole-example rejection and complete-trajectory requirements, not inherit those defaults accidentally.

vLLM inference scheduling is a separate concern: its continuous scheduling/chunked prefill uses a token budget across requests and decode/prefill work. That is not a training optimizer-batch policy and does not directly solve paired-sequence training.

### Evaluation pools and metrics

Current code at the inspected baseline uses repository-hash assignment with eval_fraction=0.1, not per-leaf sample splitting. At training startup, filtering and preflight determine eligibility and a deterministic sorted per-leaf cap selects actual evaluation examples. The selected set remains fixed within a run. Confirmed P09/P10 replace the curation split policy and expose hierarchical runtime selection; this change is not implemented yet.

Smoke artifacts contain 530 evaluation rows in 53 prepared leaves. After image/length restrictions, 256 rows in 30 leaves remain eligible. With one per leaf, 30 examples and 19,120 target tokens are evaluated, plus 10 reconstruction generations. A cap of 16 on that same eligible pool selects 192 rows. This is not the scale of a validated full-source/32K run.

Current logging emits seven teacher-forced metrics for each leaf and one global aggregate: loss, CE, KL, teacher CE, argmax agreement, target-token accuracy, tokens. Generation adds CER, LER, generated samples, character edits, line edits, and generation-limit fraction. Smoke emitted 217 eval-* keys without generation and 283 with generation, plus val_loss/bookkeeping. No source/task marginal aggregates are currently emitted.

Under confirmed P09/P10, curation selects a seeded sample fraction independently from each leaf and records fixed holdout membership. The remaining samples are training data. At training setup, source/task/bin selection controls diversity and inherited per-leaf counts control scale within this holdout pool. Examples not selected for a particular evaluation run remain held out; they are not returned to that run's training pool. Record curated holdout IDs and runtime-selected IDs. Tiny leaves and runtime eligibility can cause shortfalls; confirmed P06 uses available rows and reports absolute/percentage shortfalls. Numeric fraction/rounding defaults remain configuration details.

This design guarantees disjoint selected sample IDs, not unseen repositories or trajectories. Related front/middle/window/full examples from the same underlying file or trace can appear on opposite sides. Report results as sample-level validation; do not silently reintroduce repository grouping or claim repository-level generalization.

### Legacy dependencies

Moving training/config.py alone does not allow deletion of training/:

- benchmark/prepare.py imports training.data.load_manifest (old-format manifest validation).
- inference/backend.py imports training.models.DeepSeekVision and build_adapter and expects old adapter.json/state.json/adapter.safetensors checkpoints.
- scripts/validate_live_backend.py imports TensorCache, FrozenQwen, and old manifest loading.
- older tests import old objectives, resume, generation, caches, and trainer functions.

The shared MLP is already in adapters.py and the new encoder is already in automodel/vision.py. The old full training modules need not be retained merely to keep small shared helpers: relocate/rewrite those helpers and migrate or retire their callers. Historical results can remain as documentation without preserving the retired runtime.

### Throughput, timing, and logging semantics

Measured smoke updates on two A6000s: about 12 seconds for three global-batch-2 updates, 3.7-4.4 seconds for resumed individual updates, and 8.2 seconds for a global-batch-4 accumulation check. Teacher-forced evaluation of 30 rows took about 29 seconds; adding ten capped generations took about 50 seconds. No steady-state full-source/32K benchmark exists.

Current inherited tps counts our compact supervised labels, not complete teacher/student input sequences. Teacher-forced model inputs include ground-truth assistant tokens; calling their full length “prompt-only tokens” would be inaccurate. Proposed canonical names use teacher_input/student_input and separate loss_tokens. Exact prompt-prefix counts can be added only with a defined multi-turn meaning.

Measurement definitions and remaining optional fields (P14/P15):

- Input counts: unpadded positions actually supplied to each branch; student count includes visual embeddings. Record padding overhead separately if desired.
- Loss tokens: aligned supervised targets.
- Batch: global optimizer update across accumulation microbatches and DP ranks; count examples once, not once per TP shard.
- Intended leaf ratio: resolved absolute example-draw probability.
- Per-update observed ratio: consumed leaf examples / all consumed examples in that update. Emit zero for absent leaves rather than leaving stale last values.
- Equivalent leaf epochs: cumulative consumed leaf examples / eligible distinct leaf examples; not unique coverage.
- End-to-end input TPS: global input count / elapsed update wall time. Optional stage TPS uses that stage's elapsed critical-path time and should be named separately.
- CPU stage timers and GPU CUDA-event timers are different quantities; avoid per-operation GPU synchronization. Maximum rank duration, rather than summed GPU durations, describes the distributed critical path.
- Student backward timing includes checkpoint recomputation and may include communication; it is not pure mathematical gradient-kernel time.
- Data exposure counters must reflect consumed updates and survive resume; do not count prefetched items as trained.

Batch size is the number of original dataset records/conversations consumed in one global optimizer update. Suggested metric name: `batch/size`. With a configured global batch of 16 it would normally be 16; adaptive microbatching need not change this global count. With packing, count original records inside packs. R5.4 confirms logging this size and declines per-example means.

Retain teacher/student input-token totals, loss-token totals, and image totals per update, optionally averaged over the logging interval as requested. Omit curves formed by dividing each quantity by the example count. For example, an update consuming 16 examples, 32,000 teacher input positions, 16,000 student input positions, 8,000 targets, and 48 images logs these workload totals and batch size 16. It does not additionally log 2,000/1,000/500/3 per-example means. This reporting choice does not alter token-weighted loss normalization.

For timing, use wall-clock durations for CPU stages and CUDA events for GPU stages, collected without synchronizing after every substep. Teacher forward, vision/adapter forward, student forward, loss, backward, and optimizer timings may aid diagnosis. End-to-end wall time remains the throughput denominator; overlapping stage times need not add to it. Instrumentation overhead and distributed attribution require focused validation.

No explicit charts/ prefix or Charts section assignment was found in our code or the inspected AutoModel training logger. Training currently emits bare loss/ce_loss/kd_loss/lr/tps/etc. W&B sections normally reflect metric-key hierarchy and can be renamed. The user's exact live dashboard layout was not inspected; do not assert its section origin as a verified UI fact.

## Remaining decisions after R11

These are unresolved design choices, not a request to reconfirm the accepted architecture. Routine names, seeds, numeric run parameters, paths, and timer implementation can be made concrete through configuration without adding new architecture approval gates.

| Pending IDs | Decision still needed | Proposed choice for confirmation |
| --- | --- | --- |
| None | P11 was confirmed during implementation on September 17. | Generation counts and output limits inherit independently; numeric values are ordinary YAML configuration. |

R11 settles initial batching: ordinary padded microbatches and existing accumulation, with packing/adaptive regrouping deferred. P11 was subsequently confirmed on September 17, settling the remaining generation-budget policy. Numeric batch sizes, mesh topology, and memory validation remain implementation/configuration work. See the implementation status at the top of this document and the current training guide for the resulting code.

## Proposed implementation order

1. Make the editable fork arrangement concrete under confirmed P01/P02, preserving the normal CLI/YAML and existing recipe/mesh infrastructure.
2. Make ordinary local/global batch sizes and mesh topology concrete under confirmed P08; resolve the minor P11 generation-budget policy. Do not reopen deferred packing or reconfirm accepted data, image, metric, or retirement decisions.
3. Integrate confirmed annotations, source taxonomy, data accounting, and evaluation selection under that architecture, applying only approved mixture semantics.
4. Integrate confirmed P07 logit routing, P16 processor/model interface, and P17 removal of position-chunk loops, under the selected P08 batching design. Validate memory and model support using focused server checks.
5. Migrate retained inference/benchmark callers and remove retired training code under P13.
6. Add agreed metric names/counters/timers and perform focused server validation, including resume and mixture accounting.

This sequence is a proposal. It does not authorize work on still-pending IDs.

## Source anchors

- Local current implementation: [recipe](../src/optical_adaptor/automodel/recipe.py), [data/sampler](../src/optical_adaptor/automodel/data.py), [conversation curation](../src/optical_adaptor/automodel/conversations.py), [processing](../src/optical_adaptor/automodel/processing.py), [loss/backbone](../src/optical_adaptor/automodel/model.py), [configuration](../configs/automodel.yaml).
- [Pinned AutoModel KD recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/recipes/vlm/kd.py).
- [Pinned KD mesh bridge](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/recipes/kd_utils.py).
- [Pinned VLM loader](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/components/datasets/vlm/loader.py).
- [Pinned VLM collators](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/components/datasets/vlm/collate_fns.py).
- [Pinned KD loss](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/components/loss/kd_loss.py).
- [Pinned AutoModel VLM packing implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/components/datasets/vlm/neat_packing_vlm.py).
- [vLLM scheduling/processing reference](https://docs.vllm.ai/en/latest/configuration/optimization/).
- [W&B panels and sections](https://docs.wandb.ai/models/app/features/panels).
- Server measurements read from /workspace/optical-adaptor/outputs/automodel/smoke/ and accum-smoke/, at baseline f6ba90f. No GPU experiment was run for this discussion.
- R5 used static inspection and a tiny CPU-only `uv run python -c` saved-tensor check of the pinned official loss. No training code changed and no GPU benchmark ran.
