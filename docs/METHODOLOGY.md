# Methodology

Pix2Pred is a provided-data-only vision-language multiple-choice system. The repository is organized around one reproducible path: build prompts from the provided rows and images, adapt the official SmolVLM checkpoint with a parameter-efficient adapter, score valid answer letters directly, and aggregate auditable per-choice score dumps into `submission.csv`.

## Task Framing

Each row contains an image, a question, a JSON list of choices, optional hint and lecture text, and metadata such as grade, subject, and topic. The model must output one zero-indexed answer index. Because the metric is classification accuracy, the training objective is also classification: the model scores only the final-token logits for answer letters `A` through `E`, masks letters beyond the row's `num_choices`, and optimizes cross-entropy on the correct index.

This avoids brittle generated-answer parsing. A generated response can be verbose, malformed, or semantically correct but hard to map to an index. Letter-logit scoring makes the decision space explicit and keeps training, validation, inference, and CSV submission aligned.

## Adapter Strategy

The honest selector baseline uses rank-8 LoRA over language attention and MLP projections:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

This reached `0.8177` validation accuracy and is the reference point for method selection. MLP targets are included because scientific reasoning is not only attention routing; the model also has to transform evidence into concepts such as inheritance probability, force, erosion, or classification.

The final selected adapter adds a higher-rank vision-language connector update:

```text
modality_projection.proj
```

The final Connector-LoRA family keeps language projections at rank 8 and uses rank 16 on the connector, producing `4,996,096` trainable parameters. This stays below the 5M cap while adapting the visual bridge used to inject image features into the language model.

## Prompting

The prompt always asks for one answer letter. The controlled variants are:

| Variant | Purpose |
| --- | --- |
| `default` | Compact metadata, question, context, and choices |
| `exam` | Concise exam-style instruction |
| `context_first` | Tests whether hint and lecture should prime reasoning |
| `question_first` | Tests whether the question should anchor attention |
| `no_metadata` | Measures whether metadata adds signal or noise |
| `answer_phrase` | Tests `"The correct answer is:"` as the final cue |
| `caption_first` | Places provided visual context before other support text |

The main recipe uses compact metadata fields: `subject`, `grade`, and `topic`. These fields are short enough to be useful priors without crowding the question and choices.

## Provided-Context Caption Prompting

PCCP extracts figure-style descriptions already present in the provided row text and writes them into a stable `caption` column. The operation does not add information; it normalizes where the model sees visual-context text. This is useful because some rows contain diagram descriptions inside the question, hint, or lecture fields, and small VLMs are sensitive to where such text appears.

The caption builder writes both captioned CSVs and manifests so the process is auditable:

```text
data/pccp_captioned/train_captioned.csv
data/pccp_captioned/val_captioned.csv
data/pccp_captioned/test_captioned.csv
```

Generated captioned CSVs are reproducible artifacts, not source data, so they are rebuilt by scripts instead of being required in the repository.

## Image Handling

The default image mode is `nosplit512`. The final inference path can average `nosplit512`, `nosplit768`, and `split` when `TTA_IMAGE_MODES` is set. This is important for scientific diagrams because text, axes, and small labels may be easier to read under one image preprocessing mode than another.

Training configurations can include conservative image augmentation:

| Augmentation | Rationale |
| --- | --- |
| very small rotation | Robustness to rendering or scan variation |
| tight random crop | Framing robustness without cutting content |
| mild brightness/contrast/saturation jitter | Display and compression robustness |
| no horizontal flip | Diagrams often have directional semantics |

## Inference and Ensembling

The final PCCP path averages comparable per-choice score vectors from:

- selected late checkpoints from the final train+validation family;
- an optional same-shape adapter soup built from those checkpoints;
- prompt variants;
- deterministic choice-order permutations;
- optional image modes.

Every variant is mapped back to the original choice order before averaging. The final ensemble is therefore easy to audit: the score CSV contains the exact values used to produce every answer.

## Final Fit Protocol

Model selection is done using honest train/validation runs. After choosing the recipe, the final adapter is trained on `train.csv + val.csv` with `allow_val_label_training=true` in the final config. This final fit is used only for hidden-test submission; it is not reported as held-out validation evidence.

The final submission script:

1. builds PCCP captioned CSVs when requested;
2. trains or reuses the selected Connector-LoRA final adapter;
3. selects late checkpoints;
4. scores test rows with prompt, choice, and optional image TTA;
5. builds and scores an adapter soup when enabled;
6. averages score dumps;
7. validates the final `id,answer` CSV.

The validation guard checks exact columns, row count, ID equality, integer answers, and per-row answer bounds.
