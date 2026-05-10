# Ablations

This file is the experiment ledger for the submission repo. Failed runs are intentionally retained because they explain why the final PCCP Ensemble is narrow, reproducible, and objective-aligned.

Acceptance gate: a method must beat the honest rank-8 LoRA baseline of `0.8177` validation accuracy, improve the selected final submission path, or be documented as a negative result.

## Current Evidence

| Experiment | Config / command | Validation result | Status | Interpretation |
| --- | --- | ---: | --- | --- |
| Rank-8 Attention+MLP LoRA | `configs/exp_lr3e-4.json` | `0.8177` | Baseline | Strong train-only selector. MLP targets matter for concept transformations. |
| Connector-LoRA r8/r16 | `configs/budgetmax_connector_r8.json` | `0.8139` single checkpoint | Complementary | Slightly below the baseline alone, but useful because it adapts the visual-language bridge. |
| Connector-LoRA + Prompt/Choice TTA | `scripts/run_ablation_suite.sh` | `0.8254` | Accepted | Best honest validation recipe observed locally. Presentation-robust scoring helps. |
| Rank-8 LoRA + Prompt/Choice TTA | `scripts/run_ablation_suite.sh` | `0.8206` | Accepted diagnostic | Confirms TTA helps beyond one prompt/order. |
| Length-calibrated answer-text likelihood | `src/inference.py --length_calibrated` | approx. `0.445` | Rejected | Text likelihood measures fluency and length effects, not answer correctness. |
| Exact row-similarity resolver | validation diagnostic | `0.6813` | Rejected | Similar wording often hides different visual evidence. Not part of the maintained code path. |
| Rank-6 DoRA high-resolution augmentation | `configs/r6_dora_attn_mlp_nosplit768_aug.json` | `0.7433` | Rejected | The tested rank/optimization tradeoff underfit relative to rank-8 LoRA. |
| Final Train+Validation Connector-LoRA | `configs/final_trainval_budgetmax_connector_r8.json` | fit behavior only | Final fit | Used only after selection; not held-out validation evidence. |
| PCCP Ensemble | `bash scripts/run_final_push.sh` | final submission path | Final method | Combines provided-context captions, snapshots, adapter soup, prompt/choice TTA, and optional image TTA. |

## Reproducible Ablation Matrix

| ID | Hypothesis | Command / config | Output | Acceptance criterion |
| --- | --- | --- | --- | --- |
| A1 | Direct answer-letter CE is the correct objective. | Main training configs | validation score dumps | Must beat length-calibrated answer-text scoring. |
| A2 | Rank-8 attention+MLP LoRA is a strong legal baseline. | `configs/exp_lr3e-4.json` | `runs/r8-letter-nosplit512-lr3e-4/` | Reproduce near `0.8177`. |
| A3 | Connector adaptation adds complementary visual grounding. | `configs/budgetmax_connector_r8.json` | `runs/budgetmax-r8-connector16-nosplit512-lr3e-4/` | Keep if TTA or ensemble improves validation-gated score. |
| A4 | Compact metadata improves reasoning. | `--metadata_fields subject grade topic` | score CSVs | Keep only if global validation improves. |
| A5 | Prompt variants reduce prompt sensitivity. | `--prompt_variants default exam context_first no_metadata answer_phrase` | prompt-TTA score dumps | Keep if averaged scores improve a single prompt. |
| A6 | Choice-order TTA reduces letter-position bias. | `--choice_tta deterministic --choice_tta_max 8` | choice-TTA score dumps | Keep if global validation improves. |
| A7 | Image-mode TTA helps charts and diagrams. | `--tta_image_modes nosplit512 nosplit768 split` | image-TTA score dumps | Keep if global validation improves. |
| A8 | Provided-context captions stabilize visual text. | `scripts/build_provided_captions.py` plus captioned inference | `data/pccp_captioned/*.csv` and score dumps | Keep only if captioned validation beats non-captioned validation. |
| A9 | Score averaging beats a single score dump. | `scripts/search_score_ensemble.py` | `runs/eval/best_ensemble.json` | Use only if validation-gated. |
| A10 | Late snapshots contain useful trajectory diversity. | `scripts/select_checkpoints.py` plus final scoring | `runs/final_push_pccp/*_scores.csv` | Final-only; compare by leaderboard and report as final-fit behavior. |
| A11 | Same-shape adapter soups improve stability. | `scripts/soup_adapters.py` | `runs/final_push_pccp/final_family_adapter_soup/` | Final-only unless separately validated on train-only runs. |

## How To Update This File

After a new run, record:

- exact command and config;
- trainable parameter count from `run_metadata.json`;
- training split used;
- validation score CSV path;
- validation accuracy;
- subgroup failures from `scripts/evaluate_scores.py`;
- accepted/rejected/final-only status;
- the reason for the decision.

The most important reporting rule is to keep honest validation separate from final train+validation fit behavior.
