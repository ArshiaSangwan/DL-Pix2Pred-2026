# Results Template

Use this template for each serious run. Copy one block per experiment into your report, notebook, or `docs/ABLATIONS.md`.

## Run

- Run name:
- Date:
- Operator:
- Seed:
- Config:
- Output directory:
- Base checkpoint:
- Adapter init:
- Train CSV:
- Validation CSV:
- Image directory:

## Method

- Adapter target modules:
- LoRA rank / alpha / dropout:
- DoRA or rsLoRA:
- Trainable parameters:
- Image mode:
- Max length:
- Prompt variant:
- Metadata fields:
- Captions:
- Image augmentation:
- Gradient checkpointing:

## Training

- Epochs requested:
- Early stopping:
- Batch size:
- Gradient accumulation:
- Learning rate:
- Warmup ratio:
- Weight decay:
- Runtime notes:

## Validation

- Best epoch:
- Best validation accuracy:
- Validation loss:
- Score CSV:
- Report path:
- Error buckets:
- Accepted for final recipe: yes/no
- Reason:

## Inference

- Checkpoint(s):
- Prompt TTA:
- Image TTA:
- Choice-order TTA:
- Score ensemble weights:
- Submission path:
- Submission validation:

## Notes

- What improved:
- What failed:
- Follow-up:
