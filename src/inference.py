import argparse
import hashlib
import itertools
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

try:
    from data_utils import (
        CHOICE_LABELS,
        build_chat_messages,
        build_chat_messages_with_choice,
        get_choice_token_ids,
        parse_choices,
    )
    from modeling import get_model_and_processor
except ModuleNotFoundError:
    from .data_utils import (
        CHOICE_LABELS,
        build_chat_messages,
        build_chat_messages_with_choice,
        get_choice_token_ids,
        parse_choices,
    )
    from .modeling import get_model_and_processor


def resolve_image_path(row, image_dir):
    row_image_path = str(row["image_path"])
    basename = os.path.basename(row_image_path)
    split_hint = basename.split("_", 1)[0] if "_" in basename else None
    candidates = []
    if image_dir:
        candidates.append(os.path.join(image_dir, basename))
    candidates.append(row_image_path)
    candidates.append(os.path.join(os.getcwd(), row_image_path))
    if row_image_path.startswith("images/"):
        candidates.append(os.path.join("data", row_image_path))
        candidates.append(os.path.join(os.getcwd(), "data", row_image_path))
    if split_hint in {"train", "val", "test"}:
        candidates.append(os.path.join("data", "images", split_hint, basename))
        candidates.append(os.path.join(os.getcwd(), "data", "images", split_hint, basename))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not resolve image for {row.get('id', '<unknown>')}: {candidates}")


def autocast_context(device):
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def final_token_logits(outputs, attention_mask):
    last_positions = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
    return outputs.logits[batch_idx, last_positions, :]


def to_device(inputs, device):
    inputs.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in inputs.items()}


def load_image(row, image_dir):
    with Image.open(resolve_image_path(row, image_dir)) as image:
        return image.convert("RGB")


def make_prompt(
    processor,
    row,
    prompt_variant,
    include_metadata,
    metadata_fields,
    lecture_max_chars,
    include_caption,
    caption_max_chars,
):
    return processor.apply_chat_template(
        build_chat_messages(
            row,
            include_metadata=include_metadata,
            metadata_fields=metadata_fields,
            lecture_max_chars=lecture_max_chars,
            prompt_variant=prompt_variant,
            include_caption=include_caption,
            caption_max_chars=caption_max_chars,
        ),
        add_generation_prompt=True,
        tokenize=False,
    )


def row_with_choices(row, choices):
    updated = row.copy()
    updated["choices"] = json.dumps(list(choices), ensure_ascii=False)
    updated["num_choices"] = len(choices)
    return updated


def choice_permutations(num_choices, mode="none", max_permutations=8, seed_text=""):
    base = tuple(range(num_choices))
    if mode == "none" or num_choices <= 1:
        return [base]
    if mode == "exhaustive" and num_choices <= 5:
        perms = list(itertools.permutations(base))
        return perms[:max(1, max_permutations)] if max_permutations else perms

    candidates = [base, tuple(reversed(base))]
    for shift in range(1, num_choices):
        candidates.append(tuple(base[shift:] + base[:shift]))

    rng_seed = int(hashlib.sha256(str(seed_text).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    while len(candidates) < max_permutations:
        perm = list(base)
        rng.shuffle(perm)
        candidates.append(tuple(perm))

    deduped = []
    seen = set()
    for perm in candidates:
        if perm in seen:
            continue
        seen.add(perm)
        deduped.append(perm)
        if max_permutations and len(deduped) >= max_permutations:
            break
    return deduped


@torch.no_grad()
def score_direct(
    model,
    processor,
    row,
    image,
    device,
    max_length,
    prompt_variant,
    include_metadata,
    metadata_fields,
    lecture_max_chars,
    include_caption,
    caption_max_chars,
    leading_space,
):
    prompt = make_prompt(
        processor,
        row,
        prompt_variant,
        include_metadata,
        metadata_fields,
        lecture_max_chars,
        include_caption,
        caption_max_chars,
    )
    choices = parse_choices(row["choices"])
    choice_ids = get_choice_token_ids(processor.tokenizer, len(choices), leading_space=leading_space)
    inputs = processor(text=prompt, images=image, return_tensors="pt", max_length=max_length, truncation=False)
    inputs = to_device(inputs, device)
    with autocast_context(device):
        outputs = model(**inputs)
        last_logits = final_token_logits(outputs, inputs["attention_mask"])[0]
    return last_logits[choice_ids].detach().float().cpu()


@torch.no_grad()
def score_direct_batch(
    model,
    processor,
    jobs,
    image,
    device,
    max_length,
    leading_space,
    batch_size,
):
    scored = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        prompts = [job["prompt"] for job in chunk]
        images = [image] * len(chunk)
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            truncation=False,
            padding=True,
        )
        inputs = to_device(inputs, device)
        with autocast_context(device):
            outputs = model(**inputs)
            logits = final_token_logits(outputs, inputs["attention_mask"])
        for row_idx, job in enumerate(chunk):
            choice_ids = get_choice_token_ids(
                processor.tokenizer,
                int(job["num_choices"]),
                leading_space=leading_space,
            )
            scored.append(logits[row_idx, choice_ids].detach().float().cpu())
    return scored


@torch.no_grad()
def score_choice_length_calibrated(
    model,
    processor,
    row,
    choice_text,
    image,
    device,
    max_length,
    prompt_variant,
    include_metadata,
    metadata_fields,
    lecture_max_chars,
    include_caption,
    caption_max_chars,
):
    prompt = make_prompt(
        processor,
        row,
        prompt_variant,
        include_metadata,
        metadata_fields,
        lecture_max_chars,
        include_caption,
        caption_max_chars,
    )
    prompt_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        max_length=max_length,
        truncation=False,
    )
    full_inputs = processor(
        text=processor.apply_chat_template(
            build_chat_messages_with_choice(
                row,
                choice_text,
                include_metadata=include_metadata,
                metadata_fields=metadata_fields,
                lecture_max_chars=lecture_max_chars,
                prompt_variant=prompt_variant,
                include_caption=include_caption,
                caption_max_chars=caption_max_chars,
            ),
            add_generation_prompt=False,
            tokenize=False,
        ),
        images=image,
        return_tensors="pt",
        max_length=max_length,
        truncation=False,
    )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_input_ids = full_inputs["input_ids"].to(device)
    inputs = to_device(full_inputs, device)

    with autocast_context(device):
        outputs = model(**inputs)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = full_input_ids[:, 1:]
    start = max(prompt_len - 1, 0)
    choice_token_count = labels.shape[1] - start
    if choice_token_count <= 0:
        return float("-inf")
    token_log_probs = log_probs[:, start:, :].gather(-1, labels[:, start:].unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item() / choice_token_count


def score_row(
    model,
    processor,
    row,
    image,
    device,
    args,
):
    choices = parse_choices(row["choices"])
    if not args.length_calibrated:
        jobs = []
        for perm in choice_permutations(
            len(choices),
            mode=args.choice_tta,
            max_permutations=args.choice_tta_max,
            seed_text=row["id"],
        ):
            permuted_choices = [choices[idx] for idx in perm]
            permuted_row = row_with_choices(row, permuted_choices)
            for prompt_variant in args.prompt_variants:
                jobs.append(
                    {
                        "perm": perm,
                        "num_choices": len(choices),
                        "prompt": make_prompt(
                            processor,
                            permuted_row,
                            prompt_variant,
                            not args.no_metadata,
                            args.metadata_fields,
                            args.lecture_max_chars,
                            args.include_caption,
                            args.caption_max_chars,
                        ),
                    }
                )
        batch_scores = score_direct_batch(
            model,
            processor,
            jobs,
            image,
            device,
            args.max_length,
            not args.bare_choice_tokens,
            max(1, int(args.tta_batch_size)),
        )
        original_scores = []
        for job, permuted_scores in zip(jobs, batch_scores):
            restored = torch.empty_like(permuted_scores)
            for permuted_idx, original_idx in enumerate(job["perm"]):
                restored[original_idx] = permuted_scores[permuted_idx]
            original_scores.append(restored)
        return torch.stack(original_scores, dim=0).mean(dim=0)

    all_scores = []
    for perm in choice_permutations(
        len(choices),
        mode=args.choice_tta,
        max_permutations=args.choice_tta_max,
        seed_text=row["id"],
    ):
        permuted_choices = [choices[idx] for idx in perm]
        permuted_row = row_with_choices(row, permuted_choices)
        variant_scores = []
        for prompt_variant in args.prompt_variants:
            if args.length_calibrated:
                scores = [
                    score_choice_length_calibrated(
                        model,
                        processor,
                        permuted_row,
                        choice,
                        image,
                        device,
                        args.max_length,
                        prompt_variant,
                        not args.no_metadata,
                        args.metadata_fields,
                        args.lecture_max_chars,
                        args.include_caption,
                        args.caption_max_chars,
                    )
                    for choice in permuted_choices
                ]
                permuted_scores = torch.tensor(scores, dtype=torch.float32)
            else:
                permuted_scores = score_direct(
                    model,
                    processor,
                    permuted_row,
                    image,
                    device,
                    args.max_length,
                    prompt_variant,
                    not args.no_metadata,
                    args.metadata_fields,
                    args.lecture_max_chars,
                    args.include_caption,
                    args.caption_max_chars,
                    not args.bare_choice_tokens,
                )
            original_scores = torch.empty_like(permuted_scores)
            for permuted_idx, original_idx in enumerate(perm):
                original_scores[original_idx] = permuted_scores[permuted_idx]
            variant_scores.append(original_scores)
        all_scores.append(torch.stack(variant_scores, dim=0).mean(dim=0))
    return torch.stack(all_scores, dim=0).mean(dim=0)


def validate_submission(submission_df, test_df):
    if list(submission_df.columns) != ["id", "answer"]:
        raise ValueError("submission.csv must have exactly columns: id, answer")
    if len(submission_df) != len(test_df):
        raise ValueError(f"submission row count mismatch: {len(submission_df)} != {len(test_df)}")
    if set(submission_df["id"]) != set(test_df["id"]):
        raise ValueError("submission ids do not exactly match the input CSV ids")
    merged = test_df[["id", "num_choices"]].merge(submission_df, on="id", how="left")
    if merged["answer"].isna().any():
        raise ValueError("submission contains missing answers")
    if not pd.api.types.is_integer_dtype(merged["answer"]):
        if not (merged["answer"].astype(float) == merged["answer"].astype(int)).all():
            raise ValueError("answers must be integer choice indices")
        merged["answer"] = merged["answer"].astype(int)
    invalid = (merged["answer"] < 0) | (merged["answer"] >= merged["num_choices"])
    if invalid.any():
        bad_id = merged.loc[invalid, "id"].iloc[0]
        raise ValueError(f"answer out of bounds for id={bad_id}")


def score_dump_row(row, scores, pred):
    values = {
        "id": row["id"],
        "num_choices": int(row["num_choices"]),
        "answer": int(pred),
    }
    sorted_scores = torch.sort(scores, descending=True).values
    values["margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    for idx, label in enumerate(CHOICE_LABELS):
        values[f"score_{label}"] = float(scores[idx]) if idx < len(scores) else None
    if "answer" in row and pd.notna(row["answer"]):
        values["truth"] = int(row["answer"])
        values["correct"] = int(pred == int(row["answer"]))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test/val CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="Paths to model checkpoints or base model")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--score_out", type=str, default=None, help="Optional CSV path for per-choice scores")
    parser.add_argument("--length_calibrated", action="store_true", help="Score each answer choice by length-normalized log-likelihood")
    parser.add_argument("--choice_tta", choices=["none", "deterministic", "exhaustive"], default="none", help="Average scores over deterministic answer-choice re-orderings")
    parser.add_argument("--choice_tta_max", type=int, default=8, help="Maximum choice permutations per row when choice TTA is enabled")
    parser.add_argument("--tta_batch_size", type=int, default=8, help="Batch prompt/choice TTA variants for each row")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum token length for prompts")
    parser.add_argument("--image_mode", type=str, default="nosplit512")
    parser.add_argument("--tta_image_modes", type=str, nargs="*", default=None, help="Average scores across image modes such as nosplit512 nosplit768 split")
    parser.add_argument("--prompt_variants", type=str, nargs="+", default=["default"], help="Prompt variants to average: default, exam, context_first, no_metadata")
    parser.add_argument("--lecture_max_chars", type=int, default=None)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--bare_choice_tokens", action="store_true", help="Use A/B/C tokens instead of space-prefixed answer tokens")
    parser.add_argument("--metadata_fields", type=str, nargs="*", default=None, help="Optional metadata fields to include, e.g. subject grade topic")
    parser.add_argument("--include_caption", action="store_true", help="Include a caption column in the prompt when present")
    parser.add_argument("--caption_max_chars", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_df = pd.read_csv(args.test_csv)
    image_modes = args.tta_image_modes or [args.image_mode]

    final_scores_by_id = {row["id"]: None for _, row in test_df.iterrows()}

    for ckpt_path in args.ckpts:
        for image_mode in image_modes:
            base_model, processor = get_model_and_processor(
                model_id=args.model_id,
                device=device,
                image_mode=image_mode,
            )
            print(f"Loading checkpoint: {ckpt_path} | image_mode={image_mode}")
            if os.path.isdir(ckpt_path) and "adapter_config.json" in os.listdir(ckpt_path):
                model = PeftModel.from_pretrained(base_model, ckpt_path)
            else:
                model = base_model
            model.eval()

            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Infer {Path(ckpt_path).name}/{image_mode}"):
                image = load_image(row, args.image_dir)
                pred_scores = score_row(model, processor, row, image, device, args)
                if final_scores_by_id[row["id"]] is None:
                    final_scores_by_id[row["id"]] = pred_scores
                else:
                    final_scores_by_id[row["id"]] = final_scores_by_id[row["id"]] + pred_scores

            del model, base_model
            if device == "cuda":
                torch.cuda.empty_cache()

    final_results = []
    score_rows = []
    for _, row in test_df.iterrows():
        scores = final_scores_by_id[row["id"]]
        pred = int(scores.argmax().item())
        final_results.append({"id": row["id"], "answer": pred})
        score_rows.append(score_dump_row(row, scores, pred))

    submission_df = pd.DataFrame(final_results)
    validate_submission(submission_df, test_df)
    submission_df.to_csv(args.out, index=False)
    print(f"Saved submission to {args.out}")

    if args.score_out:
        score_df = pd.DataFrame(score_rows)
        score_df.to_csv(args.score_out, index=False)
        if "correct" in score_df.columns:
            print(f"Score dump accuracy: {score_df['correct'].mean():.4f}")
        print(f"Saved score dump to {args.score_out}")


if __name__ == "__main__":
    main()
