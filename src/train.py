import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb

from dataset import SciQADataset
from modeling import count_trainable_params, get_model_and_processor
from data_utils import get_choice_token_ids


def resolve_device(requested_device: str) -> str:
    if requested_device != "cuda":
        return "cpu"
    try:
        if torch.cuda.is_available():
            torch.cuda.current_device()
            return "cuda"
    except Exception:
        pass
    raise RuntimeError("CUDA was requested but is not available in the current runtime.")


def autocast_dtype(device: str):
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_available():
        return torch.float16
    return None


def autocast_context(device: str):
    dtype = autocast_dtype(device)
    if dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_path_list(value):
    if isinstance(value, (list, tuple)):
        return [Path(path).resolve() for path in value]
    return [Path(value).resolve()]


def enforce_no_validation_label_leakage(config):
    if config.get("allow_val_label_training", False):
        print("WARNING: allow_val_label_training=True; validation labels may be used for training.")
        return

    val_csv = Path(config["val_csv"]).resolve()
    train_csvs = as_path_list(config["train_csv"])
    leaked = [path for path in train_csvs if path == val_csv]
    if leaked:
        raise RuntimeError(
            "Refusing to train on the validation-label CSV. "
            "Use train_csv=data/train.csv for honest model selection. "
            "Set allow_val_label_training=true only for explicitly labeled memorization diagnostics."
        )


def pad_nd_tensors(tensors, padding_value=0):
    max_dim = max(t.dim() for t in tensors)
    normalized = []
    for tensor in tensors:
        while tensor.dim() < max_dim:
            tensor = tensor.unsqueeze(0)
        normalized.append(tensor)

    max_shape = [max(t.shape[dim] for t in normalized) for dim in range(max_dim)]
    padded = []
    for tensor in normalized:
        out = torch.full(max_shape, padding_value, dtype=tensor.dtype)
        slices = tuple(slice(0, size) for size in tensor.shape)
        out[slices] = tensor
        padded.append(out)
    return torch.stack(padded, dim=0)


def collate_multimodal_batch(batch, pad_token_id=0):
    collated = {"id": [item["id"] for item in batch]}

    for key in batch[0].keys():
        if key == "id":
            continue
        values = [item[key] for item in batch]
        first = values[0]

        if key == "input_ids":
            collated[key] = pad_sequence(values, batch_first=True, padding_value=pad_token_id)
        elif key == "attention_mask":
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key == "labels":
            collated[key] = pad_sequence(values, batch_first=True, padding_value=-100)
        elif key == "label":
            collated[key] = torch.tensor([int(v) for v in values], dtype=torch.long)
        elif key == "num_choices":
            collated[key] = torch.tensor([int(v) for v in values], dtype=torch.long)
        elif torch.is_tensor(first):
            collated[key] = pad_nd_tensors(values, padding_value=0)
        else:
            collated[key] = values

    return collated


def move_model_inputs(batch, device, include_labels=False):
    skip = {"id", "label", "num_choices"}
    if not include_labels:
        skip.add("labels")
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in skip and torch.is_tensor(value)
    }


def final_token_logits(outputs, attention_mask):
    last_positions = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
    return outputs.logits[batch_idx, last_positions, :]


def choice_ce_loss_and_accuracy(last_logits, labels, num_choices_list, tokenizer, leading_space=True):
    total_loss = last_logits.new_tensor(0.0)
    correct = 0
    total = 0

    for i, nc in enumerate(num_choices_list):
        choice_ids = get_choice_token_ids(tokenizer, int(nc), leading_space=leading_space)
        choice_logits = last_logits[i, choice_ids]
        label = labels[i]
        total_loss = total_loss + F.cross_entropy(choice_logits.unsqueeze(0), label.unsqueeze(0))
        pred = int(choice_logits.argmax().item())
        correct += int(pred == int(label.item()))
        total += 1

    return total_loss / max(1, total), correct, total


@torch.no_grad()
def evaluate(model, loader, processor, device, dry_run=False, leading_space=True):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="Eval")):
        if dry_run and step > 0:
            break

        model_inputs = move_model_inputs(batch, device, include_labels=False)
        labels = batch["label"].to(device)
        num_choices_list = batch["num_choices"].tolist()

        with autocast_context(device):
            outputs = model(**model_inputs)
            last_logits = final_token_logits(outputs, model_inputs["attention_mask"])
            loss, batch_correct, batch_total = choice_ce_loss_and_accuracy(
                last_logits,
                labels,
                num_choices_list,
                processor.tokenizer,
                leading_space=leading_space,
            )

        total_loss += loss.item()
        correct += batch_correct
        total += batch_total

    return total_loss / max(1, step + 1), correct / max(1, total)


def make_dataset(config, processor, split: str):
    if split == "train":
        csv_path = config["train_csv"]
        image_dir = config.get("image_dir")
        objective = config.get("objective", "letter_ce")
        prompt_variant = config.get("prompt_variant", "default")
    else:
        csv_path = config["val_csv"]
        image_dir = config.get("val_image_dir", str(config.get("image_dir", "")).replace("train", "val"))
        objective = "letter_ce"
        prompt_variant = config.get("prompt_variant_eval", config.get("prompt_variant", "default"))

    return SciQADataset(
        csv_path,
        image_dir,
        processor,
        max_length=config.get("max_length"),
        objective=objective,
        target_format=config.get("target_format", "letter"),
        include_metadata=config.get("include_metadata", True),
        metadata_fields=config.get("metadata_fields"),
        lecture_max_chars=config.get("lecture_max_chars"),
        prompt_variant=prompt_variant,
        include_caption=config.get("include_caption", False),
        caption_max_chars=config.get("caption_max_chars"),
        image_augmentation=config.get("image_augmentation") if split == "train" else None,
        split=split,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step for smoke testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dry_run = args.dry_run or config.get("dry_run", False)
    enforce_no_validation_label_leakage(config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "pixels-to-predictions"),
        entity=wandb_cfg.get("entity"),
        config=config,
        mode=wandb_cfg.get("mode", "disabled" if dry_run else "online"),
        name=config.get("run_name"),
    )

    model, processor = get_model_and_processor(
        model_id=config["model_id"],
        device=device,
        lora_config_dict=config.get("lora"),
        image_mode=config.get("image_mode", "default"),
        adapter_path=config.get("init_adapter_path"),
        adapter_trainable=True,
    )
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    trainable_params = count_trainable_params(model, limit=int(config.get("max_trainable_params", 5_000_000)))

    train_ds = make_dataset(config, processor, "train")
    val_ds = make_dataset(config, processor, "val")

    max_train_samples = config.get("max_train_samples")
    max_val_samples = config.get("max_val_samples")
    if max_train_samples:
        train_ds.df = train_ds.df.iloc[: int(max_train_samples)].reset_index(drop=True)
    if max_val_samples:
        val_ds.df = val_ds.df.iloc[: int(max_val_samples)].reset_index(drop=True)

    pad_token_id = processor.tokenizer.pad_token_id or 0
    collate_fn = lambda batch: collate_multimodal_batch(batch, pad_token_id=pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("eval_batch_size", config["batch_size"]),
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device == "cuda",
    )

    epochs = int(config["epochs"])
    grad_accum = int(config["grad_accum"])
    objective = config.get("objective", "letter_ce")
    leading_space = bool(config.get("leading_space_choices", True))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    total_steps = max(1, (len(train_loader) * epochs + grad_accum - 1) // grad_accum)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_scaler = autocast_dtype(device) == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    output_dir = config.get("output_dir", "runs/current")
    best_dir = os.path.join(output_dir, "best_model")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(
            {
                "config": config,
                "trainable_params": trainable_params,
                "max_trainable_params": int(config.get("max_trainable_params", 5_000_000)),
            },
            f,
            indent=2,
        )
    wandb.summary["trainable_params"] = trainable_params

    best_val_acc = -1.0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            if dry_run and step > 0:
                break

            include_labels = objective == "sft"
            model_inputs = move_model_inputs(batch, device, include_labels=include_labels)

            with autocast_context(device):
                outputs = model(**model_inputs)
                if objective == "sft":
                    loss = outputs.loss
                else:
                    last_logits = final_token_logits(outputs, model_inputs["attention_mask"])
                    loss, batch_correct, batch_total = choice_ce_loss_and_accuracy(
                        last_logits,
                        batch["label"].to(device),
                        batch["num_choices"].tolist(),
                        processor.tokenizer,
                        leading_space=leading_space,
                    )
                    train_correct += batch_correct
                    train_total += batch_total

            if scaler is not None:
                scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()
            total_train_loss += loss.item()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if dry_run and step == 0:
                break

        avg_train_loss = total_train_loss / max(1, step + 1)
        train_acc = train_correct / train_total if train_total else None
        val_loss, val_acc = evaluate(model, val_loader, processor, device, dry_run, leading_space=leading_space)

        metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        if train_acc is not None:
            metrics["train_acc"] = train_acc

        msg = (
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        if train_acc is not None:
            msg += f" | Train Acc: {train_acc:.4f}"
        print(msg)
        wandb.log(metrics)

        min_delta = float(config.get("min_delta", 0.0))
        improved = val_acc > best_val_acc + min_delta

        if not dry_run:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep{epoch + 1}_acc{val_acc:.4f}")
            model.save_pretrained(ckpt_path)
            if improved:
                best_val_acc = val_acc
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        elif improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        stop_val_acc = config.get("stop_val_acc")
        min_epochs = int(config.get("min_epochs", 1))
        if stop_val_acc is not None and epoch + 1 >= min_epochs and val_acc >= float(stop_val_acc):
            print(f"Early stopping: val_acc {val_acc:.4f} >= stop_val_acc {float(stop_val_acc):.4f}")
            break
        patience = config.get("patience")
        if patience is not None and epoch + 1 >= min_epochs and epochs_without_improvement >= int(patience):
            print(f"Early stopping: no val_acc improvement for {epochs_without_improvement} epoch(s)")
            break

    print(f"Best val acc: {best_val_acc:.4f} | best model: {best_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
