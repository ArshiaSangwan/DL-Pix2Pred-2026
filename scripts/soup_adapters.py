#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def adapter_file(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {path}")
    return path


def check_same_config(adapter_dirs: list[Path]) -> dict:
    first_config = adapter_dirs[0] / "adapter_config.json"
    if not first_config.exists():
        raise FileNotFoundError(f"Missing adapter config: {first_config}")
    base_config = json.loads(first_config.read_text())

    comparable_keys = [
        "base_model_name_or_path",
        "peft_type",
        "r",
        "lora_alpha",
        "target_modules",
        "rank_pattern",
        "alpha_pattern",
        "use_dora",
        "use_rslora",
    ]
    def comparable(config: dict) -> dict:
        values = {key: config.get(key) for key in comparable_keys}
        if isinstance(values.get("target_modules"), list):
            values["target_modules"] = sorted(values["target_modules"])
        return values

    reference = comparable(base_config)
    for adapter_dir in adapter_dirs[1:]:
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing adapter config: {config_path}")
        config = json.loads(config_path.read_text())
        current = comparable(config)
        if current != reference:
            raise ValueError(
                "Adapter configs are not soup-compatible:\n"
                f"reference={adapter_dirs[0]}\n"
                f"mismatch={adapter_dir}"
            )
    return base_config


def average_adapters(adapter_dirs: list[Path], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    if weights is None:
        weights = [1.0] * len(adapter_dirs)
    if len(weights) != len(adapter_dirs):
        raise ValueError(f"Expected {len(adapter_dirs)} weights, got {len(weights)}")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Adapter soup weights must sum to a positive value")

    accumulator = None
    reference_shapes = None
    for adapter_dir, weight in zip(adapter_dirs, weights):
        tensors = load_file(str(adapter_file(adapter_dir)), device="cpu")
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        if reference_shapes is None:
            reference_shapes = shapes
            accumulator = {
                key: value.to(dtype=torch.float32) * (float(weight) / total_weight)
                for key, value in tensors.items()
            }
            continue
        if shapes != reference_shapes:
            missing = sorted(set(reference_shapes) ^ set(shapes))
            changed = sorted(key for key in reference_shapes.keys() & shapes.keys() if reference_shapes[key] != shapes[key])
            raise ValueError(
                f"Adapter tensor mismatch in {adapter_dir}; missing/different keys={missing[:5]}, changed={changed[:5]}"
            )
        for key, value in tensors.items():
            accumulator[key] += value.to(dtype=torch.float32) * (float(weight) / total_weight)

    return accumulator or {}


def copy_metadata(source_dir: Path, out_dir: Path):
    for filename in ["adapter_config.json", "README.md"]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)
    for filename in ["tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Average same-shape PEFT LoRA adapters into a model soup.")
    parser.add_argument("--adapters", nargs="+", required=True, help="Adapter directories to average.")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="Optional positive soup weights.")
    parser.add_argument("--out_dir", required=True, help="Output adapter directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output adapter file.")
    args = parser.parse_args()

    adapter_dirs = [Path(path) for path in args.adapters]
    for adapter_dir in adapter_dirs:
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)
    check_same_config(adapter_dirs)

    out_dir = Path(args.out_dir)
    out_file = out_dir / "adapter_model.safetensors"
    if out_file.exists() and not args.force:
        raise FileExistsError(f"{out_file} already exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    averaged = average_adapters(adapter_dirs, args.weights)
    save_file(averaged, str(out_file))
    copy_metadata(adapter_dirs[0], out_dir)
    manifest = {
        "adapters": [str(path) for path in adapter_dirs],
        "weights": args.weights or [1.0] * len(adapter_dirs),
        "num_tensors": len(averaged),
    }
    (out_dir / "soup_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote adapter soup to {out_dir} from {len(adapter_dirs)} checkpoints")


if __name__ == "__main__":
    main()
