import re

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, PeftModel, get_peft_model


def _can_use_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def _parse_nosplit_edge(image_mode: str) -> int | None:
    if image_mode == "nosplit":
        return 512
    match = re.fullmatch(r"nosplit(\d+)", image_mode)
    if match:
        return int(match.group(1))
    return None


def configure_image_processor(processor, image_mode: str = "default"):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return processor

    if image_mode == "default":
        return processor
    if image_mode == "split":
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = True
        return processor

    edge = _parse_nosplit_edge(image_mode)
    if edge is not None:
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False
        if hasattr(image_processor, "size"):
            image_processor.size = {"longest_edge": edge}
        if hasattr(image_processor, "max_image_size"):
            image_processor.max_image_size = {"longest_edge": edge}
        return processor
    raise ValueError(f"Unknown image_mode: {image_mode}")


def get_model_and_processor(
    model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
    device="cuda",
    lora_config_dict=None,
    image_mode="default",
    adapter_path=None,
    adapter_trainable=True,
):
    processor = AutoProcessor.from_pretrained(model_id)
    configure_image_processor(processor, image_mode=image_mode)

    use_cuda = device == "cuda" and _can_use_cuda()
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is not available in the current runtime.")
    model_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32 if not use_cuda else torch.float16

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=model_dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load model on the requested device '{device}'.") from exc
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=adapter_trainable)
        return model, processor

    if lora_config_dict is None:
        return model, processor

    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 4),
        lora_alpha=lora_config_dict.get("lora_alpha", 16),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
        bias="none",
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        rank_pattern=lora_config_dict.get("rank_pattern", {}),
        alpha_pattern=lora_config_dict.get("alpha_pattern", {}),
        use_rslora=lora_config_dict.get("use_rslora", False),
        use_dora=lora_config_dict.get("use_dora", False),
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    return model, processor

def count_trainable_params(model, limit=5_000_000):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,} ({total/1e6:.2f}M)")
    if total > limit:
        raise ValueError(f"HARD LIMIT EXCEEDED: {total} > {limit} trainable parameters.")
    return total
