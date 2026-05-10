import json
import re
from typing import List, Sequence

import pandas as pd

CHOICE_LABELS = ["A","B","C","D","E"]
DEFAULT_METADATA_FIELDS = ["task", "grade", "subject", "topic", "category", "skill"]


def _normalize_field_list(fields: Sequence[str] | None) -> list[str]:
    if fields is None:
        return DEFAULT_METADATA_FIELDS
    return [str(field).strip() for field in fields if str(field).strip()]


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def is_filled(value) -> bool:
    return pd.notna(value) and str(value).strip() and str(value).strip().lower() != "none"


def parse_choices(choices) -> List[str]:
    return json.loads(choices) if isinstance(choices, str) else list(choices)


def answer_to_letter(answer: int) -> str:
    return CHOICE_LABELS[int(answer)]


def format_choices(choices) -> str:
    return "\n".join(f"{CHOICE_LABELS[i]}. {choice}" for i, choice in enumerate(choices))


def letter_to_answer(text: str, num_choices: int) -> int | None:
    match = re.search(r"\b([A-E])\b", text.upper())
    if not match:
        return None
    idx = CHOICE_LABELS.index(match.group(1))
    return idx if idx < num_choices else None


def build_user_text(
    row: pd.Series,
    include_metadata: bool = True,
    metadata_fields: Sequence[str] | None = None,
    lecture_max_chars: int | None = None,
    prompt_variant: str = "default",
    include_caption: bool = False,
    caption_max_chars: int | None = None,
) -> str:
    metadata_fields = _normalize_field_list(metadata_fields)
    caption = _truncate_text(str(row.get("caption", "")).strip(), caption_max_chars)
    parts: List[str] = []
    if prompt_variant == "no_metadata":
        include_metadata = False
    if prompt_variant in {"exam", "context_first", "question_first", "caption_first", "answer_short", "answer_phrase"}:
        parts.append("Select the single best answer choice. Respond with only the answer letter.")

    def extend_metadata(target_parts: List[str]):
        if not include_metadata:
            return
        for field in metadata_fields:
            if is_filled(row.get(field)):
                target_parts.append(f"{field.capitalize()}: {row[field]}")

    def extend_context(target_parts: List[str]):
        if is_filled(row.get("hint")):
            target_parts.append(f"Hint: {row['hint']}")
        if is_filled(row.get("lecture")):
            lecture = _truncate_text(str(row["lecture"]), lecture_max_chars)
            target_parts.append(f"Context: {lecture}")

    def extend_caption(target_parts: List[str]):
        if include_caption and is_filled(caption):
            target_parts.append(f"Image caption: {caption}")

    if prompt_variant == "question_first":
        parts.append(f"Question: {row['question']}")
        extend_metadata(parts)
        extend_caption(parts)
        extend_context(parts)
    elif prompt_variant == "context_first":
        extend_context(parts)
        extend_caption(parts)
        extend_metadata(parts)
        parts.append(f"Question: {row['question']}")
    elif prompt_variant == "caption_first":
        extend_caption(parts)
        extend_metadata(parts)
        parts.append(f"Question: {row['question']}")
        extend_context(parts)
    else:
        extend_metadata(parts)
        extend_caption(parts)
        parts.append(f"Question: {row['question']}")
        extend_context(parts)
    choices = parse_choices(row["choices"])
    parts.append(f"Choices:\n{format_choices(choices)}")
    if prompt_variant == "answer_short":
        parts.append("Answer:")
    elif prompt_variant == "answer_phrase":
        parts.append("The correct answer is:")
    return "\n".join(parts)


def build_chat_messages(
    row: pd.Series,
    include_metadata: bool = True,
    metadata_fields: Sequence[str] | None = None,
    lecture_max_chars: int | None = None,
    prompt_variant: str = "default",
    include_caption: bool = False,
    caption_max_chars: int | None = None,
) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": build_user_text(
                        row,
                        include_metadata=include_metadata,
                        metadata_fields=metadata_fields,
                        lecture_max_chars=lecture_max_chars,
                        prompt_variant=prompt_variant,
                        include_caption=include_caption,
                        caption_max_chars=caption_max_chars,
                    ),
                },
            ],
        }
    ]


def build_chat_messages_with_assistant(
    row: pd.Series,
    assistant_text: str,
    include_metadata: bool = True,
    metadata_fields: Sequence[str] | None = None,
    lecture_max_chars: int | None = None,
    prompt_variant: str = "default",
    include_caption: bool = False,
    caption_max_chars: int | None = None,
) -> List[dict]:
    return build_chat_messages(
        row,
        include_metadata=include_metadata,
        metadata_fields=metadata_fields,
        lecture_max_chars=lecture_max_chars,
        prompt_variant=prompt_variant,
        include_caption=include_caption,
        caption_max_chars=caption_max_chars,
    ) + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]


def build_chat_messages_with_choice(
    row: pd.Series,
    choice_text: str,
    include_metadata: bool = True,
    metadata_fields: Sequence[str] | None = None,
    lecture_max_chars: int | None = None,
    prompt_variant: str = "default",
    include_caption: bool = False,
    caption_max_chars: int | None = None,
) -> List[dict]:
    return build_chat_messages_with_assistant(
        row,
        choice_text,
        include_metadata=include_metadata,
        metadata_fields=metadata_fields,
        lecture_max_chars=lecture_max_chars,
        prompt_variant=prompt_variant,
        include_caption=include_caption,
        caption_max_chars=caption_max_chars,
    )


def get_choice_text(row: pd.Series, choice_idx: int) -> str:
    choices = parse_choices(row["choices"])
    return choices[choice_idx]


def build_assistant_text(row: pd.Series, target_format: str = "letter") -> str:
    letter = answer_to_letter(int(row["answer"]))
    choice = get_choice_text(row, int(row["answer"]))
    if target_format == "letter":
        return letter
    if target_format == "answer":
        return f"Answer: {letter}"
    if target_format == "choice":
        return f"{choice}\nAnswer: {letter}"
    if target_format == "cot_answer":
        if is_filled(row.get("solution")):
            return f"Reasoning: {str(row['solution']).strip()}\nAnswer: {letter}"
        return f"Answer: {letter}"
    raise ValueError(f"Unknown target_format: {target_format}")


def build_prompt(
    row: pd.Series,
    include_metadata: bool = True,
    lecture_max_chars: int | None = 300,
) -> str:
    parts: List[str] = ["<image>"]
    if include_metadata:
        for field in ["task", "grade", "subject", "topic", "category", "skill"]:
            if is_filled(row.get(field)):
                parts.append(f"{field.capitalize()}: {row[field]}")
    if parts:
        parts.append("")
    parts.append(f"Question: {row['question']}")
    if is_filled(row.get("hint")):
        parts.append(f"Hint: {row['hint']}")
    if is_filled(row.get("lecture")):
        lecture = str(row['lecture'])
        if lecture_max_chars and len(lecture) > lecture_max_chars:
            lecture = lecture[:lecture_max_chars].rstrip() + "..."
        parts.append(f"Context: {lecture}")
    choices = parse_choices(row['choices'])
    parts.append(f"Choices:\n{format_choices(choices)}")
    parts.append("Answer:")
    return "\n".join(parts)


def get_choice_token_ids(tokenizer, num_choices: int, leading_space: bool = True) -> List[int]:
    labels = CHOICE_LABELS[:num_choices]
    if leading_space:
        labels = [f" {label}" for label in labels]
    ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    return ids


def get_choice_token_ids_for_all(tokenizer, max_choices: int = 5, leading_space: bool = True) -> Sequence[int]:
    return get_choice_token_ids(tokenizer, max_choices, leading_space=leading_space)
