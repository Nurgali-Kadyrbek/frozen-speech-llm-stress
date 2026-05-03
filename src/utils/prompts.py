"""Chat-template helpers — single source of truth for the audio-slot layout.

Builds the (left_text, right_text, response_text) triple around an
`{{AUDIO_SLOT}}` placeholder so the audio slot inserts cleanly in embed
space. Always uses Qwen3 `apply_chat_template(..., enable_thinking=False)`
per kickoff rule R3.
"""
from __future__ import annotations

from dataclasses import dataclass

AUDIO_PLACEHOLDER = "{{AUDIO_SLOT}}"

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Listen to the audio and respond appropriately."
)


@dataclass(frozen=True)
class PromptHalves:
    left_text: str
    right_text: str
    response_text: str


def build_training_halves(tok, *, system: str, question: str, response: str,
                          enable_thinking: bool = False) -> PromptHalves:
    """Render the chat template, split at the audio placeholder.

    The user content is `{AUDIO_PLACEHOLDER}\\n{question}` so the placeholder
    sits before the question text. After splitting, `right_text` contains
    `\\n{question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
    (the empty think block appears with `enable_thinking=False`).
    """
    user_content = f"{AUDIO_PLACEHOLDER}\n{question}"
    full = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    parts = full.split(AUDIO_PLACEHOLDER, 1)
    if len(parts) != 2:
        raise ValueError(
            "audio placeholder not preserved in chat template output; "
            "tokenizer may be normalizing the placeholder string"
        )
    return PromptHalves(left_text=parts[0], right_text=parts[1], response_text=response)


def build_probe_g_halves(tok, *, system: str, question: str,
                          option_a: str, option_b: str,
                          enable_thinking: bool = False) -> PromptHalves:
    """Probe-G prompt halves. The candidates ' A' / ' B' are scored separately
    by the caller; here we just produce left / right text around the audio
    slot, with the assistant turn opened via `add_generation_prompt=True`
    + an explicit 'Answer:' delimiter so the next token is the option letter.
    """
    user_content = (
        f"{AUDIO_PLACEHOLDER}\n{question}\n"
        f"A) {option_a}\n"
        f"B) {option_b}"
    )
    full = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    parts = full.split(AUDIO_PLACEHOLDER, 1)
    if len(parts) != 2:
        raise ValueError("audio placeholder lost in chat template")
    # `Answer:` delimiter goes at the very end so the next token is the option letter.
    right_with_delim = parts[1] + "Answer:"
    return PromptHalves(left_text=parts[0], right_text=right_with_delim, response_text="")
