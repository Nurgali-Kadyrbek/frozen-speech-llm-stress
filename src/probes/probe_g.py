"""Probe-G — pair-level signed correctness margin on StressPresso (n=202).

`P_probe` = 10 neutral paraphrases × 2 stress-cue conditions {neutral,
explicit} = 20 prompts per item. For each item we score candidates
' A' / ' B' under each prompt and aggregate signed margin = score(correct
letter) − score(other). The kickoff calls for bootstrap CI by
transcription_id (clusters of items sharing T) at 1000 iters, 95 %.

Used by the Stage-2 evaluator with four "audio-slot" variants:
  - K_speech (adapter output, run live)
  - K_text   = embed_tokens(true_transcript)        [K_T baseline]
  - K_text_predicted = embed_tokens(Whisper ASR)    [Cascade-T]
  - K_oracle = embed_tokens(transcript with [[X]] stress markup)
              + question prefixed with "In the transcript, the speaker
              emphasizes the word 'X'." [Oracle re-confirm]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


SYSTEM_PROBE_G = (
    "You are a careful reader. Use the speaker's word emphasis to choose "
    "the correct interpretation."
)

# 10 NEUTRAL paraphrases — no explicit stress cue. The audio slot is the
# only carrier of stress information for the adapter; cascade-T / K_T sees
# nothing prosodic; oracle gets stress via [[…]] markup placed by the caller.
NEUTRAL_PARAPHRASES = [
    "What does the speaker most likely mean?",
    "What is the underlying intention behind this utterance?",
    "Choose the correct interpretation.",
    "Which interpretation best matches the speaker's intent?",
    "Pick the meaning the speaker is conveying.",
    "What is the speaker really saying here?",
    "Identify the most likely intended meaning.",
    "Which option captures the implied meaning?",
    "Which interpretation is most accurate?",
    "What is the speaker emphasizing in this utterance?",
]

# 10 EXPLICIT-CUE paraphrases — the cue is in the question text itself.
# `{word}` will be substituted with the stressed word.
EXPLICIT_PARAPHRASES = [
    "The speaker emphasizes the word '{word}'. Which interpretation is correct?",
    "Given that the stressed word is '{word}', what does the speaker mean?",
    "The word '{word}' is emphasized. Choose the matching interpretation.",
    "Stress is on the word '{word}'. Which option is correct?",
    "Knowing that '{word}' is stressed, pick the correct meaning.",
    "The emphasis falls on the word '{word}'. What is the implied meaning?",
    "The word '{word}' is highlighted by stress. Which is the correct reading?",
    "Given the emphasis on the word '{word}', identify the correct interpretation.",
    "If '{word}' is the stressed word, which option matches?",
    "The speaker stresses the word '{word}'. Choose the right interpretation.",
]


@dataclass
class ProbeGResult:
    n_items: int
    accuracy: float
    accuracy_neutral: float
    accuracy_explicit: float
    signed_margin_mean: float
    signed_margin_neutral_mean: float
    signed_margin_explicit_mean: float
    ci_low: float
    ci_high: float
    ci_low_neutral: float
    ci_high_neutral: float
    ci_low_explicit: float
    ci_high_explicit: float
    rows: list


def bootstrap_by_cluster(
    values: np.ndarray, cluster_ids: list,
    n_iter: int = 1000, ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap mean(values) by resampling clusters."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(cluster_ids)
    unique = np.unique(cl)
    by_cluster = {c: np.where(cl == c)[0] for c in unique}
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        sel = rng.choice(unique, size=unique.shape[0], replace=True)
        idx = np.concatenate([by_cluster[c] for c in sel])
        means[i] = values[idx].mean()
    a = (1.0 - ci) / 2.0
    return float(values.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def build_user_text(audio_marker: str, paraphrase: str, *, word: str | None,
                    option_a: str, option_b: str,
                    transcript_for_oracle: str | None = None) -> str:
    """Render the user-turn content WITH a literal `audio_marker` standing in
    for the audio slot. Caller substitutes the marker with the actual
    embedding stream.
    """
    pieces = []
    if transcript_for_oracle is not None:
        # Oracle re-confirm: stress info is in the transcript markup, not
        # injected in the audio slot. The audio slot is empty here.
        pieces.append(transcript_for_oracle)
    else:
        pieces.append(audio_marker)

    q = paraphrase.format(word=word) if "{word}" in paraphrase and word else paraphrase
    pieces.append(q)
    pieces.append(f"A) {option_a}")
    pieces.append(f"B) {option_b}")
    return "\n".join(pieces)
