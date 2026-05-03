"""Stage 1a — Probe-G-oracle two-criterion kill-switch on StressPresso.

Per the session prompt, builds the EXACT prompt template (no per-item
variation), scores ' A' / ' B' via mean-log-prob constrained-answer scoring
on Qwen3-1.7B-Instruct, computes accuracy + signed-margin 95% bootstrap CI,
and applies the decision rule.

NB: StressPresso is the held-out evaluation set. It is used here only because
the design (§7.3) defines the oracle on the same task we score Probe-G on,
NOT for any selection — see the session prompt's governing rule.

Implementation note on chat template: Qwen3 ships with a unified
thinking/non-thinking chat template. With `enable_thinking=False` the template
inserts an empty `<think>\\n\\n</think>\\n\\n` block to switch the model into
direct-answer mode. Without it, the assistant turn naturally continues with
`<think>...` content, which would corrupt the constrained-answer log-probs.
We use the tokenizer's apply_chat_template(..., enable_thinking=False) so the
prompt format matches what Qwen3 was trained on for direct answers.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage1a_oracle.py
  (optional) --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner, report_check  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.stress_data import StressPressoItem, load_stresspresso_test  # noqa: E402
from src.probes.scoring import mean_logprob_candidates, tokenize_no_specials  # noqa: E402

SYSTEM = (
    "You are a careful reader. Use the speaker's word emphasis to choose the "
    "correct interpretation."
)

USER_TEMPLATE = (
    "In the transcript, the speaker emphasizes the word '{word}'. Use that "
    "emphasis to interpret the meaning.\n"
    "Transcript: {marked}\n"
    "Which interpretation is correct?\n"
    "A) {option_a}\n"
    "B) {option_b}"
)


def chat_left_text(tok, system: str, user: str) -> str:
    """Use Qwen3's apply_chat_template with enable_thinking=False so the prompt
    matches the model's direct-answer training distribution."""
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score_item(model, embed, tok, item: StressPressoItem, device: str) -> tuple[float, float, int]:
    """Return (score_A, score_B, predicted_label) for one item."""
    user_text = USER_TEMPLATE.format(
        word=item.stressed_word,
        marked=item.transcript_marked,
        option_a=item.options[0],
        option_b=item.options[1],
    )
    left_text = chat_left_text(tok, SYSTEM, user_text)
    delim_text = "Answer:"  # candidate begins with leading space, e.g. ' A'
    cand_texts = [" A", " B"]

    left_ids   = tokenize_no_specials(tok, left_text, device)
    right_ids  = tokenize_no_specials(tok, "", device)            # empty right (no audio slot)
    delim_ids  = tokenize_no_specials(tok, delim_text, device)
    candidates = [tokenize_no_specials(tok, c, device) for c in cand_texts]

    scores = mean_logprob_candidates(
        model, embed,
        left_ids=left_ids, right_ids=right_ids, delim_ids=delim_ids,
        candidates=candidates,
    )
    s_A = float(scores[0].item())
    s_B = float(scores[1].item())
    pred = 0 if s_A > s_B else 1
    return s_A, s_B, pred


def bootstrap_ci(values: np.ndarray, *, n_iter: int = 1000, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the percentile bootstrap of the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def decide(accuracy: float, ci_low: float) -> tuple[str, str]:
    """Apply the decision rule from the session prompt; return (decision, note)."""
    if accuracy >= 0.80 and ci_low > 0:
        return "KEEP_1.7B", "accuracy ≥ 0.80 AND signed-margin CI > 0"
    if accuracy < 0.75 or ci_low <= 0:
        return "SWITCH_8B", "accuracy < 0.75 OR signed-margin CI overlaps 0"
    return "KEEP_1.7B_RECHECK", "0.75 ≤ accuracy < 0.80, CI > 0; flag for re-check on 8B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="HF model id; use Qwen/Qwen3-8B to switch to the larger LLM.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                        help="Model precision. fp32 is safest; bf16/fp16 fits 8B more comfortably.")
    parser.add_argument("--out_dir_name", default=None,
                        help="Sub-dir under outputs/ (defaults to model name slug).")
    args = parser.parse_args()

    fails: list = []
    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        report_check("CUDA available", False, "design requires GPU 6", fails)
        return 1
    device = "cuda"

    banner("Loading StressPresso test (n=202)")
    t0 = time.time()
    items = load_stresspresso_test()
    print(f"  loaded {len(items)} items in {time.time()-t0:.1f}s", flush=True)
    multi_stress = sum(1 for it in items if len(it.transcript_marked.split('[[')) > 2)
    print(f"  items with multi-stress markup (≥ 2 [[): {multi_stress}", flush=True)

    banner(f"Loading {args.model}")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    tdtype = dtype_map[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=tdtype).eval().to(device)
    embed = model.get_input_embeddings()
    print(f"  model in {time.time()-t0:.1f}s, d_llm={embed.weight.shape[1]}, dtype={tdtype}", flush=True)

    banner("Scoring all items")
    t0 = time.time()
    rows: list[dict] = []
    for i, it in enumerate(items):
        s_A, s_B, pred = score_item(model, embed, tok, it, device)
        correct = pred == it.label
        # signed margin: score(correct) − score(incorrect)
        if it.label == 0:
            margin = s_A - s_B
        else:
            margin = s_B - s_A
        rows.append({
            "transcription_id": it.transcription_id,
            "interpretation_id": it.interpretation_id,
            "stressed_word": it.stressed_word,
            "label": it.label,
            "score_A": s_A, "score_B": s_B,
            "pred": pred, "correct": int(correct),
            "signed_margin": margin,
        })
        if (i + 1) % 50 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            running_acc = np.mean([r["correct"] for r in rows])
            print(f"  scored {i+1}/{len(items)}  acc_so_far={running_acc:.3f}  elapsed={elapsed:.1f}s", flush=True)

    accuracies = np.array([r["correct"] for r in rows], dtype=np.float64)
    margins    = np.array([r["signed_margin"] for r in rows], dtype=np.float64)
    acc = float(accuracies.mean())
    margin_mean, margin_lo, margin_hi = bootstrap_ci(margins, n_iter=1000, seed=0, ci=0.95)
    n_correct = int(accuracies.sum())

    decision, note = decide(acc, margin_lo)

    banner("Stage 1a results")
    print(f"  N = {len(rows)}", flush=True)
    print(f"  accuracy        = {acc:.4f}  ({n_correct} / {len(rows)})", flush=True)
    print(f"  signed margin   = {margin_mean:+.4f}  nats  (95% CI: {margin_lo:+.4f}, {margin_hi:+.4f})", flush=True)
    print(f"  decision        = {decision}  ({note})", flush=True)

    slug = (args.out_dir_name
            or args.model.replace("/", "__"))
    out_dir = ROOT / "outputs" / "stage1a" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({
        "model": args.model,
        "dtype": args.dtype,
        "n_items": len(rows),
        "n_correct": n_correct,
        "accuracy": acc,
        "signed_margin_mean": margin_mean,
        "signed_margin_ci_lo": margin_lo,
        "signed_margin_ci_hi": margin_hi,
        "ci_level": 0.95,
        "bootstrap_n_iter": 1000,
        "decision": decision,
        "decision_note": note,
    }, indent=2))
    print(f"\n  rows  → {out_dir / 'rows.jsonl'}", flush=True)
    print(f"  summary → {out_dir / 'summary.json'}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
