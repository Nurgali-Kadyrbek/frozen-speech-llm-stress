"""Cache K_T = mean-pool(Qwen3 embed_tokens(transcript)) for each row in the
WavLM cache. Used as the transcript-only baseline in Stage 1c.

Reads outputs/cache/wavlm_pooled.pt for the row order, computes
embed_tokens(tokenize(transcript)).mean(0) per row, writes to
outputs/cache/kt_pooled.pt with the same row ordering.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.env import setup_env, banner
setup_env()

import torch
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    print(f"transformers=={__import__('transformers').__version__}, torch=={torch.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible."); return 1
    device = "cuda"

    src_path = ROOT / "outputs" / "cache" / "wavlm_pooled.pt"
    if not src_path.exists():
        print(f"ERROR: missing {src_path}; run stage1_cache_wavlm.py first.")
        return 1

    banner(f"Loading WavLM cache + {args.llm}")
    src = torch.load(src_path, weights_only=False, map_location="cpu")
    transcripts = src["transcripts"]
    print(f"  rows: {len(transcripts)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float32).eval().to(device)
    embed = model.get_input_embeddings()
    d_llm = embed.weight.shape[1]
    print(f"  d_llm={d_llm}", flush=True)

    banner("Computing pool(embed_tokens(transcript))")
    N = len(transcripts)
    pooled = torch.zeros(N, 1, d_llm, dtype=torch.float32)
    n_tokens = torch.zeros(N, dtype=torch.long)

    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(transcripts):
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            ids = ids.to(dtype=torch.long)
            if ids.shape[1] == 0:
                # safety: empty transcript shouldn't happen but guard
                pooled[i, 0] = 0
                continue
            e = embed(ids[0])  # (T, d)
            pooled[i, 0] = e.mean(dim=0).cpu()
            n_tokens[i] = ids.shape[1]
            if (i + 1) % 1000 == 0 or i == N - 1:
                print(f"  {i+1}/{N}  elapsed={time.time()-t0:.1f}s", flush=True)

    out_path = ROOT / "outputs" / "cache" / "kt_pooled.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_ids":         src["audio_ids"],
        "sources":           src["sources"],
        "transcription_ids": src["transcription_ids"],
        "interpretation_ids":src["interpretation_ids"],
        "split_origin":      src["split_origin"],
        "transcripts":       src["transcripts"],
        "words":             src["words"],
        "n_words":           src["n_words"],
        "stress_index":      src["stress_index"],
        "options":           src["options"],
        "label":             src["label"],
        "pooled":            pooled,         # (N, 1, d_llm)
        "n_tokens":          n_tokens,
        "model_id":          args.llm,
    }
    torch.save(payload, out_path)
    print(f"\n  saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
