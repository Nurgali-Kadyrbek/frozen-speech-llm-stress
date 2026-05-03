"""Stage 0 — Qwen3-ASR feasibility spike (PRE_IMPLEMENTATION_DESIGN §1, §9 Stage 0).

Goal: determine whether Qwen3-ASR encoder hidden states are accessible cleanly
through transformers (with `output_hidden_states=True`), or via forward hooks.

STRICT 1-HOUR TIME-BOX. If blocked, log the failure mode and mark DEFERRED — do
NOT install new packages or change the locked stack.

Outcome categories:
  PASS      — encoder hidden states accessible via transformers (no hooks needed).
  PASS-HOOK — hidden states accessible only via forward hooks (still usable).
  DEFER     — neither path works within 1h, OR a model/package is unavailable.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage0_qwen3_asr_spike.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner, report_check  # noqa: E402

setup_env()

import importlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CANDIDATE_MODEL_IDS = [
    # In rough order of preference; first match wins.
    "Qwen/Qwen3-ASR-Flash",
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
]
TIME_BUDGET_SEC = 60 * 60  # 1 hour


def try_load_via_transformers(model_id: str, device: str):
    """Try AutoModel.from_pretrained with trust_remote_code; return (model, processor) or None."""
    from transformers import AutoModel, AutoProcessor, AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  config loaded: model_type={cfg.model_type}, "
              f"architectures={getattr(cfg, 'architectures', None)}", flush=True)
    except Exception as exc:
        print(f"  config load FAILED for {model_id}: {type(exc).__name__}: {exc}", flush=True)
        return None

    try:
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).eval().to(device)
    except Exception as exc:
        print(f"  AutoModel load FAILED for {model_id}: {type(exc).__name__}: {exc}", flush=True)
        return None

    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        print(f"  AutoProcessor unavailable for {model_id}: {type(exc).__name__}: {exc}", flush=True)
        proc = None

    return model, proc


def find_audio_encoder(model):
    """Heuristic search for the audio encoder submodule by name."""
    for name, mod in model.named_modules():
        n_low = name.lower()
        # Look for typical names; prefer the top-level encoder, not nested blocks.
        if n_low in {"audio_encoder", "encoder", "audio_tower", "audio_model.encoder"}:
            return name, mod
    # Fallback: pick the encoder of the audio sub-tree if any
    for name, mod in model.named_modules():
        if "audio" in name.lower() and name.lower().endswith("encoder"):
            return name, mod
    return None, None


def main() -> int:
    fails: list = []
    t_start = time.time()

    print("Stage 0 — Qwen3-ASR feasibility spike", flush=True)
    print(f"transformers=={__import__('transformers').__version__}", flush=True)
    print(f"torch=={torch.__version__}, "
          f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: no CUDA visible; spike will run on CPU.", flush=True)

    # Probe whether the qwen-asr pip package is present (do NOT pip install).
    qwen_asr_pkg_present = importlib.util.find_spec("qwen_asr") is not None
    print(f"  qwen-asr pip package available: {qwen_asr_pkg_present}", flush=True)

    # ----------------------------------------------------------------------
    banner("Path A — AutoModel.from_pretrained(..., trust_remote_code=True)")
    # ----------------------------------------------------------------------
    model = proc = None
    model_id_used = None
    for mid in CANDIDATE_MODEL_IDS:
        if time.time() - t_start > TIME_BUDGET_SEC:
            print("  TIME BUDGET EXHAUSTED — aborting Path A.", flush=True)
            break
        print(f"  trying {mid} ...", flush=True)
        loaded = try_load_via_transformers(mid, device)
        if loaded is not None:
            model, proc = loaded
            model_id_used = mid
            print(f"  loaded: {mid}", flush=True)
            break

    if model is None:
        print("  No Qwen3-ASR variant loadable via transformers within budget.", flush=True)
        banner("Spike outcome: DEFER (Qwen3-ASR not accessible via transformers in 1h)")
        return 2  # 2 == DEFER

    # ----------------------------------------------------------------------
    banner("Path B — try output_hidden_states=True on audio encoder")
    # ----------------------------------------------------------------------
    enc_name, enc_mod = find_audio_encoder(model)
    print(f"  candidate audio encoder submodule: name={enc_name!r}, "
          f"type={type(enc_mod).__name__ if enc_mod else None}", flush=True)
    if enc_mod is None:
        print("  could not locate an obvious audio encoder; listing top-level modules:", flush=True)
        for name, mod in model.named_children():
            print(f"    - {name}: {type(mod).__name__}", flush=True)
        banner("Spike outcome: DEFER (audio encoder submodule not located)")
        return 2

    # Build a tiny dummy input to feed the encoder.
    audio = np.random.randn(16000 * 4).astype(np.float32) * 0.02
    inputs = None
    if proc is not None:
        try:
            inputs = proc(audios=[audio], sampling_rate=16000, return_tensors="pt")
        except TypeError:
            try:
                inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
            except Exception as exc:
                print(f"  processor call FAILED: {type(exc).__name__}: {exc}", flush=True)

    if inputs is None:
        # No processor — try to build features manually only if shape obvious; else defer.
        print("  no processor available — cannot prepare input features. Marking DEFER.", flush=True)
        banner("Spike outcome: DEFER (no processor / unable to prepare input features)")
        return 2

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    hs_via_kwarg = None
    try:
        with torch.no_grad():
            out = enc_mod(**inputs, output_hidden_states=True)
        hs_via_kwarg = getattr(out, "hidden_states", None)
        print(f"  encoder forward(output_hidden_states=True) → "
              f"hidden_states is {None if hs_via_kwarg is None else f'tuple of length {len(hs_via_kwarg)}'}",
              flush=True)
    except Exception as exc:
        print(f"  encoder forward(output_hidden_states=True) FAILED: "
              f"{type(exc).__name__}: {exc}", flush=True)

    if hs_via_kwarg is not None and len(hs_via_kwarg) > 1:
        print("  PASS — encoder exposes hidden_states via output_hidden_states=True kwarg.", flush=True)
        for i, h in enumerate(hs_via_kwarg):
            print(f"    layer {i}: shape={tuple(h.shape)}", flush=True)
        banner(f"Spike outcome: PASS (model={model_id_used})")
        return 0

    # ----------------------------------------------------------------------
    banner("Path C — fall back to forward hooks on encoder blocks")
    # ----------------------------------------------------------------------
    captured = []

    def make_hook(idx):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                captured.append((idx, tuple(t.shape)))
        return _hook

    blocks = []
    # Common patterns: encoder.layers, encoder.blocks, encoder.h
    for attr in ("layers", "blocks", "h"):
        if hasattr(enc_mod, attr):
            mods = getattr(enc_mod, attr)
            if hasattr(mods, "__iter__"):
                blocks = list(mods)
                print(f"  hooking {len(blocks)} blocks at .{attr}", flush=True)
                break
    if not blocks:
        print("  could not locate encoder block list; cannot hook.", flush=True)
        banner("Spike outcome: DEFER (no block list to hook)")
        return 2

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            _ = enc_mod(**inputs)
    finally:
        for h in handles:
            h.remove()

    if captured:
        print(f"  captured {len(captured)} block outputs via hooks:", flush=True)
        for i, shape in captured[:4]:
            print(f"    block {i}: shape={shape}", flush=True)
        if len(captured) > 4:
            print(f"    ... ({len(captured) - 4} more)", flush=True)
        banner(f"Spike outcome: PASS-HOOK (model={model_id_used})")
        return 0

    banner("Spike outcome: DEFER (neither kwarg nor hooks captured hidden states)")
    return 2


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\nINTERRUPTED — marking spike DEFERRED.", flush=True)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        print("\nUNHANDLED EXCEPTION — marking spike DEFERRED.", flush=True)
        sys.exit(2)
