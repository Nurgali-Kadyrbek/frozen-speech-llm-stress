"""Stage 3.6 / B1 pre-flight gates.

Two checks run BEFORE Stage 3.6 training. If either fails (>0.30), revert
to Stage-3 abstract-template L_NCE (and accept the term stays dead).

  Gate 1 — T-only contamination: train a fresh C_φ_T-only on
           restricted_T → Φ for many epochs (independent of any adapter
           training). If probe-eval accuracy ≥ 0.30 → lexical features
           solve the task without audio. Stage 3 measured this jointly-
           trained at 0.186 (under-fit in 400 steps); re-verify here
           with fully-converged probe.

  Gate 2 — Description-only probe: for each Stress17kItem, compute
           description = "emphasizes the word '{words[stress_index]}'",
           pool(embed(description tokens)) ∈ R^{d_llm}. Train linear
           probe on probe-train, eval on probe-eval. If accuracy ≥ 0.30
           → the description text itself carries Φ info even in d_llm
           space — word-substitution L_NCE is uninformative because
           the contrastive direction is solvable from text alone.

Run:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/stage3p6_b1_gates.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import setup_env, banner  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from src.losses.r1 import CPhi, restricted_t_features  # noqa: E402
from src.data.stress_data import (  # noqa: E402
    load_stress17k, partition_transcript_ids,
)


QWEN3_MODEL = "Qwen/Qwen3-8B"
N_MAX_CLASSES = 13


def gate_t_only(s17_train, s17_eval, *, device: str,
                epochs: int = 80, lr: float = 1e-3, batch_size: int = 256,
                seed: int = 0) -> dict:
    """Train C_phi_t_only on restricted_T(transcript) → Φ; eval on held-out."""
    banner("Gate 1 — T-only contamination (fully-converged C_phi_t_only)")

    def _features_and_labels(items):
        feats = np.stack([
            restricted_t_features(it.transcription, n_words=it.n_words)
            for it in items
        ])
        labels = np.asarray([it.stress_index for it in items], dtype=np.int64)
        labels = np.clip(labels, 0, N_MAX_CLASSES - 1)
        return feats, labels

    X_tr_np, y_tr_np = _features_and_labels(s17_train)
    X_ev_np, y_ev_np = _features_and_labels(s17_eval)
    print(f"  train: {X_tr_np.shape},  eval: {X_ev_np.shape}", flush=True)

    head = CPhi(d_in=X_tr_np.shape[1], hidden=256, n_classes=N_MAX_CLASSES).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    torch.manual_seed(seed)
    X_tr = torch.from_numpy(X_tr_np).to(device=device, dtype=torch.float32)
    y_tr = torch.from_numpy(y_tr_np).to(device=device, dtype=torch.long)
    X_ev = torch.from_numpy(X_ev_np).to(device=device, dtype=torch.float32)
    y_ev = torch.from_numpy(y_ev_np).to(device=device, dtype=torch.long)

    n = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    best_eval_acc = 0.0
    final_eval_acc = 0.0
    for ep in range(epochs):
        head.train()
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = torch.tensor(idx[s:s+batch_size], device=device, dtype=torch.long)
            xb = X_tr.index_select(0, b)
            yb = y_tr.index_select(0, b)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(X_ev).argmax(dim=-1)
            acc = float((pred == y_ev).float().mean().item())
        final_eval_acc = acc
        if acc > best_eval_acc:
            best_eval_acc = acc
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  "
                  f"best_so_far={best_eval_acc:.4f}", flush=True)

    chance = 1.0 / N_MAX_CLASSES
    res = {
        "n_train":    int(n),
        "n_eval":     int(X_ev.shape[0]),
        "epochs":     epochs,
        "chance":     chance,
        "final_eval_acc": final_eval_acc,
        "best_eval_acc":  best_eval_acc,
        "FAIL":       bool(best_eval_acc >= 0.30),
    }
    print(f"  → final_eval_acc={final_eval_acc:.4f}  best={best_eval_acc:.4f}  "
          f"chance={chance:.4f}  FAIL(>=0.30)={res['FAIL']}", flush=True)
    return res


def gate_description_only(s17_train, s17_eval, *, embed_layer, tokenizer,
                          device: str, seed: int = 0) -> dict:
    """Build word-sub descriptions; pool(embed(desc)) → linear probe → Φ."""
    banner("Gate 2 — Description-only probe (word-substitution descriptions)")

    template = "emphasizes the word '{word}'"

    def _description_for(it):
        if it.stress_index < len(it.words):
            w = it.words[it.stress_index]
        else:
            w = ""
        return template.format(word=w)

    @torch.no_grad()
    def _pool_features(items):
        feats = []
        for it in items:
            text = _description_for(it)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = ids.to(dtype=torch.long, device=device)
            emb = embed_layer(ids).float().mean(dim=0).cpu().numpy()
            feats.append(emb)
        return np.stack(feats)

    print(f"  computing pool(embed(desc)) on {len(s17_train)} train items…", flush=True)
    t0 = time.time()
    X_tr = _pool_features(s17_train)
    print(f"    done in {time.time()-t0:.1f}s ({len(s17_train)/(time.time()-t0):.1f} samples/s)", flush=True)
    print(f"  computing pool(embed(desc)) on {len(s17_eval)} eval items…", flush=True)
    t0 = time.time()
    X_ev = _pool_features(s17_eval)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    y_tr = np.asarray([it.stress_index for it in s17_train], dtype=np.int64)
    y_ev = np.asarray([it.stress_index for it in s17_eval], dtype=np.int64)
    y_tr = np.clip(y_tr, 0, N_MAX_CLASSES - 1)
    y_ev = np.clip(y_ev, 0, N_MAX_CLASSES - 1)

    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict(X_tr)
    p_ev = clf.predict(X_ev)
    a_tr = float(accuracy_score(y_tr, p_tr))
    a_ev = float(accuracy_score(y_ev, p_ev))
    chance = 1.0 / N_MAX_CLASSES
    res = {
        "template":   template,
        "n_train":    int(X_tr.shape[0]),
        "n_eval":     int(X_ev.shape[0]),
        "feat_dim":   int(X_tr.shape[1]),
        "chance":     chance,
        "train_acc":  a_tr,
        "eval_acc":   a_ev,
        "FAIL":       bool(a_ev >= 0.30),
    }
    print(f"  → train_acc={a_tr:.4f}  eval_acc={a_ev:.4f}  "
          f"chance={chance:.4f}  FAIL(>=0.30)={res['FAIL']}", flush=True)
    return res


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA visible.", flush=True); return 1
    device = "cuda"

    # ---- Data: Stress-17K probe-train / probe-eval ---- #
    banner("Loading Stress-17K-raw and partitioning by transcription_id")
    s17_all = load_stress17k()
    train_ids, eval_ids = partition_transcript_ids(s17_all, eval_frac=0.20, seed="BTA-2026-05-02")
    s17_train = [it for it in s17_all if it.transcription_id in train_ids]
    s17_eval  = [it for it in s17_all if it.transcription_id in eval_ids]
    print(f"  train: {len(s17_train)} rows  /  eval: {len(s17_eval)} rows", flush=True)

    # ---- Gate 1: T-only ---- #
    res_t_only = gate_t_only(s17_train, s17_eval, device=device, epochs=80)

    # ---- Gate 2: description-only (multi-format comparison) ---- #
    banner(f"Loading Qwen3-8B embedding layer for gate 2…")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL, torch_dtype=torch.bfloat16).eval().to(device)
    embed_layer = llm.get_input_embeddings()

    # Run description-only gate for word-substitution AND abstract templates.
    # Spec: word-substitution gate result drives the B1 decision. Abstract
    # template result is a baseline reference for what a "passed" gate looks
    # like under this design.
    res_desc = gate_description_only(s17_train, s17_eval,
                                       embed_layer=embed_layer, tokenizer=tok, device=device)
    # Also run abstract baseline by patching the template inside _description_for.
    banner("Gate 2-baseline — abstract-template description-only probe")
    template_abstract = "emphasis position: {idx}"

    @torch.no_grad()
    def _abstract_pool_features(items):
        feats = []
        for it in items:
            text = template_abstract.format(idx=int(it.stress_index))
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = ids.to(dtype=torch.long, device=device)
            emb = embed_layer(ids).float().mean(dim=0).cpu().numpy()
            feats.append(emb)
        return np.stack(feats)

    Xtr_abs = _abstract_pool_features(s17_train)
    Xev_abs = _abstract_pool_features(s17_eval)
    ytr_abs = np.clip(np.asarray([it.stress_index for it in s17_train], dtype=np.int64), 0, N_MAX_CLASSES - 1)
    yev_abs = np.clip(np.asarray([it.stress_index for it in s17_eval], dtype=np.int64), 0, N_MAX_CLASSES - 1)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
    clf.fit(Xtr_abs, ytr_abs)
    a_tr_abs = float(accuracy_score(ytr_abs, clf.predict(Xtr_abs)))
    a_ev_abs = float(accuracy_score(yev_abs, clf.predict(Xev_abs)))
    res_desc_abstract = {
        "template":  template_abstract,
        "n_train":   int(Xtr_abs.shape[0]), "n_eval": int(Xev_abs.shape[0]),
        "feat_dim":  int(Xtr_abs.shape[1]), "chance": 1.0/N_MAX_CLASSES,
        "train_acc": a_tr_abs, "eval_acc": a_ev_abs,
        "FAIL":      bool(a_ev_abs >= 0.30),
    }
    print(f"  abstract template description-only: train_acc={a_tr_abs:.4f} eval_acc={a_ev_abs:.4f} "
          f"FAIL(>=0.30)={res_desc_abstract['FAIL']}", flush=True)

    # ---- Decision ---- #
    banner("B1 pre-flight gate decision")
    payload = {
        "gate_t_only":            res_t_only,
        "gate_description_only_word_sub":  res_desc,
        "gate_description_only_abstract":  res_desc_abstract,
        "word_sub_safe":          bool(not res_desc["FAIL"]),
        "abstract_safe":          bool(not res_desc_abstract["FAIL"]),
        "t_only_safe":            bool(not res_t_only["FAIL"]),
    }
    payload["any_fail"] = bool(res_t_only["FAIL"] or res_desc["FAIL"])
    if payload["any_fail"]:
        print("  ❌ AT LEAST ONE GATE FAILED — revert L_NCE to abstract templates.", flush=True)
        print(f"     T-only fail: {res_t_only['FAIL']}; description-only fail: {res_desc['FAIL']}", flush=True)
    else:
        print("  ✓ BOTH GATES PASS — proceed to Stage 3.6 training with word-substitution L_NCE.", flush=True)

    out_path = ROOT / "outputs" / "stage3p6" / "b1_gate_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved → {out_path}", flush=True)
    return 0 if not payload["any_fail"] else 3


if __name__ == "__main__":
    sys.exit(main())
