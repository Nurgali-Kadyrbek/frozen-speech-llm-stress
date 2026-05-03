"""Three shortcut probes (kickoff §3.4.C / design §8.3 (c)):

  voice_id_probe : predict TTS voice (nova vs echo) from pool(H) on
                   Stress-17K subset only — tests whether augmentation
                   eliminates the TTS voice fingerprint.
  tts_vs_real    : 4-class predict source corpus from pool(H)
                   {stress17k, stresspresso, librispeech, expresso}.
  domain_probe   : binary {synthetic = stress17k, real = others}.

Each is a single-layer linear classifier (sklearn LogisticRegression with
strong regularization) trained on 80 % sample-level split, evaluated on
20 % held-out. The 70 % threshold (kickoff §3.0.5) flags augmentation
failure if any of the three exceeds it on held-out data — meaning the
augmentation pipeline is not anonymizing the targeted shortcut.

Pre-training (Stage 3.0.5): probe features = pool(WavLM_L16(augmented(x))).
Post-training (Stage 3.4.C): probe features = pool(A_R1(H(augmented(x))))).
Same probe code, different feature source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class ProbeResult:
    name:        str
    n_classes:   int
    n_train:     int
    n_eval:      int
    chance:      float
    train_acc:   float
    eval_acc:    float
    fail_70pct:  bool   # True if eval_acc >= 0.70 (kickoff threshold)


def _split_train_eval(
    X: np.ndarray, y: np.ndarray, *, eval_frac: float = 0.20, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * (1.0 - eval_frac)))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def fit_linear_probe(
    X: np.ndarray, y: np.ndarray, *,
    name: str,
    eval_frac: float = 0.20, seed: int = 0,
    C: float = 1.0, max_iter: int = 1000,
) -> ProbeResult:
    """Standard linear probe with sklearn. Reports held-out eval accuracy."""
    classes = np.unique(y)
    n_classes = len(classes)
    chance = 1.0 / max(n_classes, 1)

    X_tr, y_tr, X_ev, y_ev = _split_train_eval(X, y, eval_frac=eval_frac, seed=seed)
    if len(X_tr) == 0 or len(X_ev) == 0 or n_classes < 2:
        return ProbeResult(
            name=name, n_classes=n_classes, n_train=len(X_tr), n_eval=len(X_ev),
            chance=chance, train_acc=float("nan"), eval_acc=float("nan"),
            fail_70pct=False,
        )

    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=max_iter,
        multi_class="auto", random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict(X_tr)
    p_ev = clf.predict(X_ev)
    a_tr = accuracy_score(y_tr, p_tr)
    a_ev = accuracy_score(y_ev, p_ev)
    return ProbeResult(
        name=name, n_classes=n_classes, n_train=len(X_tr), n_eval=len(X_ev),
        chance=chance, train_acc=float(a_tr), eval_acc=float(a_ev),
        fail_70pct=bool(a_ev >= 0.70),
    )


def run_three_shortcut_probes(
    *,
    features: np.ndarray,                # (N, d) pool(H) per sample
    source: list[str],                   # 'stress17k' | 'stresspresso' | 'librispeech' | 'expresso'
    voice_or_speaker: list[str],         # per-sample voice/speaker id
    seed: int = 0,
) -> dict[str, ProbeResult]:
    """Run all three probes; return {'voice_id', 'tts_vs_real', 'domain'} → ProbeResult.

    `voice_id` runs ONLY on the Stress-17K subset (nova vs echo); excluding
    other corpora keeps the task well-conditioned with a clean 2-class signal
    on the TTS fingerprint that codec randomization is meant to anonymize.
    """
    src = np.asarray(source)
    feat = np.asarray(features)
    voice = np.asarray(voice_or_speaker)

    # ---- voice_id (Stress-17K only, 2-class nova vs echo) ---- #
    is_stress = (src == "stress17k")
    if is_stress.sum() >= 50:
        # Only nova / echo — drop others if dataset has more.
        valid_voices = {"nova", "echo"}
        keep = is_stress & np.isin(voice, list(valid_voices))
        if keep.sum() >= 50 and len(np.unique(voice[keep])) >= 2:
            X = feat[keep]
            y_str = voice[keep]
            classes_sorted = sorted(np.unique(y_str))
            cls_to_int = {c: i for i, c in enumerate(classes_sorted)}
            y = np.array([cls_to_int[s] for s in y_str], dtype=np.int64)
            voice_id_res = fit_linear_probe(X, y, name="voice_id", seed=seed)
        else:
            voice_id_res = ProbeResult(
                name="voice_id", n_classes=0, n_train=0, n_eval=0,
                chance=float("nan"), train_acc=float("nan"), eval_acc=float("nan"),
                fail_70pct=False,
            )
    else:
        voice_id_res = ProbeResult(
            name="voice_id", n_classes=0, n_train=0, n_eval=0,
            chance=float("nan"), train_acc=float("nan"), eval_acc=float("nan"),
            fail_70pct=False,
        )

    # ---- tts_vs_real (4-class, all sources) ---- #
    src_classes = ["stress17k", "stresspresso", "librispeech", "expresso"]
    src_to_int = {c: i for i, c in enumerate(src_classes)}
    keep = np.isin(src, src_classes)
    X = feat[keep]
    y = np.array([src_to_int[s] for s in src[keep]], dtype=np.int64)
    tts_vs_real_res = fit_linear_probe(X, y, name="tts_vs_real", seed=seed)

    # ---- domain (binary synthetic vs real) ---- #
    is_real = ((src == "stresspresso") | (src == "librispeech") | (src == "expresso"))
    is_synth = (src == "stress17k")
    keep = is_real | is_synth
    X = feat[keep]
    y = np.where(is_synth[keep], 0, 1).astype(np.int64)
    domain_res = fit_linear_probe(X, y, name="domain", seed=seed)

    return {
        "voice_id":   voice_id_res,
        "tts_vs_real": tts_vs_real_res,
        "domain":      domain_res,
    }
