"""Per-sample audio augmentation (Stage 2.1 / design §8.3 / Stage 3 codec).

Stochastic augmentations applied to 16 kHz mono float32 waveforms BEFORE the
WavLM forward pass. The job of the audio-domain anchor is to keep the
adapter from being a TTS-detector — augmentations vary the surface acoustic
signature while preserving the prosodic / lexical content.

Implemented:
  - bandlimit       (lowpass cutoff ∈ {3.5, 5, 7, 8} kHz)
  - synthetic reverb (exponential-decay impulse, T60 ∈ [0.1, 0.5] s)
  - additive noise   (white / pink / brown at SNR ∈ {5, 10, 20, 40} dB)
  - codec roundtrip  (one-of {mu-law, MP3-64, MP3-128, Opus-24, Opus-48})
  - resample jitter  (8 / 16 / 22.05 / 24 kHz round-trip)

Stage 3 makes ffmpeg-backed MP3/Opus roundtrips MANDATORY because the
structural counterfactual loss explicitly rewards response divergence on
same-transcript pairs, which makes a TTS-fingerprint shortcut load-bearing.
The codec_roundtrip() function dispatches between the numpy mu-law path and
ffmpeg subprocess paths for MP3/Opus at the chosen bitrate.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.signal as sps
import soxr

SR = 16000

# ffmpeg from /home/nurgaly/.local/bin (ships libmp3lame + libopus)
FFMPEG_BIN = shutil.which("ffmpeg") or "/home/nurgaly/.local/bin/ffmpeg"
HAVE_FFMPEG = os.path.exists(FFMPEG_BIN)


# ---------- individual augmentations ---------- #

def bandlimit(x: np.ndarray, *, cutoff_hz: float) -> np.ndarray:
    """Apply an 8-th order Butterworth lowpass at `cutoff_hz` (zero-phase)."""
    nyq = SR / 2
    if cutoff_hz >= nyq * 0.95:
        return x
    sos = sps.butter(8, cutoff_hz / nyq, btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32, copy=False)


def synthetic_reverb(x: np.ndarray, *, t60_s: float,
                     np_rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic exponential-decay RIR at T60 = `t60_s`. Cheap stand-in for
    a real RIR set; keeps the convolutional spectral colour without the
    overhead of bundling impulse responses.

    `np_rng` lets the caller pass a seeded numpy Generator for deterministic
    cf-pair augmentation; defaults to the global RNG for back-compat.
    """
    if t60_s <= 0.005:
        return x
    L = max(64, int(0.05 * SR + 4 * t60_s * SR))
    n = np.arange(L, dtype=np.float32)
    decay = np.exp(-n / (t60_s * SR / np.log(1000.0)))
    g = np_rng if np_rng is not None else np.random
    rir = decay * g.standard_normal(L).astype(np.float32)   # exponential * white
    rir /= max(np.linalg.norm(rir), 1e-8)
    y = sps.fftconvolve(x, rir, mode="full")[: x.shape[0]]
    # match output RMS to input RMS
    rms_x = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-10))
    rms_y = float(np.sqrt(np.mean(y.astype(np.float32) ** 2) + 1e-10))
    if rms_y > 0:
        y *= rms_x / rms_y
    return y.astype(np.float32, copy=False)


def _make_noise(n_samples: int, color: str,
                np_rng: np.random.Generator | None = None) -> np.ndarray:
    g = np_rng if np_rng is not None else np.random.default_rng()
    if color == "white":
        return g.standard_normal(n_samples).astype(np.float32)
    # 1/f and 1/f^2 via FFT shaping
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SR)
    spectrum = g.standard_normal(freqs.shape) + 1j * g.standard_normal(freqs.shape)
    if color == "pink":
        weight = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-3), 1.0)
    elif color == "brown":
        weight = np.where(freqs > 0, 1.0 / (freqs + 1e-3), 1.0)
    else:
        raise ValueError(color)
    spectrum = spectrum * weight
    y = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    if y.std() > 0:
        y /= y.std()
    return y


def additive_noise(x: np.ndarray, *, color: str, snr_db: float,
                   np_rng: np.random.Generator | None = None) -> np.ndarray:
    rms_x = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-10))
    if rms_x <= 0:
        return x
    n = _make_noise(x.shape[0], color, np_rng=np_rng)
    rms_n = float(np.sqrt(np.mean(n ** 2) + 1e-10))
    target_rms_n = rms_x / (10 ** (snr_db / 20.0))
    if rms_n > 0:
        n *= target_rms_n / rms_n
    return (x + n).astype(np.float32, copy=False)


def mu_law_roundtrip(x: np.ndarray, *, mu: int = 255) -> np.ndarray:
    """Symmetric mu-law companding → 8-bit quantize → expand."""
    x = np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)
    # encode
    enc = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # 8-bit quantize
    q = np.round((enc + 1.0) * (mu / 2.0)).astype(np.int16)
    q = np.clip(q, 0, mu)
    # decode
    enc_q = q.astype(np.float32) / (mu / 2.0) - 1.0
    out = np.sign(enc_q) * (np.expm1(np.abs(enc_q) * np.log1p(mu)) / mu)
    return out.astype(np.float32, copy=False)


# ---- ffmpeg-backed codec roundtrip (Stage 3.1) ---- #

_FFMPEG_CONFIG = {
    "mp3_64":  ("libmp3lame", 64,  "mp3"),
    "mp3_128": ("libmp3lame", 128, "mp3"),
    "opus_24": ("libopus",    24,  "ogg"),
    "opus_48": ("libopus",    48,  "ogg"),
}


def _ffmpeg_roundtrip(x: np.ndarray, *, codec_id: str) -> np.ndarray:
    """Encode `x` to `codec_id`, decode back to PCM-16, return float32 array.

    Uses two short subprocess pipes (~40-70 ms total per call). Length
    typically grows by ~1k samples on MP3 (decoder padding); we crop /pad
    back to the input length so downstream attention masks line up.
    """
    if not HAVE_FFMPEG:
        # Graceful degradation: behave like mu-law if ffmpeg is missing.
        return mu_law_roundtrip(x)
    codec, kbps, container = _FFMPEG_CONFIG[codec_id]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    enc_cmd = [
        FFMPEG_BIN, "-loglevel", "error",
        "-f", "s16le", "-ar", str(SR), "-ac", "1",
        "-i", "pipe:0",
        "-c:a", codec, "-b:a", f"{kbps}k",
        "-f", container, "pipe:1",
    ]
    dec_cmd = [
        FFMPEG_BIN, "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-ar", str(SR), "-ac", "1",
        "pipe:1",
    ]
    try:
        enc = subprocess.run(enc_cmd, input=pcm, capture_output=True,
                             check=True, timeout=30).stdout
        dec = subprocess.run(dec_cmd, input=enc, capture_output=True,
                             check=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        # If ffmpeg blows up on a particular waveform, fall back silently.
        return mu_law_roundtrip(x)
    y = np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32768.0
    n = x.shape[0]
    if y.shape[0] >= n:
        y = y[:n]
    else:
        y = np.pad(y, (0, n - y.shape[0]))
    return y.astype(np.float32, copy=False)


def codec_roundtrip(x: np.ndarray, *, codec_id: str) -> np.ndarray:
    """Dispatch between mu-law (numpy) and ffmpeg (subprocess) codecs.

    `codec_id` ∈ {"mu_law", "mp3_64", "mp3_128", "opus_24", "opus_48"}.
    """
    if codec_id == "mu_law":
        return mu_law_roundtrip(x)
    if codec_id in _FFMPEG_CONFIG:
        return _ffmpeg_roundtrip(x, codec_id=codec_id)
    raise ValueError(f"unknown codec_id: {codec_id}")


def resample_jitter(x: np.ndarray, *, intermediate_sr: int) -> np.ndarray:
    if intermediate_sr == SR:
        return x
    y = soxr.resample(x.astype(np.float32, copy=False), SR, intermediate_sr)
    z = soxr.resample(y, intermediate_sr, SR)
    # tiny length adjustment from rounding
    if z.shape[0] >= x.shape[0]:
        z = z[: x.shape[0]]
    else:
        z = np.pad(z, (0, x.shape[0] - z.shape[0]))
    return z.astype(np.float32, copy=False)


# ---------- combined per-sample policy ---------- #

@dataclass
class AugmentConfig:
    apply_prob: float = 0.85       # per-augmentation independent dropout
    bandlimit_choices: tuple = (3500.0, 5000.0, 7000.0, 8000.0)
    reverb_t60_range: tuple = (0.1, 0.5)
    noise_colors: tuple = ("white", "pink", "brown")
    snr_db_choices: tuple = (5, 10, 20, 40)
    intermediate_sr_choices: tuple = (8000, 16000, 22050, 24000)
    # Stage 3.1: codec randomization. Default policy is uniform over the
    # 5-codec family. ffmpeg-backed codecs cost ~50-70ms each; total budget
    # at apply_prob=0.85 with 4/5 ffmpeg = ~45ms expected per audio.
    codec_choices: tuple = ("mu_law", "mp3_64", "mp3_128", "opus_24", "opus_48")
    # When False, skip ffmpeg codecs entirely (Stage-2 behavior).
    use_ffmpeg_codecs: bool = True
    # Stage 3.0.5 revision: pre-augment RMS normalization to a common
    # target. Set to None to disable (Stage-2 behavior).
    rms_norm_target: float | None = None


def aggressive_stage3_config() -> "AugmentConfig":
    """Strengthened augmentation preset for Stage 3.0.5 revision.

    All augmentations always apply; bandlimit floor low; SNR floor low;
    RMS-normalized input. Aim is to mask source-distinguishing recording
    signatures across {Stress-17K-TTS, LibriSpeech, Expresso, StressPresso}.
    """
    return AugmentConfig(
        apply_prob=1.0,                                  # everything always
        bandlimit_choices=(3500.0, 4000.0, 5000.0),       # tighter
        reverb_t60_range=(0.2, 0.7),                      # heavier
        snr_db_choices=(0, 3, 6, 10, 15),                 # noisier
        intermediate_sr_choices=(8000, 12000, 16000),     # narrower
        codec_choices=("mu_law", "mp3_64", "mp3_128", "opus_24", "opus_48"),
        use_ffmpeg_codecs=True,
        rms_norm_target=0.05,
    )


def rms_normalize(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Scale `x` so that its RMS equals `target_rms`."""
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12))
    if rms <= 1e-8:
        return x
    return (x.astype(np.float32) * (target_rms / rms)).astype(np.float32, copy=False)


def augment_one(x: np.ndarray, *, cfg: AugmentConfig | None = None,
                rng: random.Random | None = None) -> np.ndarray:
    """Apply each augmentation independently w/ probability cfg.apply_prob.

    Order matters slightly (reverb before noise feels more natural). Each
    augmentation has its own per-call random draw; total compute per call is
    bounded — at most 5 small operations on a ~5 s audio buffer.

    For counterfactual pair augmentation (kickoff §3.1), the caller should
    pass the SAME `rng` (or seed) to both pair members so the augmentation
    parameters AND internal random draws (RIR, noise) match across the pair.
    Internally we derive a numpy Generator from a 32-bit draw of `rng` so the
    seed pinning propagates into reverb and additive_noise.
    """
    cfg = cfg or AugmentConfig()
    rng = rng or random.Random()
    # Derive a deterministic numpy RNG from the python rng for reverb/noise.
    np_rng = np.random.default_rng(rng.getrandbits(63))

    y = x.astype(np.float32, copy=True)
    # Optional pre-augment RMS normalization for cross-corpus level matching.
    if cfg.rms_norm_target is not None:
        y = rms_normalize(y, target_rms=cfg.rms_norm_target)
    if rng.random() < cfg.apply_prob:
        y = resample_jitter(y, intermediate_sr=rng.choice(cfg.intermediate_sr_choices))
    if rng.random() < cfg.apply_prob:
        y = bandlimit(y, cutoff_hz=rng.choice(cfg.bandlimit_choices))
    if rng.random() < cfg.apply_prob:
        t60 = rng.uniform(*cfg.reverb_t60_range)
        y = synthetic_reverb(y, t60_s=t60, np_rng=np_rng)
    if rng.random() < cfg.apply_prob:
        y = additive_noise(y, color=rng.choice(cfg.noise_colors),
                           snr_db=rng.choice(cfg.snr_db_choices),
                           np_rng=np_rng)
    if rng.random() < cfg.apply_prob:
        if cfg.use_ffmpeg_codecs and HAVE_FFMPEG:
            codec_id = rng.choice(cfg.codec_choices)
        else:
            codec_id = "mu_law"
        y = codec_roundtrip(y, codec_id=codec_id)
    # Clip to a sane peak in case any stage blew up amplitude.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = (y / peak * 0.99).astype(np.float32, copy=False)
    return y
