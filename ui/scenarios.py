"""Request builders: Dr. Fasina's 5 decision-table rows + Stage 5 tailgating, environment
demos, walk-ups, manual uploads and the hands-free live sentry payload."""
from __future__ import annotations

import io
import json
import re
import time
import wave
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from neurogate.core.config import SR
from neurogate.core.utils import (augment_face, augment_voice, decode_data_url, face_params, render_face,
                                  replay_channel, seed_of, synth_voice, to_mono16k, voice_params)
from neurogate.pipeline.engine import Request, Resident, get_engine
from neurogate.pipeline.environment import simulate_low_light


def _eng():
    return get_engine()


def mic_window(x: np.ndarray, seed: int = 0, seconds: float = 3.5, lead: float = 0.45, room: float = 0.0025) -> np.ndarray:
    """Place a simulated utterance inside a 3.5 s microphone window with room noise, like the live sentry."""
    rng = np.random.default_rng(seed)
    n = max(int(seconds * SR), len(x) + int(lead * SR))
    y = rng.normal(0, room, n).astype(np.float32)
    s = int(lead * SR)
    y[s:s + len(x)] += x[: n - s]
    return y


def _genuine_voice(r: Resident):
    def f(attempt, challenge, _top):
        if r.wav is None:
            return None, None, f"{r.name} has no enrolled voiceprint"
        return (mic_window(augment_voice(r.wav, seed_of("live", r.stem, attempt, challenge)), seed_of("room", attempt)),
                f"Hello, my name is {r.name}. My code is {' '.join(challenge)}.", f"auto-paired live voice: {r.name}")
    return f


def _walk_face(r: Resident, tag: str):
    return lambda attempt: (augment_face(r.photo, seed_of(tag, r.stem, attempt, time.time_ns())), r.synthetic, False)


def _pick_pair():
    R = [r for r in _eng().residents if r.voice_e is not None]
    if len(R) < 2:
        return None
    G = np.stack([r.face_e for r in R])
    S = G @ G.T
    np.fill_diagonal(S, -9)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    twins = (R[i], R[j])
    distinct = sorted(R, key=lambda r: float(np.max(S[R.index(r)])))
    return twins, distinct


def stranger(seed: int, verified: bool = False):
    """Synthetic non-resident. verified=True rejection-samples a visitor who is genuinely unlike every
    enrolled resident (face below τ_low, voice below τ_v), so Row 3 shows a true C = 0 case."""
    for k in range(60 if verified else 1):
        sd = seed * 101 + k
        face = render_face(face_params(seed_of("visitor", sd)))
        wav = synth_voice(voice_params(seed_of("visitor-voice", sd)), sd)
        if not verified or not _eng().residents:
            return face, wav
        e = _eng().face_backend.embed(_eng().capture(face, True)[0])[0]
        v = _eng().voice_backend.embed(wav)
        vmax = max((float(r.voice_e @ v) for r in _eng().residents if r.voice_e is not None), default=-1)
        if float((_eng().G @ e).max()) < _eng().calib["tau_low"] and vmax < _eng().calib["tau_v"]:
            return face, wav
    return face, wav


SCENARIOS = {
    "row1": "✅ Row 1: Full Recognition (C=1 + Live Voice Match -> Allow)",
    "row2": "❌ Row 2: Face/Voice Identity Conflict (Face=i, Voice=j -> Deny & 1 Retry)",
    "row3": "👤 Row 3: Unrecognized Visitor (C=0 -> Route to Human Host)",
    "row4": "👥 Row 4: Ambiguous Lookalikes/Twins (C>1 -> Resolved by Voice / Escalate)",
    "row5": "🚨 Row 5: Voice Liveness / Replay Attack Failure (Deny & Security Alert)",
    "tail": "🚪 Stage 5: Anti-Tailgating Alert (2 People Cross on 1 Unlock)",
}


def build_scenario(key: str) -> Tuple[Optional[Request], str]:
    pp = _pick_pair()
    if pp is None:
        return None, "Enrol at least two residents with photos and voiceprints to run scenarios."
    (t1, t2), distinct = pp
    A = distinct[0]
    B = next(r for r in distinct if r is not A and r not in (t1, t2)) if len(distinct) > 2 else distinct[1]
    label = SCENARIOS[key]
    if key in ("row1", "tail"):
        return Request(label, _walk_face(A, key), _genuine_voice(A), crossings=2 if key == "tail" else 1), ""
    if key == "row2":
        def v(attempt, challenge, _t):
            wav, tr, _ = _genuine_voice(B)(attempt, challenge, None)
            return wav, tr, f"voice belongs to {B.name} (face is {A.name})"
        return Request(label, _walk_face(A, key), v), ""
    if key == "row3":
        n = int(time.time()) % 997
        face, wav = stranger(n, verified=True)
        return Request(label, lambda a: (augment_face(face, a), True, False),
                       lambda a, c, t: (wav, f"Good afternoon, my name is Musa Ibrahim, I'm here to see a resident. Code {' '.join(c)}.",
                                        "visitor voice")), ""
    if key == "row4":
        def face(attempt):
            blend = Image.blend(t1.aligned, t2.aligned, 0.5)
            return augment_face(blend, seed_of("twin", attempt, time.time_ns()), 0.5), True, True
        return Request(label, face, _genuine_voice(t1)), ""
    if key == "row5":
        def v(attempt, challenge, _t):
            return replay_channel(t_wav(A)), f"My name is {A.name}.", f"loudspeaker replay of {A.name}'s enrolment recording"
        return Request(label, _walk_face(A, key), v), ""
    return None, "Unknown scenario"


def t_wav(r: Resident) -> np.ndarray:
    return r.wav if r.wav is not None else np.zeros(SR, np.float32)


def build_walkup(name: str) -> Tuple[Optional[Request], str]:
    r = _eng().by_name(name)
    if r is None:
        return None, "Pick an enrolled resident."
    return Request(f"Walk-up: {r.name}", _walk_face(r, "walk"), _genuine_voice(r)), ""


def build_camera(img: Image.Image, mic, typed_code: str) -> Tuple[Optional[Request], str]:
    if img is None:
        return None, "No camera frame."
    synthetic = getattr(img, "info", {}).get("neurogate_synthetic") == "1"

    def face(attempt):
        return (img if attempt == 1 else augment_face(img, attempt, 0.3)), synthetic, False

    if mic is not None:
        sr, data = mic
        wav = to_mono16k(sr, data)
        return Request("Doorway camera + live microphone", face,
                       lambda a, c, t: (wav, None, "live microphone"), typed_code=typed_code), ""

    def auto_voice(attempt, challenge, top):
        # Hands-free simulation: the person in frame speaks with their enrolled voice.
        if top is not None and _eng().residents[top].wav is not None:
            return _genuine_voice(_eng().residents[top])(attempt, challenge, top)
        _, wav = stranger(seed_of("cam-visitor", attempt))
        return wav, f"Hello, I'm a visitor. Code {' '.join(challenge)}.", "visitor voice (no enrolled match to pair)"
    return Request("Doorway camera (voice auto-paired)", face, auto_voice), ""


def build_live(payload: str) -> Tuple[Optional[Request], str]:
    """Payload from the browser Auto-Sentry: auto-captured frame + 3.5 s 16 kHz WAV + Web Speech transcript."""
    try:
        d = json.loads(payload or "{}")
        img = Image.open(io.BytesIO(decode_data_url(d["img"]))).convert("RGB")
    except Exception as e:
        return None, f"Live capture could not be decoded ({e})"
    wav = None
    if d.get("wav"):
        try:
            with wave.open(io.BytesIO(decode_data_url(d["wav"])), "rb") as w:
                sr = w.getframerate()
                a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            wav = to_mono16k(sr, a)
        except Exception as e:
            print("[live] bad wav:", e)
    transcript = (d.get("transcript") or "").strip() or None
    challenge = re.sub(r"\D", "", str(d.get("challenge", "")))[:4] or None
    n = int(d.get("n_faces") or 0)
    src = "Web Speech API" if transcript else "server ASR"
    mode = f"live microphone · {d.get('rec_seconds', 3.5)} s auto-record · transcript via {src}"
    return Request(f"Live doorway sentry ({d.get('detector', 'browser')})",
                   lambda attempt: (img, False, False),
                   lambda attempt, c, t: (wav, transcript, mode),
                   occupancy=n if n > 1 else None, challenge=challenge, live=True,
                   nonce=re.sub(r"[^0-9A-Za-z]", "", str(d.get("nonce", "")))[:40]), ""


# =============================================================================
# Environmental-awareness demos
# =============================================================================
ENV_SCENARIOS = {
    "env_dim": "🌙 Environment: Dim corridor (auto-brightened → Allow)",
    "env_dark": "🌑 Environment: Pitch-dark doorway (spoken guidance → retry → guard)",
}


def build_env_scenario(key: str) -> Tuple[Optional[Request], str]:
    pp = _pick_pair()
    if pp is None:
        return None, "Enrol at least two residents with photos and voiceprints to run scenarios."
    _, distinct = pp
    A = distinct[0]
    factor = 0.30 if key == "env_dim" else 0.05

    def face(attempt):
        base = augment_face(A.photo, seed_of(key, A.stem, attempt, time.time_ns()))
        return simulate_low_light(base, factor, noise=3.0 if key == "env_dim" else 2.0, seed=attempt), A.synthetic, False
    return Request(ENV_SCENARIOS[key], face, _genuine_voice(A)), ""
