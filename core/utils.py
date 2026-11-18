"""Math, audio, image and synthetic-demo helpers shared by every layer."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import math
import os
import re
import wave
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, PngImagePlugin

from neurogate.core.config import AUDIO_DIR, DEMO_RESIDENTS, GALLERY_DIR, IMG_EXT, SR


# =============================================================================
# Small numeric / string helpers
# =============================================================================
def l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def stem_from_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_") or "Resident"


def name_from_stem(stem: str) -> str:
    return re.sub(r"_+", " ", stem).strip()


def img_b64(img: Image.Image, size: Optional[int] = None) -> str:
    im = img.convert("RGB")
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def robust_fit(xs: List[float], floor: float = 0.02) -> Tuple[float, float]:
    """Median and MAD-sigma: robust to look-alike outliers such as twins."""
    if not xs:
        return 0.0, floor
    a = np.asarray(xs, dtype=np.float64)
    med = float(np.median(a))
    sig = float(1.4826 * np.median(np.abs(a - med)))
    return med, max(sig, floor)


# =============================================================================
# Feature primitives (HOG, log-mel) and audio I/O
# =============================================================================
def hog(gray: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    idx = np.minimum((ang / np.pi * bins).astype(np.int32), bins - 1)
    H, W = g.shape
    ch, cw = H // cell, W // cell
    m = mag[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)
    a = idx[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)
    hist = np.stack([(m * (a == b)).sum(axis=(1, 3)) for b in range(bins)], -1)
    if ch < 2 or cw < 2:
        v = hist.ravel()
        return v / (np.linalg.norm(v) + 1e-6)
    blk = np.concatenate([hist[:-1, :-1], hist[1:, :-1], hist[:-1, 1:], hist[1:, 1:]], -1)
    blk = blk / (np.linalg.norm(blk, axis=-1, keepdims=True) + 1e-6)
    blk = np.minimum(blk, 0.2)
    blk = blk / (np.linalg.norm(blk, axis=-1, keepdims=True) + 1e-6)
    return blk.ravel().astype(np.float32)


def _mel_fb(n_mels=40, n_fft=512, sr=SR, fmin=20.0, fmax=7600.0) -> np.ndarray:
    def hz2mel(f): return 2595 * np.log10(1 + f / 700.0)
    def mel2hz(m): return 700 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins_ = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins_[i - 1], bins_[i], bins_[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


MEL_FB = _mel_fb()


def logmel(x: np.ndarray) -> np.ndarray:
    n_fft, hop, win = 512, 160, 400
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    w = np.hamming(win).astype(np.float32)
    frames = np.stack([x[i * hop: i * hop + win] * w for i in range(n)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    return np.log(spec @ MEL_FB.T + 1e-6)


def to_mono16k(sr: int, data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if sr != SR and len(x) > 1:
        t_new = np.linspace(0, len(x) / sr, int(len(x) * SR / sr), endpoint=False)
        x = np.interp(t_new, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def read_wav(path: str) -> np.ndarray:
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32")
        return to_mono16k(sr, data)
    except Exception:
        pass
    with wave.open(path, "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    a = np.frombuffer(raw, dtype=dtype)
    if sw == 1:
        a = (a.astype(np.float32) - 128) / 128.0
    if nch > 1:
        a = a.reshape(-1, nch)
    return to_mono16k(sr, a)


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    y = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def hf_ratio(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[freqs > 4000].sum() / (spec.sum() + 1e-12))


def decode_wav_bytes(data: bytes) -> np.ndarray:
    """16-bit PCM WAV bytes (e.g. from the browser sentry) → mono float32 at 16 kHz."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, nch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if nch > 1:
        a = a.reshape(-1, nch)
    return to_mono16k(sr, a)


def decode_data_url(s: str) -> bytes:
    return base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)


# =============================================================================
# Augmentation and replay-attack simulation
# =============================================================================
def augment_face(img: Image.Image, seed: int, strength: float = 1.0) -> Image.Image:
    r = np.random.default_rng(seed)
    im = img.convert("RGB")
    im = ImageEnhance.Brightness(im).enhance(1 + strength * r.uniform(-0.1, 0.1))
    im = ImageEnhance.Contrast(im).enhance(1 + strength * r.uniform(-0.1, 0.1))
    im = im.rotate(strength * r.uniform(-3, 3), resample=Image.BILINEAR, fillcolor=im.getpixel((0, 0)))
    a = np.asarray(im, dtype=np.float32) + r.normal(0, 3.0 * strength, (im.size[1], im.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def augment_voice(x: np.ndarray, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    y = np.roll(x, int(r.integers(0, SR // 8))) * r.uniform(0.75, 1.0)
    p = np.mean(y ** 2)
    return (y + r.normal(0, math.sqrt(p / 10 ** 2.5), len(y))).astype(np.float32)  # 25 dB SNR


def replay_channel(x: np.ndarray) -> np.ndarray:
    """Loudspeaker/phone replay: 300-3400 Hz band-limit + small-room reverb."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(X, n=len(x))
    ir = np.exp(-np.arange(int(0.05 * SR)) / (0.012 * SR))
    y = np.convolve(y, ir / ir.sum())[: len(x)]
    return (0.5 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


# =============================================================================
# Synthetic demo residents (procedural faces + harmonic voices)
# =============================================================================
SKIN = [(141, 85, 36), (198, 134, 66), (110, 68, 36), (160, 102, 60), (90, 56, 30), (175, 120, 80)]


def face_params(seed: int) -> dict:
    r = np.random.default_rng(seed)
    return dict(
        bg=tuple(int(v) for v in r.integers(200, 245, 3)),
        skin=SKIN[int(r.integers(0, len(SKIN)))],
        rx=int(r.integers(60, 80)), ry=int(r.integers(80, 98)),
        jaw=float(r.uniform(0.55, 0.95)),
        hair=int(r.integers(0, 3)), hair_col=tuple(int(v) for v in r.integers(10, 60, 3)),
        eye_y=int(r.integers(96, 112)), eye_dx=int(r.integers(24, 38)),
        eye_rx=int(r.integers(9, 15)), eye_ry=int(r.integers(5, 9)),
        brow_t=int(r.integers(3, 7)), brow_dy=int(r.integers(11, 19)), brow_tilt=int(r.integers(-5, 6)),
        nose_len=int(r.integers(22, 38)), nose_w=int(r.integers(10, 22)),
        mouth_y=int(r.integers(158, 176)), mouth_w=int(r.integers(22, 42)), lip=int(r.integers(4, 10)),
        glasses=bool(r.random() < 0.2),
    )


def render_face(p: dict) -> Image.Image:
    S = 224
    im = Image.new("RGB", (S, S), p["bg"])
    d = ImageDraw.Draw(im)
    cx, cy = 112, 120
    skin = p["skin"]
    dark = tuple(max(0, c - 40) for c in skin)
    # hair back
    if p["hair"] == 1:
        d.ellipse([cx - p["rx"] - 12, cy - p["ry"] - 18, cx + p["rx"] + 12, cy + 10], fill=p["hair_col"])
    # ears
    d.ellipse([cx - p["rx"] - 9, cy - 14, cx - p["rx"] + 9, cy + 18], fill=dark)
    d.ellipse([cx + p["rx"] - 9, cy - 14, cx + p["rx"] + 9, cy + 18], fill=dark)
    # head + jaw
    d.ellipse([cx - p["rx"], cy - p["ry"], cx + p["rx"], cy + int(p["ry"] * 0.6)], fill=skin)
    jw = int(p["rx"] * p["jaw"])
    d.polygon([(cx - p["rx"] + 2, cy), (cx + p["rx"] - 2, cy), (cx + jw, cy + int(p["ry"] * 0.75)),
               (cx, cy + p["ry"]), (cx - jw, cy + int(p["ry"] * 0.75))], fill=skin)
    # hair top
    if p["hair"] in (0, 1):
        d.chord([cx - p["rx"], cy - p["ry"] - 6, cx + p["rx"], cy - p["ry"] + 70], 180, 360, fill=p["hair_col"])
    else:
        for k in range(-p["rx"] + 8, p["rx"] - 4, 9):
            d.ellipse([cx + k - 6, cy - p["ry"] - 4, cx + k + 6, cy - p["ry"] + 12], fill=p["hair_col"])
    # eyes + brows
    for sgn in (-1, 1):
        ex, ey = cx + sgn * p["eye_dx"], p["eye_y"]
        d.ellipse([ex - p["eye_rx"], ey - p["eye_ry"], ex + p["eye_rx"], ey + p["eye_ry"]], fill=(245, 245, 240))
        ir = max(3, p["eye_ry"] - 1)
        d.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=(40, 25, 15))
        by = ey - p["brow_dy"]
        d.line([(ex - p["eye_rx"] - 2, by + sgn * p["brow_tilt"] // 2),
                (ex + p["eye_rx"] + 2, by - sgn * p["brow_tilt"] // 2)], fill=(30, 20, 15), width=p["brow_t"])
        if p["glasses"]:
            d.ellipse([ex - p["eye_rx"] - 6, ey - p["eye_ry"] - 7, ex + p["eye_rx"] + 6, ey + p["eye_ry"] + 7],
                      outline=(30, 30, 30), width=2)
    # nose
    ny0, ny1 = p["eye_y"] + 6, p["eye_y"] + 6 + p["nose_len"]
    d.line([(cx - 3, ny0), (cx - p["nose_w"] // 3, ny1)], fill=dark, width=3)
    d.ellipse([cx - p["nose_w"] // 2 - 2, ny1 - 4, cx - p["nose_w"] // 2 + 6, ny1 + 3], fill=dark)
    d.ellipse([cx + p["nose_w"] // 2 - 6, ny1 - 4, cx + p["nose_w"] // 2 + 2, ny1 + 3], fill=dark)
    # mouth
    my = p["mouth_y"]
    d.chord([cx - p["mouth_w"], my - p["lip"], cx + p["mouth_w"], my + p["lip"] + 4], 0, 180, fill=(120, 40, 45))
    d.line([(cx - p["mouth_w"], my), (cx + p["mouth_w"], my)], fill=(70, 20, 25), width=2)
    return im


def voice_params(seed: int) -> dict:
    r = np.random.default_rng(seed)
    f0 = float(r.uniform(95, 230))
    return dict(f0=f0, F=[float(r.uniform(450, 850)), float(r.uniform(1100, 2100)), float(r.uniform(2300, 3200))],
                bw=[float(r.uniform(80, 160)), float(r.uniform(120, 240)), float(r.uniform(180, 320))],
                rate=float(r.uniform(3.0, 5.5)), breath=float(r.uniform(0.006, 0.014)))


def synth_voice(p: dict, seed: int, dur: float = 1.8) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f0t = p["f0"] * (1 + 0.06 * np.sin(2 * np.pi * 1.3 * t + r.uniform(0, 6)) + 0.01 * r.standard_normal(len(t)).cumsum() / 400)
    phase = 2 * np.pi * np.cumsum(f0t) / SR
    y = np.zeros_like(t)
    nh = int(7500 // p["f0"])
    for h in range(1, nh + 1):
        hf = h * p["f0"]
        amp = sum(math.exp(-((hf - F) / bw) ** 2) for F, bw in zip(p["F"], p["bw"])) + 0.03
        y += amp / math.sqrt(h) * np.sin(h * phase)
    y *= 0.55 + 0.45 * np.sin(np.pi * p["rate"] * t) ** 2
    noise = r.standard_normal(len(t))
    noise = noise - np.convolve(noise, np.ones(4) / 4, mode="same")  # high-passed breath
    y = y / (np.abs(y).max() + 1e-9) + p["breath"] * 6 * noise
    return (0.5 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def synthetic_face_for(name: str) -> Image.Image:
    if name == "Kehinde Adekunle":  # identical twin of Taiwo: tiny differences only
        p = face_params(seed_of("face", "Taiwo Adekunle"))
        p["mouth_w"] += 2
        p["brow_tilt"] = -p["brow_tilt"] if p["brow_tilt"] else 2
        p["eye_dx"] += 1
        return render_face(p)
    return render_face(face_params(seed_of("face", name)))


def save_synthetic_png(img: Image.Image, path: str):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("neurogate_synthetic", "1")
    img.save(path, pnginfo=meta)


def seed_demo_gallery():
    if any(f.lower().endswith(IMG_EXT) for f in os.listdir(GALLERY_DIR)):
        return
    for name, _unit in DEMO_RESIDENTS:
        stem = stem_from_name(name)
        save_synthetic_png(synthetic_face_for(name), os.path.join(GALLERY_DIR, stem + ".png"))
        write_wav(os.path.join(AUDIO_DIR, stem + ".wav"),
                  synth_voice(voice_params(seed_of("voice", name)), seed_of("utt", name, 0)))
