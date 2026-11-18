"""Environmental awareness for the doorway (AAS Stage 1 scene check + Stage 2 acoustics).

Lighting
    Measures exposure on the whole frame and on a central subject zone: mean luma,
    saturated / crushed pixel fractions, border-vs-centre backlight and Laplacian sharpness.
    Dim, dark and backlit frames are corrected (gamma + percentile stretch) *before*
    face alignment. Only a frame that is still too dark after correction is marked
    unusable, which feeds the existing `quality_ok(P)` fact: the ASP gate is unchanged.

    Fairness note: every test uses exposure statistics (clipping, frame-level luma,
    border/centre contrast) rather than the brightness of the face itself, so darker
    skin tones in front of a bright wall are not treated as "dark" or "backlit".
    Correction only ever brightens; it never denies on its own.

Acoustics
    Estimates noise floor, speech level and SNR from the 3.5 s recording, plus digital
    clipping. Informational: it explains liveness failures and shapes the spoken advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
from PIL import Image, ImageOps

from neurogate.core.config import CFG, SR


# =============================================================================
# Lighting
# =============================================================================
@dataclass
class LightingReport:
    condition: str              # good | dim | dark | backlit | glare | blurry | too_dark
    luma: float                 # frame mean luma (0-255)
    center_luma: float          # subject-zone mean luma
    contrast: float             # frame luma std
    clip_bright: float          # saturated fraction in the subject zone
    border_clip: float          # saturated fraction on the frame border
    crushed: float              # near-black fraction of the frame
    sharpness: float            # Laplacian variance of the subject zone
    usable: bool = True
    corrected: bool = False
    gamma: float = 1.0
    luma_after: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def icon(self) -> str:
        return {"good": "💡", "dim": "🌥️", "dark": "🌙", "too_dark": "🌑", "backlit": "🌅",
                "glare": "☀️", "blurry": "💨"}.get(self.condition, "💡")

    @property
    def label(self) -> str:
        base = {"good": "Good light", "dim": "Dim light", "dark": "Low light", "too_dark": "Too dark",
                "backlit": "Backlit", "glare": "Glare", "blurry": "Motion blur"}.get(self.condition, self.condition)
        return base + (" · auto-corrected" if self.corrected and self.usable else "")

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def _luma(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def assess_lighting(img: Image.Image) -> LightingReport:
    y = _luma(img)
    H, W = y.shape
    cy0, cy1, cx0, cx1 = int(H * 0.15), int(H * 0.85), int(W * 0.25), int(W * 0.75)
    center = y[cy0:cy1, cx0:cx1]
    bw = max(2, int(min(H, W) * 0.08))
    border = np.concatenate([y[:bw].ravel(), y[-bw:].ravel(), y[:, :bw].ravel(), y[:, -bw:].ravel()])
    lap = center[1:-1, 1:-1] * 4 - center[:-2, 1:-1] - center[2:, 1:-1] - center[1:-1, :-2] - center[1:-1, 2:]
    r = LightingReport(condition="good", luma=float(y.mean()), center_luma=float(center.mean()),
                       contrast=float(y.std()), clip_bright=float((center >= 250).mean()),
                       border_clip=float((border >= 250).mean()), crushed=float((y <= 8).mean()),
                       sharpness=float(lap.var()))
    if (r.luma < CFG["env_dark_luma"] and r.crushed > 0.05) or r.luma < CFG["env_dark_luma"] * 0.7:
        r.condition = "dark"
    elif r.border_clip > CFG["env_backlit_clip"] and r.center_luma < CFG["env_backlit_center"]:
        r.condition = "backlit"
    elif r.clip_bright > CFG["env_glare_clip"]:
        r.condition = "glare"
    elif r.luma < CFG["env_dim_luma"]:
        r.condition = "dim"
    elif r.sharpness < CFG["env_blur_floor"]:
        r.condition = "blurry"
    r.luma_after = r.luma
    return r


def correct_lighting(img: Image.Image, r: LightingReport) -> Image.Image:
    """Exposure compensation for dim / dark / backlit frames; others pass through."""
    if r.condition not in ("dim", "dark", "backlit"):
        return img
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    ref = max(r.center_luma if r.condition == "backlit" else r.luma, 1.0) / 255.0
    gamma = float(np.clip(np.log(0.45) / np.log(ref), 0.35, 1.0))   # lift mid-tones toward 45 % grey
    out = np.power(np.clip(a, 1e-4, 1.0), gamma)
    out_img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    out_img = ImageOps.autocontrast(out_img, cutoff=1)
    r.corrected, r.gamma = True, round(gamma, 3)
    r.luma_after = float(_luma(out_img).mean())
    if r.luma < CFG["env_unusable_luma"]:   # tiny signal amplified = noise: too dark to recognise anyone
        r.usable = False
        r.condition = "too_dark"
        r.notes.append(f"frame luma {r.luma:.0f} < {CFG['env_unusable_luma']:.0f}: too dark even after correction")
    else:
        r.notes.append(f"gamma {gamma:.2f} + 1 % stretch: luma {r.luma:.0f} → {r.luma_after:.0f}")
    return out_img


def prepare_frame(img: Image.Image):
    """Stage 1 helper: (possibly corrected frame, LightingReport)."""
    r = assess_lighting(img)
    return correct_lighting(img, r), r


def simulate_low_light(img: Image.Image, factor: float = 0.22, noise: float = 4.0, seed: int = 0) -> Image.Image:
    """Demo helper: a dim corridor (exposure × factor + sensor noise)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGB"), dtype=np.float32) * factor
    a += rng.normal(0, noise, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# =============================================================================
# Acoustics
# =============================================================================
@dataclass
class AcousticReport:
    condition: str          # quiet | noisy | very_noisy | clipping | silent
    noise_db: float
    speech_db: float
    snr_db: float
    clipped: float

    @property
    def icon(self) -> str:
        return {"quiet": "🔈", "noisy": "🔉", "very_noisy": "🔊", "clipping": "📢", "silent": "🔇"}.get(self.condition, "🔈")

    @property
    def label(self) -> str:
        return {"quiet": f"Quiet · SNR {self.snr_db:.0f} dB", "noisy": f"Noisy · SNR {self.snr_db:.0f} dB",
                "very_noisy": f"Very noisy · SNR {self.snr_db:.0f} dB", "clipping": "Mic clipping",
                "silent": "No speech heard"}.get(self.condition, self.condition)

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def assess_acoustics(x) -> AcousticReport:
    if x is None or len(x) < SR // 4:
        return AcousticReport("silent", -90.0, -90.0, 0.0, 0.0)
    x = np.asarray(x, dtype=np.float32)
    win = int(0.03 * SR)
    n = len(x) // win
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    noise_db = float(np.percentile(db, 15))
    speech_db = float(np.percentile(db, 90))
    snr = speech_db - noise_db
    clipped = float((np.abs(x) >= 0.99).mean())
    if speech_db < -50:
        cond = "silent"
    elif clipped > 0.01:
        cond = "clipping"
    elif snr < CFG["env_very_noisy_snr"]:
        cond = "very_noisy"
    elif snr < CFG["env_noisy_snr"]:
        cond = "noisy"
    else:
        cond = "quiet"
    return AcousticReport(cond, noise_db, speech_db, snr, clipped)
