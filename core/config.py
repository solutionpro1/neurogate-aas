"""NeuroGate AAS — configuration: paths, thresholds, constants and base CSS.

Every tunable used by the AAS outer loop (Stages 1-5) and the NSFRS inner module
(Stages 0-3) lives in CFG so the kiosk threshold sliders can change it at runtime.
"""
from __future__ import annotations

import os
from typing import Dict

VERSION = "3.1.0"

# Package root (the folder that contains app.py); data folders live beside it.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY_DIR = os.path.join(BASE_DIR, "gallery")
AUDIO_DIR = os.path.join(BASE_DIR, "gallery_audio")
COHORT_DIR = os.path.join(BASE_DIR, "cohort")   # optional non-resident faces/voices for impostor calibration
CACHE_DIR = os.path.join(BASE_DIR, ".neurogate_cache")
DB_PATH = os.path.join(BASE_DIR, "neurogate_aas.db")
for _d in (GALLERY_DIR, AUDIO_DIR, CACHE_DIR, COHORT_DIR):
    os.makedirs(_d, exist_ok=True)

SR = 16000
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

CFG: Dict[str, float] = {
    "theta": 60.0,           # NSFRS belief threshold for match(p, I)
    "quality_floor": 12.0,   # AdaFace ||z||_2 floor for quality_ok(P)
    "voice_margin": 0.05,    # min best-vs-second voice margin for voice_ok
    "far_face": 1e-2,        # system-level face FAR budget (per-pair f = FAR / N)
    "far_voice": 1e-2,       # system-level voice FAR budget (per-pair v = FAR / N)
    "frag_delta": 0.06,      # Alibi flip distance below which a match is fragile
    "frag_penalty": 12.0,    # max belief points removed from a fragile top match
    "shortlist_k": 5,
    "energy_floor": 0.004,   # RMS floor for liveness (3.5 s window incl. pauses)
    "hf_floor": 0.002,       # >4 kHz energy ratio floor (loudspeaker replay check)
    # ---- environmental awareness (pipeline/environment.py) ----
    "env_dark_luma": 55.0,       # frame mean luma below this = dark  (0-255)
    "env_dim_luma": 85.0,        # below this = dim (auto-brightened)
    "env_unusable_luma": 22.0,   # even after correction: too dark to recognise → quality_ok fails
    "env_glare_clip": 0.30,      # fraction of saturated pixels in the face zone = glare
    "env_backlit_clip": 0.35,    # saturated fraction of the frame border ...
    "env_backlit_center": 70.0,  # ... while the subject zone is darker than this = backlit
    "env_blur_floor": 25.0,      # Laplacian variance below this = motion blur / out of focus
    "env_noisy_snr": 12.0,       # dB: below = noisy doorway
    "env_very_noisy_snr": 6.0,   # dB: below = very noisy (speech hard to hear)
}

ATTRS = ["periocular_eyes", "nasal_bridge", "jaw_mouth", "craniofacial_proportions"]


ATTR_LABEL = {"periocular_eyes": "Eyes", "nasal_bridge": "Nose",
              "jaw_mouth": "Jaw", "craniofacial_proportions": "Proportions"}

# Aligned 112x112 crop regions used by the attribute (VLM surrogate) scorers.
REGIONS = {"periocular_eyes": (26, 58, 6, 106), "nasal_bridge": (38, 86, 30, 82), "jaw_mouth": (68, 112, 10, 102)}

DEMO_RESIDENTS = [
    ("Adaeze Okafor", "Flat 2A"), ("Bola Adeyemi", "Flat 3B"), ("Chinedu Eze", "Flat 1C"),
    ("Fatima Bello", "Flat 4A"), ("Taiwo Adekunle", "Flat 5B"), ("Kehinde Adekunle", "Flat 5B"),
]

# Retry window for live captures: a Row-2 "fresh capture" pending from the previous capture.
LIVE_RETRY_WINDOW = 90.0

# Base kiosk CSS (premium layer lives in ui/frontend.py, sentry HUD CSS in ui/sentry.py).
CSS = '\n.gradio-container{max-width:1500px!important;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif!important}\n.ng-banner,.ng-banner *{color:#fff!important}\n#ng-header{display:flex;align-items:center;gap:14px;padding:6px 2px 2px}\n#ng-header .logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#111827,#b91c1c);display:grid;place-items:center;color:#fff;font-size:22px}\n#ng-header h1{font-size:20px;margin:0;font-weight:700;letter-spacing:-.01em}\n#ng-header p{margin:0;font-size:12.5px;opacity:.7}\n.ng-banner{display:flex;gap:18px;align-items:flex-start;border-radius:22px;padding:20px 22px;color:#fff;transition:background .4s;box-shadow:0 10px 30px rgba(0,0,0,.12)}\n.ng-locked,.ng-hold{background:linear-gradient(135deg,#1f2937,#111827)}\n.ng-hold{background:linear-gradient(135deg,#78350f,#1f2937)}\n.ng-scanning{background:linear-gradient(135deg,#1e3a8a,#0f172a)}\n.ng-unlocked{background:linear-gradient(135deg,#047857,#064e3b)}\n.ng-alert{background:linear-gradient(135deg,#b91c1c,#450a0a);animation:ngflash 1.2s ease-in-out 2}\n@keyframes ngflash{50%{filter:brightness(1.35)}}\n.ng-lock{font-size:46px;line-height:1;width:74px;height:74px;border-radius:20px;background:rgba(255,255,255,.12);display:grid;place-items:center;flex:none}\n.ng-scanning .ng-lock{animation:ngpulse 1.1s ease-in-out infinite}\n.ng-banner-body{flex:1;min-width:0}\n.ng-banner-title{font-size:26px;font-weight:800;letter-spacing:.04em}\n.ng-banner-sub{font-size:14px;opacity:.85;margin-top:2px}\n.ng-chal{margin-top:10px;display:inline-block;background:rgba(255,255,255,.14);padding:6px 12px;border-radius:999px;font-size:14px}\n.ng-chal b{letter-spacing:.35em;font-size:16px;margin-left:6px}\n.ng-cot{margin-top:12px;background:rgba(0,0,0,.22);border-radius:14px;padding:10px 12px;font-size:14px;line-height:1.45}\n.ng-cot span{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.12em;opacity:.65;margin-bottom:3px}\n.ng-tts{margin-top:8px;font-size:14px;font-style:italic;opacity:.95}\n.ng-stepper{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:14px}\n.ng-arrow{display:none}\n.ng-step{min-width:0;border-radius:16px;padding:12px;background:#f8fafc;border:1.5px solid #e2e8f0;color:#0f172a;transition:all .3s}\n.ng-step-active{border-color:#3b82f6;background:#eff6ff;box-shadow:0 0 0 4px rgba(59,130,246,.15)}\n.ng-step-pass{border-color:#10b981;background:#ecfdf5}\n.ng-step-fail{border-color:#ef4444;background:#fef2f2}\n.ng-step-warn{border-color:#f59e0b;background:#fffbeb}\n.ng-step-skip{opacity:.55}\n.ng-step-head{display:flex;gap:8px;align-items:center}\n.ng-num{width:26px;height:26px;border-radius:50%;background:#e2e8f0;display:grid;place-items:center;font-weight:700;font-size:13px;flex:none}\n.ng-step-active .ng-num{background:#3b82f6;color:#fff;animation:ngpulse 1s infinite}\n.ng-step-pass .ng-num{background:#10b981;color:#fff}.ng-step-fail .ng-num{background:#ef4444;color:#fff}.ng-step-warn .ng-num{background:#f59e0b;color:#fff}\n@keyframes ngpulse{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.55)}50%{box-shadow:0 0 0 8px rgba(59,130,246,0)}}\n.ng-step-title{font-weight:700;font-size:13px;line-height:1.15}\n.ng-step-head{flex-wrap:wrap}\n.ng-step-exp{font-size:10.5px;opacity:.6}\n.ng-badge{margin-left:auto;font-size:9.5px;font-weight:800;padding:3px 7px;border-radius:999px;letter-spacing:.06em;flex:none}\n.ng-b-active{background:#3b82f6;color:#fff}.ng-b-pass{background:#10b981;color:#fff}.ng-b-fail{background:#ef4444;color:#fff}\n.ng-b-warn{background:#f59e0b;color:#fff}.ng-b-skip{background:#cbd5e1;color:#334155}\n.ng-step-sum{font-size:12px;margin:8px 0 8px;min-height:32px;line-height:1.35}\n.ng-chips{display:flex;flex-wrap:wrap;gap:4px}\n.ng-chip{font-size:10.5px;padding:2px 7px;border-radius:999px;background:#e2e8f0;color:#475569}\n.ng-chip-active{background:#3b82f6;color:#fff}.ng-chip-pass{background:#d1fae5;color:#065f46}\n.ng-chip-fail{background:#fee2e2;color:#991b1b}.ng-chip-warn{background:#fef3c7;color:#92400e}.ng-chip-skip{opacity:.5}\n.ng-compare{margin-top:14px;display:grid;grid-template-columns:auto 1fr;gap:18px;border-radius:20px;padding:16px;background:#fff;border:1.5px solid #e2e8f0;color:#0f172a}\n.ng-compare-idle{display:block;color:#64748b;font-size:13.5px;text-align:center;padding:26px}\n.ng-pair{display:flex;align-items:center;gap:10px}\n.ng-pair figure{margin:0;text-align:center}\n.ng-pair img{width:150px;height:150px;border-radius:16px;object-fit:cover;border:1px solid #e2e8f0}\n.ng-pair figcaption{font-size:11.5px;margin-top:5px;color:#475569;max-width:150px}\n.ng-vs{font-weight:800;color:#94a3b8}\n.ng-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}\n.ng-kpis div{background:#f1f5f9;border-radius:12px;padding:8px 10px}\n.ng-kpis span{display:block;font-size:10.5px;color:#64748b}.ng-kpis b{font-size:18px}\n.ng-shap-title{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#64748b;margin:4px 0 6px}\n.ng-bar{display:grid;grid-template-columns:86px 1fr 46px;align-items:center;gap:8px;font-size:12.5px;margin-bottom:5px}\n.ng-bar-t{height:9px;background:#f1f5f9;border-radius:999px;overflow:hidden}\n.ng-bar-f{display:block;height:100%;background:linear-gradient(90deg,#10b981,#059669);border-radius:999px}\n.ng-bar-f.ng-neg{background:linear-gradient(90deg,#f87171,#dc2626)}\n.ng-bar-v{text-align:right;font-variant-numeric:tabular-nums}\n.ng-flip{margin-top:9px;font-size:12.5px;background:#f0fdf4;border-radius:10px;padding:7px 10px;color:#14532d}\n.ng-flip.ng-fragile{background:#fffbeb;color:#78350f}\n.ng-muted{color:#94a3b8;font-size:12.5px}\n.ng-scn button,.ng-scn{text-align:left!important;justify-content:flex-start!important;border-radius:14px!important;font-size:13px!important;\n  padding:10px 12px!important;white-space:normal!important;line-height:1.3!important;min-height:44px}\n.ng-zone-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;opacity:.6;margin:6px 0 2px}\n@media (max-width:900px){.ng-stepper{grid-template-columns:repeat(2,minmax(0,1fr))}.ng-compare{grid-template-columns:1fr}.ng-kpis{grid-template-columns:repeat(2,1fr)}}\n'
