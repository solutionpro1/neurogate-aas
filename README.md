# NeuroGate AAS v3

Explainable multimodal neuro-symbolic access control, following Dr. E. P. Fasina's architecture:
the Autonomous Access System (AAS, Stages 1–5) wrapping the Neuro-Symbolic Face Recognition
System (NSFRS, Stages 0–3).

## Run

```bash
pip install -r neurogate/requirements.txt
python neurogate/app.py          # or: python -m neurogate.app
```

Open **http://localhost:7860** in Chrome or Edge. Press **Start Auto-Sentry** once (browsers
require one click before allowing the camera, microphone and speaker); the kiosk is hands-free after that.

## Layout

| Module | Responsibility |
|---|---|
| `core/config.py` | `CFG` thresholds (incl. environment), paths, constants, base CSS. Thresholds are tuned in the kiosk's ⚙️ accordion |
| `core/database.py` | SQLite residents + Stage 5 audit log, last-visit lookup |
| `core/utils.py` | Math/audio/image helpers, augmentation, synthetic demo residents |
| `models/face.py` | Haar/SCRFD capture, quality, InsightFace / AdaFace / surrogate backends |
| `models/voice.py` | ECAPA-TDNN / surrogate voiceprints, faster-whisper ASR fallback |
| `pipeline/environment.py` | Lighting (dark, dim, backlit, glare, blur) + acoustic (SNR, clipping) awareness |
| `pipeline/nsfrs.py` | 3 VLM branches, Grad-CAM, SHAP, IG, Alibi, graph fusion, Clingo cardinality |
| `pipeline/symbolic.py` | Clingo ASP programs (unchanged), Python evaluator, CoT reasoner, voice persona |
| `pipeline/engine.py` | Engine (gallery, enrolment, calibration) + AAS session orchestrator |
| `ui/frontend.py` | HTML renderers, calibration summary, premium CSS |
| `ui/sentry.py` | `AUTONOMOUS_SENTRY_JS` + camera HUD |
| `ui/scenarios.py` | Decision-table scenarios, environment demos, live payload decoding |
| `app.py` | Gradio Blocks, event wiring, launch |

Dependency direction: `core → models → pipeline → ui → app`. The pipeline never imports
Gradio or builds HTML; NSFRS and the reasoner receive the engine explicitly.

## Environment variables (all optional)

`NEUROGATE_FACE=surrogate` / `NEUROGATE_VOICE=surrogate` force the offline backends ·
`ADAFACE_REPO` + `ADAFACE_CKPT` use AdaFace · `NEUROGATE_ASR=off` disables faster-whisper ·
`NEUROGATE_REAL_VLM=1` calls the Hugging Face VLMs · `NEUROGATE_COT_MODEL=Qwen/Qwen3-8B` uses an LLM
for explanations · `PORT`, `GRADIO_SERVER_NAME`, `GRADIO_SHARE=1`.

## Data

`gallery/Name_Surname.jpg` + `gallery_audio/Name_Surname.wav` (created on first run with six
synthetic demo residents if empty) · optional `cohort/` of non-resident faces/voices for
impostor calibration · `neurogate_aas.db` (SQLite).
