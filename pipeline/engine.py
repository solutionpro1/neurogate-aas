"""AAS engine: closed-set gallery (NSFRS Stage 0 enrolment + calibration) and the
streaming session orchestrator for AAS Stages 1-5.

The orchestrator yields a `View` (pure state). It never builds HTML and never imports
Gradio: ui/frontend.py renders the state and app.py adapts it to components.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from neurogate.core.config import (ATTRS, AUDIO_DIR, CFG, COHORT_DIR, DEMO_RESIDENTS, GALLERY_DIR, IMG_EXT,
                                   LIVE_RETRY_WINDOW, SR)
from neurogate.core.database import DB
from neurogate.core.utils import (augment_face, augment_voice, face_params, hf_ratio, l2n, name_from_stem, now_iso,
                                  read_wav, render_face, robust_fit, seed_of, sigmoid, stem_from_name, synth_voice,
                                  to_mono16k, voice_params, write_wav)
from neurogate.models.face import FaceDetector, load_face_backend, region_raw
from neurogate.models.voice import ASR, extract_digits, load_voice_backend
from neurogate.pipeline.environment import assess_acoustics, prepare_frame
from neurogate.pipeline.nsfrs import (VLMS, NSFRSResult, nsfrs_gradcam, nsfrs_neural, nsfrs_symbolic,
                                      nsfrs_xai_fusion)
from neurogate.pipeline.symbolic import COT, GATE_RULES, PERSONA, asp_solve, py_gate


# =============================================================================
# NSFRS Stage 0 — gallery, enrolment, closed-set calibration
# =============================================================================
@dataclass
class Resident:
    idx: int
    stem: str
    name: str
    unit: str
    img_path: str
    wav_path: Optional[str]
    synthetic: bool
    photo: Image.Image = None
    aligned: Image.Image = None
    face_raw: np.ndarray = None
    face_e: np.ndarray = None
    face_q: float = 0.0
    reg_raw: Dict[str, np.ndarray] = None
    reg: Dict[str, np.ndarray] = None
    wav: Optional[np.ndarray] = None
    voice_raw: Optional[np.ndarray] = None
    voice_e: Optional[np.ndarray] = None


class Engine:
    """Holds backends, the enrolled closed set and its calibration (NSFRS Stage 0)."""

    def __init__(self):
        self.db = DB()
        self.detector = FaceDetector()
        self.face_backend = load_face_backend()
        self.voice_backend = load_voice_backend()
        self.asr = ASR()
        self.residents: List[Resident] = []
        self.reg_mu: Dict[str, np.ndarray] = {}
        self.calib: Dict = {}
        self.cohort_f: List[np.ndarray] = []
        self.cohort_v: List[np.ndarray] = []
        self.lock = threading.RLock()
        self.pyg = self._probe_pyg()

    # -- backends ---------------------------------------------------------
    @staticmethod
    def _probe_pyg():
        try:
            import torch_geometric  # type: ignore  # noqa
            return True
        except Exception:
            return False

    # -- capture ------------------------------------------------------------
    def capture(self, img: Image.Image, synthetic: bool = False, prealigned: bool = False):
        """Returns (aligned 112x112, n_faces_detected, detector label)."""
        img = img.convert("RGB")
        if prealigned:
            return img.resize((112, 112), Image.BILINEAR), 1, "pre-aligned crop"
        if synthetic:
            return self.detector.align(img, None), 1, "synthetic frame (full-frame crop)"
        if hasattr(self.face_backend, "detect_align"):
            try:
                al, n = self.face_backend.detect_align(img)
                if al is not None:
                    return al, n, f"SCRFD + 5-point norm_crop ({n} face{'s' if n > 1 else ''})"
            except Exception as e:
                print("[capture] SCRFD failed:", e)
        boxes = self.detector.detect(img)
        if not boxes:
            return self.detector.align(img, None), 0, "no face box → centre-crop fallback"
        box = max(boxes, key=lambda b: b[2] * b[3])
        return self.detector.align(img, box), len(boxes), f"Haar + eye-line alignment ({len(boxes)} face box{'es' if len(boxes) > 1 else ''})"

    # -- enrolment ----------------------------------------------------------
    def load_gallery(self):
        with self.lock:
            files = sorted(f for f in os.listdir(GALLERY_DIR) if f.lower().endswith(IMG_EXT))
            res: List[Resident] = []
            for f in files:
                stem = os.path.splitext(f)[0]
                path = os.path.join(GALLERY_DIR, f)
                try:
                    photo = Image.open(path)
                    synthetic = photo.info.get("neurogate_synthetic") == "1"
                    photo = photo.convert("RGB")
                except Exception as e:
                    print("[gallery] skip", f, e)
                    continue
                wav = os.path.join(AUDIO_DIR, stem + ".wav")
                name = name_from_stem(stem)
                unit = self.db.unit_of(stem) or dict(DEMO_RESIDENTS).get(name, "Unit —")
                r = Resident(len(res), stem, name, unit, path, wav if os.path.exists(wav) else None, synthetic, photo=photo)
                r.aligned, _, _ = self.capture(photo, synthetic)
                r.face_raw = self.face_backend.raw(r.aligned)
                r.reg_raw = region_raw(r.aligned)
                if r.wav_path:
                    try:
                        r.wav = read_wav(r.wav_path)
                        r.voice_raw = self.voice_backend.raw(r.wav)
                    except Exception as e:
                        print("[gallery] bad wav", wav, e)
                        r.wav_path = None
                res.append(r)
            self.residents = res
            self._score_cache = None   # gallery changed → recollect calibration score samples
            # background cohort + cohort normalisation (surrogates), then embeddings
            c_faces, c_wavs = self._cohort_raw(any(r.synthetic for r in res))
            self.face_backend.set_cohort([r.face_raw for r in res] + [self.face_backend.raw(a) for a in c_faces])
            self.voice_backend.set_cohort([r.voice_raw for r in res if r.voice_raw is not None]
                                          + [self.voice_backend.raw(w) for w in c_wavs])
            self.cohort_f = [self.face_backend.embed(a)[0] for a in c_faces]
            self.cohort_v = [self.voice_backend.embed(w) for w in c_wavs]
            self.reg_mu = {k: np.mean([r.reg_raw[k] for r in res], axis=0) if len(res) >= 2 else 0.0 for k in ATTRS}
            for r in res:
                r.face_e, r.face_q = self.face_backend.embed(r.aligned)
                r.reg = self.center_regions(r.reg_raw)
                if r.wav is not None:
                    r.voice_e = self.voice_backend.embed(r.wav)
                self.db.upsert_resident(r)
            self.db.prune([r.stem for r in res])
            self.calibrate()

    def center_regions(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: l2n(raw[k] - self.reg_mu.get(k, 0.0)) for k in ATTRS}

    @property
    def G(self) -> np.ndarray:
        return np.stack([r.face_e for r in self.residents]) if self.residents else np.zeros((0, 512))

    def calibrate(self):
        """Closed-set calibration. Per-pair false-accept rates scale with 1/N:
        f = FAR_face/N, v = FAR_voice/N; joint independent system FAR = N·f·v."""
        res = self.residents
        N = len(res)
        c: Dict = {"N": N, "notes": []}
        cache = getattr(self, "_score_cache", None)
        if cache is None:  # score samples depend on the gallery only, not on thresholds → computed once per load
            cf, cv = self.cohort_f, self.cohort_v
            imp_f, gen_f, imp_v, gen_v = [], [], [], []
            imp_r = {k: [] for k in ATTRS}
            gen_r = {k: [] for k in ATTRS}
            for i in range(N):
                for j in range(i + 1, N):
                    imp_f.append(float(res[i].face_e @ res[j].face_e))
                    for k in ATTRS:
                        imp_r[k].append(float(res[i].reg[k] @ res[j].reg[k]))
                    if res[i].voice_e is not None and res[j].voice_e is not None:
                        imp_v.append(float(res[i].voice_e @ res[j].voice_e))
            for r in res:
                imp_f.extend(float(r.face_e @ e) for e in cf)
                if r.voice_e is not None:
                    imp_v.extend(float(r.voice_e @ e) for e in cv)
            for r in res:
                for a in range(3):
                    al, _, _ = self.capture(augment_face(r.photo, seed_of("cal", r.stem, a)), r.synthetic)
                    e, _ = self.face_backend.embed(al)
                    gen_f.append(float(e @ r.face_e))
                    rg = self.center_regions(region_raw(al))
                    for k in ATTRS:
                        gen_r[k].append(float(rg[k] @ r.reg[k]))
                    if r.wav is not None and a < 2:
                        gen_v.append(float(self.voice_backend.embed(augment_voice(r.wav, seed_of("calv", r.stem, a))) @ r.voice_e))
            self._score_cache = cache = (imp_f, gen_f, imp_v, gen_v, imp_r, gen_r, (len(cf), len(cv)))
        imp_f, gen_f, imp_v, gen_v, imp_r, gen_r, c["cohort"] = cache
        Nn = max(N, 1)
        f = CFG["far_face"] / Nn
        v = CFG["far_voice"] / Nn
        zf, zv = NormalDist().inv_cdf(1 - f), NormalDist().inv_cdf(1 - v)
        mi, si = robust_fit(imp_f, 0.03)
        mg, sg = robust_fit(gen_f, 0.02)
        tau_f = mi + zf * si
        if gen_f and tau_f > mg - 1.0 * sg:
            capped = (mi + mg) / 2
            c["notes"].append(f"Face τ_f from impostor tail ({tau_f:.3f}) exceeded genuine median − σ; capped to midpoint {capped:.3f}. "
                              "Enrol more/better references or relax FAR budget.")
            tau_f = capped
        tau_low = min(mi + 1.0 * si, tau_f - 0.05)
        mvi, svi = robust_fit(imp_v, 0.03)
        mvg, svg = robust_fit(gen_v, 0.02)
        tau_v = mvi + zv * svi
        if gen_v and tau_v > mvg - 1.0 * svg:
            capped = (mvi + mvg) / 2
            c["notes"].append(f"Voice τ_v from impostor tail ({tau_v:.3f}) exceeded genuine median − σ; capped to {capped:.3f}.")
            tau_v = capped
        regions = {}
        for k in ATTRS:
            m_i, _ = robust_fit(imp_r[k], 0.02)
            m_g, _ = robust_fit(gen_r[k], 0.02)
            if m_g <= m_i + 0.05:
                m_g = m_i + 0.2
            regions[k] = ((m_i + m_g) / 2, max((m_g - m_i) / 6, 0.02))
        c.update(dict(f=f, v=v, system_far=N * f * v, tau_f=tau_f, tau_low=tau_low, tau_v=tau_v,
                      imp_face=(mi, si), gen_face=(mg, sg), imp_voice=(mvi, svi), gen_voice=(mvg, svg),
                      regions=regions, n_imp_pairs=len(imp_f)))
        self.calib = c

    def _cohort_raw(self, synthetic_gallery: bool):
        """Background impostor cohort (never enrolled): files in cohort/ if present; when the gallery
        holds synthetic demo residents, a held-out synthetic cohort of 40 faces/voices is added."""
        faces, voices = [], []
        for f in sorted(os.listdir(COHORT_DIR)):
            p = os.path.join(COHORT_DIR, f)
            try:
                if f.lower().endswith(IMG_EXT):
                    im = Image.open(p)
                    syn = im.info.get("neurogate_synthetic") == "1"
                    faces.append(self.capture(im.convert("RGB"), syn)[0])
                elif f.lower().endswith(".wav"):
                    voices.append(read_wav(p))
            except Exception as e:
                print("[cohort] skip", f, e)
        if synthetic_gallery:
            for k in range(40):
                faces.append(self.capture(render_face(face_params(seed_of("cohort", k))), True)[0])
                voices.append(synth_voice(voice_params(seed_of("cohort-voice", k)), k))
        return faces, voices

    def enrol(self, name: str, unit: str, photo: Image.Image, audio) -> str:
        stem = stem_from_name(name)
        for f in os.listdir(GALLERY_DIR):
            if os.path.splitext(f)[0] == stem:
                os.remove(os.path.join(GALLERY_DIR, f))
        photo.convert("RGB").save(os.path.join(GALLERY_DIR, stem + ".jpg"), quality=95)
        if audio is not None:
            sr, data = audio
            write_wav(os.path.join(AUDIO_DIR, stem + ".wav"), to_mono16k(sr, data))
        with self.db._c() as c:
            c.execute("INSERT OR REPLACE INTO residents(stem,name,unit,img_path,enrolled_at) VALUES(?,?,?,?,?)",
                      (stem, name.strip(), unit.strip() or "Unit —", "", now_iso()))
        self.load_gallery()
        return stem

    def by_name(self, name: str) -> Optional[Resident]:
        return next((r for r in self.residents if r.name == name), None)


_ENGINE: Optional[Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    """Process-wide Engine singleton (models load once)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()
            PERSONA.last_visit = _ENGINE.db.last_visit
        return _ENGINE


# =============================================================================
# AAS Stage 2 — voice interaction & liveness
# =============================================================================
@dataclass
class VoiceResult:
    mode: str
    transcript: str
    claim: Optional[str]
    scores: Dict[int, float]
    top: Optional[int]
    score: float
    margin: float
    heard: str
    digits_ok: Optional[bool]
    energy: float
    hf: float
    live: bool
    reasons: List[str]


def voice_stage(eng: Engine, wav: Optional[np.ndarray], transcript: Optional[str], challenge: str, mode: str,
                typed_code: str = "") -> VoiceResult:
    if wav is None or len(wav) < SR // 4:
        return VoiceResult(mode, "", None, {}, None, 0.0, 0.0, "", False, 0.0, 0.0, False, ["No speech captured"])
    if transcript is None:
        transcript = eng.asr.transcribe(wav)
        if transcript is None:
            transcript = f"[ASR offline] code typed by resident: {typed_code}" if typed_code else "[ASR offline — no transcript]"
    names = [r.name for r in eng.residents]
    claim = None
    m = re.search(r"(?:name is|i am|i'm)\s+([a-z][a-z \-']+?)(?:[.,]|$| my)", transcript.lower())
    if m:
        hit = difflib.get_close_matches(m.group(1).strip(), [n.lower() for n in names], n=1, cutoff=0.6)
        claim = names[[n.lower() for n in names].index(hit[0])] if hit else m.group(1).strip().title()
    e = eng.voice_backend.embed(wav)
    scores = {r.idx: float(r.voice_e @ e) for r in eng.residents if r.voice_e is not None}
    ranked = sorted(scores, key=lambda i: -scores[i])
    top = ranked[0] if ranked else None
    s1 = scores[top] if top is not None else 0.0
    s2 = scores[ranked[1]] if len(ranked) > 1 else eng.calib.get("tau_v", 0.0)
    heard = extract_digits(transcript if "[ASR offline" not in transcript else typed_code)
    digits_ok = challenge in heard if heard else False
    energy = float(np.sqrt(np.mean(wav ** 2)))
    hf = hf_ratio(wav)
    reasons = []
    if not digits_ok:
        reasons.append(f"challenge {challenge} not heard (heard '{heard or '—'}')")
    if energy < CFG["energy_floor"]:
        reasons.append(f"speech energy too low ({energy:.3f})")
    if hf < CFG["hf_floor"]:
        reasons.append(f"no >4 kHz content ({hf:.4f}) — loudspeaker/phone replay signature")
    return VoiceResult(mode, transcript, claim, scores, top, s1, s1 - s2, heard, digits_ok, energy, hf, not reasons, reasons)


# =============================================================================
# Session request + kiosk state
# =============================================================================
@dataclass
class Request:
    source: str
    face: Callable[[int], Tuple[Image.Image, bool, bool]]           # attempt -> (img, synthetic, prealigned)
    voice: Callable[[int, str, Optional[int]], Tuple[Optional[np.ndarray], Optional[str], str]]  # (attempt, challenge, face_top) -> (wav, transcript, mode)
    occupancy: Optional[int] = None
    crossings: int = 1
    typed_code: str = ""
    challenge: Optional[str] = None   # set by the browser sentry, which already spoke it aloud
    live: bool = False                 # hands-free browser capture (one attempt per capture)
    nonce: str = ""                    # echoed to the browser so it knows which capture a decision belongs to


class View:
    """Kiosk state streamed to the UI: 5 stage cards, lock banner, evidence, environment."""

    def __init__(self):
        self.st = {n: {"status": "idle", "summary": "Waiting", "sub": ["idle"] * 4} for n in range(1, 6)}
        self.banner = dict(state="locked", title="DOOR LOCKED", sub="Waiting for someone to approach",
                           cot="", tts="", challenge="", who="", say="", say_id="", final=False, retry=False, session="")
        self.evidence: Optional[Tuple[NSFRSResult, Image.Image]] = None
        self.env: Dict[str, object] = {}

    def stage(self, n, status=None, summary=None):
        if status:
            self.st[n]["status"] = status
        if summary is not None:
            self.st[n]["summary"] = summary

    def sub(self, n, k, status):
        self.st[n]["sub"][k] = status

    def say(self, text: str):
        """Queue a sentence for the kiosk speaker (browser speechSynthesis picks up the new id)."""
        self.banner["say"] = text
        self.banner["say_id"] = secrets.token_hex(6)


# =============================================================================
# AAS orchestration — one attempt (Stages 1 → 5)
# =============================================================================
def run_attempt(eng: Engine, req: Request, attempt: int, view: View, retries_left: int):
    trace: Dict = {"source": req.source, "attempt": attempt, "backends": {
        "face": eng.face_backend.name, "voice": eng.voice_backend.name, "asr": eng.asr.name, "cot": COT.name,
        "vlm": [b.hf_id if b.real else b.name + " (offline scorer)" for b in VLMS]}}
    for n in range(1, 6):
        view.stage(n, "idle", "Waiting")
        view.st[n]["sub"] = ["idle"] * 4
    view.banner.update(state="scanning", title="SCANNING…", sub=f"Attempt {attempt} · {req.source}", cot="", tts="", challenge="",
                       final=False, retry=False)
    view.evidence = None
    view.env = {}
    yield view, None

    # ---------------- Stage 1: presence & capture ----------------
    view.stage(1, "active", "LWM tracking arrival…")
    view.sub(1, 0, "active")
    yield view, None
    img, synthetic, pre = req.face(attempt)
    img, light = prepare_frame(img)              # environmental awareness: exposure check + correction
    view.env["light"] = light
    aligned0, n_det, det_label = eng.capture(img, synthetic, pre)
    occupancy = req.occupancy if req.occupancy is not None else max(1, n_det)
    view.sub(1, 0, "pass")
    view.sub(1, 1, "active")
    view.stage(1, summary=f"{occupancy} person{'s' if occupancy > 1 else ''} in door zone")
    yield view, None
    anomaly = occupancy > 1
    view.sub(1, 1, "warn" if anomaly else "pass")
    trace["stage1"] = {"detector": det_label, "faces_detected": n_det, "occupancy": occupancy, "anomaly": anomaly,
                       "lighting": light.as_dict()}

    nsfrs = None
    vr = None
    challenge = req.challenge or "".join(str(secrets.randbelow(10)) for _ in range(4))
    if anomaly:
        view.stage(1, "warn", f"{occupancy} people in zone — queue warning")
        for k in (2, 3):
            view.sub(1, k, "skip")
        for n in (2, 3):
            view.stage(n, "skip", "Paused until one person approaches")
        q, aligned = 0.0, aligned0
        yield view, None
    else:
        view.sub(1, 2, "active")
        yield view, None
        # three consecutive frames: as captured, slight motion blur, slight under-exposure
        frames = [aligned0, aligned0.filter(ImageFilter.GaussianBlur(0.8)),
                  ImageEnhance.Brightness(aligned0).enhance(0.85)]
        embs = [eng.face_backend.embed(f) for f in frames]
        qs = [e[1] for e in embs]
        best = int(np.argmax(qs))
        aligned, q, emb = frames[best], qs[best], embs[best]   # embedding reused by Stage 3
        q_ok = q >= CFG["quality_floor"] and light.usable
        view.sub(1, 2, "pass" if q_ok else "warn")
        view.sub(1, 3, "active")
        view.stage(1, summary=f"Best of 3 frames ‖z‖₂ = {q:.1f}")
        yield view, None
        view.sub(1, 3, "pass")
        view.stage(1, "pass" if q_ok else "warn",
                   f"1 person · {light.icon} {light.label} · ‖z‖₂ {q:.1f} · challenge issued")
        view.banner.update(challenge=challenge, sub=f"Attempt {attempt} · {req.source}")
        if not req.live:  # the live sentry already spoke the prompt and waited for it to finish
            view.say(PERSONA.prompt(challenge, light))
        trace["stage1"].update(frame_quality=[round(x, 2) for x in qs], best_frame=best, challenge=challenge)
        yield view, None

        # -------- Stages 2 & 3 concurrently --------
        quick_top = None
        if eng.residents:
            c0 = eng.G @ emb[0]
            if float(c0.max()) >= eng.calib["tau_low"]:
                quick_top = int(np.argmax(c0))
        wav, transcript, vmode = req.voice(attempt, challenge, quick_top)
        view.stage(2, "active", "Greeting resident…")
        view.stage(3, "active", "NSFRS waking…")
        view.sub(2, 0, "active")
        yield view, None
        pool = ThreadPoolExecutor(max_workers=1)
        vfut = pool.submit(voice_stage, eng, wav, transcript, challenge, vmode, req.typed_code)

        def voice_tick(k):
            for i in range(4):
                if view.st[2]["sub"][i] == "active":
                    view.sub(2, i, "pass")
            if k < 4:
                view.sub(2, k, "active")
                view.stage(2, summary=["Greeting resident…", "Transcribing claim (hint only)…",
                                       f"ECAPA 1:{len(eng.residents)} voiceprint…", "Checking challenge digits…"][k])

        # NSFRS 0
        view.sub(3, 0, "active")
        view.stage(3, summary=f"Closed set of N = {len(eng.residents)} residents")
        yield view, None
        view.sub(3, 0, "pass")
        voice_tick(1)
        view.sub(3, 1, "active")
        view.stage(3, summary="AdaFace embed + shortlist + 3 VLM branches…")
        yield view, None
        nsfrs = nsfrs_neural(eng, aligned, emb)
        view.sub(3, 1, "pass")
        voice_tick(2)
        view.sub(3, 2, "active")
        view.stage(3, summary=f"{len(nsfrs.shortlist)} shortlisted ≥ τ_low; XAI + graph fusion…")
        yield view, None
        nsfrs_xai_fusion(eng, nsfrs, aligned, with_cam=False)   # heatmap deferred until after Stage 5
        view.evidence = (nsfrs, aligned)
        view.sub(3, 2, "pass")
        voice_tick(3)
        view.sub(3, 3, "active")
        view.stage(3, summary="Clingo counting match(p, I)…")
        yield view, None
        nsfrs_symbolic(eng, nsfrs)
        view.evidence = (nsfrs, aligned)
        vr = vfut.result()
        pool.shutdown(wait=False)
        names = ", ".join(eng.residents[i].name for i in nsfrs.match)
        view.sub(3, 3, "pass")
        c_text = {0: "C = 0 · no resident above θ", 1: f"C = 1 · {names}"}.get(nsfrs.C, f"C = {nsfrs.C} · {names}")
        view.stage(3, "pass" if nsfrs.C == 1 else "warn", c_text)
        voice_tick(4)
        vtop = eng.residents[vr.top].name if vr.top is not None else "—"
        view.sub(2, 2, "pass" if vr.top is not None else "fail")
        view.sub(2, 3, "pass" if vr.live else "fail")
        view.stage(2, "pass" if vr.live else "fail",
                   (f"Voice → {vtop} ({vr.score:.2f}, Δv {vr.margin:.2f}) · liveness OK" if vr.live
                    else f"Liveness FAILED · {vr.reasons[0] if vr.reasons else ''}"))
        sound = assess_acoustics(wav)
        view.env["sound"] = sound
        trace["stage2"] = {"acoustics": sound.as_dict(), "mode": vr.mode, "transcript": vr.transcript, "claim_hint": vr.claim,
                           "voice_scores": {eng.residents[i].name: round(s, 4) for i, s in vr.scores.items()},
                           "voice_top": vtop, "margin": round(vr.margin, 4), "heard_digits": vr.heard,
                           "energy": round(vr.energy, 4), "hf_ratio": round(vr.hf, 5), "live": vr.live, "reasons": vr.reasons}
        trace["stage3"] = {"q": round(nsfrs.q, 2), "shortlist": [eng.residents[i].name for i in nsfrs.shortlist],
                           "beliefs": {eng.residents[i].name: round(b, 1) for i, b in nsfrs.beliefs.items()},
                           "fusion": {eng.residents[i].name: f for i, f in nsfrs.fusion.items()},
                           "shap": None if nsfrs.shap is None else dict(zip(ATTRS, np.round(nsfrs.shap, 2).tolist())),
                           "integrated_gradients": None if nsfrs.ig is None else dict(zip(ATTRS, np.round(nsfrs.ig, 2).tolist())),
                           "alibi_flip": nsfrs.flip, "fragile": nsfrs.fragile, "match": [eng.residents[i].name for i in nsfrs.match],
                           "C": nsfrs.C, "asp_engine": nsfrs.asp_engine, "grad_cam": "rendered after the decision"}
        yield view, (nsfrs, vr, aligned)

    # ---------------- Stage 4: cross-modal decision ----------------
    view.stage(4, "active", "Comparing face set F with voice j*…")
    view.sub(4, 0, "active")
    yield view, None
    F = nsfrs.match if nsfrs else []
    C = len(F)
    vtop_i = vr.top if vr else None
    agree = vtop_i is not None and vtop_i in F
    view.sub(4, 0, "pass" if agree else ("warn" if F else "fail"))
    view.sub(4, 1, "active")
    yield view, None
    quality_ok = q >= CFG["quality_floor"] and light.usable   # environment feeds the existing quality_ok fact
    live = bool(vr.live) if vr else True  # anomaly short-circuit: liveness not yet tested
    view.sub(4, 1, "pass" if quality_ok and live else "fail")
    calib = eng.calib
    tv = int(round(calib["tau_v"] * 1000))
    vs = int(round((vr.score if vr else 0) * 1000))
    vm = int(round((vr.margin if vr else 0) * 1000))
    mm = int(round(CFG["voice_margin"] * 1000))
    facts = ["probe(p)."] + [f"match(p,{i}).  % {eng.residents[i].name}" for i in F]
    if vtop_i is not None:
        facts += [f"voice_top(p,{vtop_i}).  % {eng.residents[vtop_i].name}", f"voice_score(p,{vtop_i},{vs})."]
    facts += [f"voice_margin(p,{vm}).", f"tau_v({tv}).", f"margin_min({mm})."]
    if live:
        facts.append("live(p).")
    if quality_ok:
        facts.append("quality_ok(p).")
    if anomaly:
        facts.append("anomaly(p).")
    pyfacts = dict(match=F, voice_top=vtop_i, voice_score=vs, voice_margin=vm, tau_v=tv, margin_min=mm,
                   live=live, quality_ok=quality_ok, anomaly=anomaly)
    view.sub(4, 3, "active")
    atoms, gate_engine, gate_program = asp_solve(GATE_RULES, "\n".join(facts), lambda: py_gate(pyfacts))
    decision = next((m.group(1) for a in atoms for m in [re.match(r"decision\(p,(\w+)\)", a)] if m), "escalate")
    allow_id = next((int(m.group(1)) for a in atoms for m in [re.match(r"allow\(p,(\d+)\)", a)] if m), None)
    retry_exhausted = False
    if decision == "retry" and retries_left <= 0:
        decision, retry_exhausted = "escalate", True
    ctx = dict(match=F, C=C, voice_top=vtop_i, voice_score=vr.score if vr else 0.0, voice_margin=vr.margin if vr else 0.0,
               tau_v=calib["tau_v"], live=live, live_reasons=vr.reasons if vr else [], q=q, quality_ok=quality_ok,
               occupancy=occupancy, allow_id=allow_id, claim=vr.claim if vr else None, retry_exhausted=retry_exhausted,
               light=light, sound=view.env.get("sound"))
    view.sub(4, 2, "active")
    yield view, None
    steps, summary, tts = COT.reason(decision, ctx, eng.residents)
    view.sub(4, 2, "pass")
    view.sub(4, 3, "pass" if decision == "allow" else "fail" if decision == "alert" else "warn")
    dlabel = {"allow": "ALLOW", "retry": "DENY · 1 retry", "host": "ROUTE TO HOST", "escalate": "ESCALATE TO GUARD",
              "alert": "DENY · SECURITY ALERT", "queue": "QUEUE · one at a time"}[decision]
    view.stage(4, "pass" if decision == "allow" else "fail" if decision in ("alert",) else "warn", f"Gate: {dlabel}")
    trace["stage4"] = {"facts": facts, "answer_set": atoms, "engine": gate_engine, "decision": decision,
                       "allow": eng.residents[allow_id].name if allow_id is not None else None, "cot": steps,
                       "retry_exhausted": retry_exhausted}
    yield view, None

    # ---------------- Stage 5: door & anti-tailgate ----------------
    view.stage(5, "active", "Driving lock…" if decision == "allow" else "Holding lock…")
    view.sub(5, 0, "active")
    yield view, None
    door, crossings = "locked", 0
    if decision == "allow":
        view.sub(5, 0, "pass")
        view.sub(5, 1, "active")
        view.banner.update(state="unlocked", title="DOOR UNLOCKED", sub=f"Welcome, {eng.residents[allow_id].name} · {eng.residents[allow_id].unit}",
                           cot=summary, tts=tts, challenge="")
        view.say(tts)
        view.stage(5, summary="VLA actuator ACK · camera confirming open")
        yield view, None
        view.sub(5, 1, "pass")
        view.sub(5, 2, "active")
        view.stage(5, summary="Counting people crossing the threshold…")
        yield view, None
        crossings = req.crossings
        if crossings > 1:
            door = "relocked-tailgate"
            view.sub(5, 2, "fail")
            tts = PERSONA.decision("tailgate", ctx, eng.residents)
            summary = (f"{crossings} people crossed on a single unlock for {eng.residents[allow_id].name}. "
                       "Door relocked; tailgating alert raised and the unknown second person flagged.")
            view.banner.update(state="alert", title="SECURITY ALERT", sub="Tailgating detected — door relocked", cot=summary, tts=tts)
            view.say(tts)
            view.stage(5, "fail", f"{crossings} crossings on 1 unlock → relocked + alert")
            decision_final = "tailgate-alert"
        else:
            door = "opened-closed"
            view.sub(5, 2, "pass")
            view.stage(5, "pass", "1 crossing · door closed & relocked")
            decision_final = "allow"
    else:
        view.sub(5, 0, "skip")
        view.sub(5, 1, "skip")
        view.sub(5, 2, "skip")
        st = {"alert": ("alert", "SECURITY ALERT", "Access denied"), "retry": ("hold", "DOOR LOCKED", "Identity not confirmed — one fresh capture"),
              "host": ("hold", "DOOR LOCKED", "Visitor — host is being notified"), "escalate": ("hold", "DOOR LOCKED", "Guard called to verify in person"),
              "queue": ("hold", "DOOR LOCKED", "One person at a time, please")}[decision]
        view.banner.update(state=st[0], title=st[1], sub=st[2], cot=summary, tts=tts, challenge="")
        view.say(tts)
        view.stage(5, "fail" if decision == "alert" else "warn", {"alert": "Locked · security alerted", "retry": "Locked · awaiting fresh capture",
                                                                   "host": "Locked · host notified", "escalate": "Locked · guard paged",
                                                                   "queue": "Locked · queue prompt"}[decision])
        decision_final = decision
    view.sub(5, 3, "pass")
    view.banner.update(final=True, retry=(decision == "retry"))
    trace["stage5"] = {"door": door, "crossings": crossings}
    eng.db.log(source=req.source, attempt=attempt, face_ids=", ".join(eng.residents[i].name for i in F) or "∅", C=C,
               voice_top=eng.residents[vtop_i].name if vtop_i is not None else "—", live=live, decision=decision_final,
               door=door, crossings=crossings, rationale=summary, trace=trace)
    if nsfrs is not None and nsfrs.top is not None and decision != "retry":   # a retry replaces this evidence at once
        yield view, None                       # decision shown + spoken before the heatmap is drawn
        nsfrs_gradcam(eng, nsfrs, aligned)     # explanatory only: off the verification critical path
    yield view, {"decision": decision, "final": decision_final, "trace": trace, "gate_program": gate_program,
                 "gate_atoms": atoms, "gate_engine": gate_engine, "nsfrs": nsfrs, "vr": vr}


# =============================================================================
# AAS session driver — retries across attempts (scenario) or across captures (live)
# =============================================================================
LIVE_RETRY = {"at": 0.0}


def run_session(eng: Engine, req: Request):
    """Yields (view, final, done). `done` marks the end of an attempt; `final` then carries
    the decision payload (trace, ASP programs, NSFRS/voice results); the trace is also stored in the audit log."""
    view = View()
    view.banner["session"] = req.nonce
    if not eng.residents:
        view.banner.update(state="locked", title="DOOR LOCKED", sub="No residents enrolled — use the Stage 0 tab", final=True)
        view.say("No residents are enrolled yet.")
        yield view, None, True
        return
    if req.live:
        pending = time.time() - LIVE_RETRY["at"] < LIVE_RETRY_WINDOW
        attempts = (2,) if pending else (1,)
    else:
        attempts = (1, 2)
    for attempt in attempts:
        final = None
        for v, payload in run_attempt(eng, req, attempt, view, retries_left=2 - attempt):
            if payload and "decision" in payload:
                final = payload
            yield v, None, False
        yield view, final, True
        is_retry = bool(final and final["decision"] == "retry" and attempt == 1)
        if req.live:
            LIVE_RETRY["at"] = time.time() if is_retry else 0.0
            break
        if is_retry:  # no pause: the browser queues the retry message before the next challenge
            view.banner.update(state="scanning", title="SCANNING…", sub="🔁 Fresh capture — retry 1 of 1", final=False, retry=False)
            yield view, None, False
            continue
        break
