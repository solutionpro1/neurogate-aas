"""NSFRS — Neuro-Symbolic Face Recognition System (Stages 1-3 inside AAS Stage 3).

Stage 1 neural evidence  : AdaFace/ArcFace 1:N + shortlist, three VLM verifier branches (JSON).
Stage 2 XAI + fusion     : Grad-CAM (or occlusion), exact SHAP, Captum-style IG, Alibi flip
                           distance, graph-structured belief fusion.
Stage 3 symbolic         : Clingo #count over match(p, I) → cardinality C.

All functions receive the Engine explicitly (dependency injection: no import cycles).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from neurogate.core.config import ATTRS, ATTR_LABEL, CFG
from neurogate.core.utils import logit, seed_of, sigmoid
from neurogate.models.face import region_raw
from neurogate.pipeline.symbolic import NSFRS_RULES, asp_solve, py_nsfrs

if TYPE_CHECKING:  # pragma: no cover
    from neurogate.pipeline.engine import Engine, Resident


# =============================================================================
# Stage 1 — three VLM branches
# =============================================================================
class VLMBranch:
    """One VLM verifier branch. Real mode prompts an HF image-text-to-text model for JSON;
    offline mode scores the same four attributes from aligned regional descriptors."""

    def __init__(self, name: str, hf_id: str, weights: List[float], jitter: float, gain: float = 9.0):
        self.name, self.hf_id, self.w, self.jitter, self.gain = name, hf_id, np.array(weights), jitter, gain
        self._pipe = None
        self.real = os.environ.get("NEUROGATE_REAL_VLM") == "1"

    def p_same(self, att: np.ndarray) -> float:
        return float(sigmoid(self.gain * (float(self.w @ att) - 0.5)))

    def noise(self, probe_key: str, cand: str) -> np.ndarray:
        r = np.random.default_rng(seed_of(self.name, probe_key, cand))
        return r.normal(0, self.jitter, len(ATTRS))

    def _real(self, probe: Image.Image, ref: Image.Image) -> Optional[dict]:
        try:
            if self._pipe is None:
                from transformers import pipeline  # type: ignore
                self._pipe = pipeline("image-text-to-text", model=self.hf_id, device_map="auto")
            prompt = ("Image 1 is a live door-camera face, image 2 an enrolled resident. Compare them. Reply ONLY JSON: "
                      '{"verdict":"same_person|different_person","confidence":0-100,"attributes":{"periocular_eyes":0-1,'
                      '"nasal_bridge":0-1,"jaw_mouth":0-1,"craniofacial_proportions":0-1},"rationale":"..."}')
            msgs = [{"role": "user", "content": [{"type": "image", "image": probe}, {"type": "image", "image": ref},
                                                 {"type": "text", "text": prompt}]}]
            out = self._pipe(text=msgs, max_new_tokens=200)[0]["generated_text"]
            out = out[-1]["content"] if isinstance(out, list) else out
            return json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        except Exception as e:  # pragma: no cover
            print(f"[VLM {self.name}] real call failed, using regional scorer:", e)
            self.real = False
            return None

    def compare(self, probe_key: str, base_att: np.ndarray, cand: "Resident", probe_img=None) -> Tuple[dict, np.ndarray]:
        if self.real and probe_img is not None:
            j = self._real(probe_img, cand.aligned)
            if j and "attributes" in j:
                att = np.array([float(j["attributes"].get(k, 0.5)) for k in ATTRS])
                j.update(model=self.name, candidate=cand.name, backend="hf:" + self.hf_id)
                return j, att
        att = np.clip(base_att + self.noise(probe_key, cand.stem), 0, 1)
        p = self.p_same(att)
        hi, lo = ATTRS[int(np.argmax(att))], ATTRS[int(np.argmin(att))]
        verdict = "same_person" if p >= 0.5 else "different_person"
        rationale = (f"{ATTR_LABEL[hi]} region strongly consistent; {ATTR_LABEL[lo].lower()} weakest."
                     if verdict == "same_person" else f"{ATTR_LABEL[lo]} region differs; {ATTR_LABEL[hi].lower()} only partly similar.")
        return {"model": self.name, "candidate": cand.name, "verdict": verdict,
                "confidence": int(round(100 * max(p, 1 - p))),
                "attributes": {k: round(float(a), 3) for k, a in zip(ATTRS, att)},
                "rationale": rationale, "backend": "regional-descriptor scorer (offline)"}, att


VLMS = [
    VLMBranch("Qwen3-VL-4B", "Qwen/Qwen3-VL-4B-Instruct", [0.35, 0.25, 0.20, 0.20], 0.03),
    VLMBranch("InternVL3.5-4B", "OpenGVLab/InternVL3_5-4B", [0.25, 0.30, 0.25, 0.20], 0.035),
    VLMBranch("SmolVLM2 / Idefics3", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", [0.30, 0.20, 0.30, 0.20], 0.05),
]


# =============================================================================
# Stage 2 — graph fusion + XAI
# =============================================================================
FUSION_W = {"adaface": 0.5, "vlm": 0.3, "attr": 0.2, "temp": 0.035}


def fuse_belief(s_face: float, att_by_branch: np.ndarray, tau_f: float) -> Tuple[float, dict]:
    """Message passing on the evidence graph:
    attribute nodes → VLM branch nodes → resident belief node ← AdaFace similarity node."""
    p_b = np.array([b.p_same(att_by_branch[i]) for i, b in enumerate(VLMS)])
    l_face = (s_face - tau_f) / FUSION_W["temp"]
    l_vlm = float(np.mean(logit(p_b)))
    l_attr = 6.0 * (float(att_by_branch.mean()) - 0.5)
    L = FUSION_W["adaface"] * l_face + FUSION_W["vlm"] * l_vlm + FUSION_W["attr"] * l_attr
    return float(100 * sigmoid(L)), {"l_face": l_face, "l_vlm": l_vlm, "l_attr": l_attr, "p_branch": p_b.round(3).tolist()}


def shapley(s_face, att_raw: np.ndarray, tau_f) -> np.ndarray:
    """Exact Shapley values of the 4 facial attributes on Bel_i (baseline = neutral 0.5)."""
    from itertools import combinations
    n = len(ATTRS)

    def v(S):
        a = np.full_like(att_raw, 0.5)
        for k in S:
            a[:, k] = att_raw[:, k]
        return fuse_belief(s_face, a, tau_f)[0]

    phi = np.zeros(n)
    for k in range(n):
        others = [j for j in range(n) if j != k]
        for size in range(n):
            for S in combinations(others, size):
                wgt = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                phi[k] += wgt * (v(S + (k,)) - v(S))
    return phi


def integrated_gradients(s_face, att_raw: np.ndarray, tau_f, steps: int = 32) -> np.ndarray:
    """Captum-style Integrated Gradients over the attribute inputs (finite-difference gradients)."""
    base = np.full_like(att_raw, 0.5)
    ig = np.zeros(len(ATTRS))
    eps = 1e-3
    for a in (np.arange(steps) + 0.5) / steps:
        x = base + a * (att_raw - base)
        for k in range(len(ATTRS)):
            xp = x.copy(); xp[:, k] += eps
            xm = x.copy(); xm[:, k] -= eps
            g = (fuse_belief(s_face, xp, tau_f)[0] - fuse_belief(s_face, xm, tau_f)[0]) / (2 * eps)
            ig[k] += g * (att_raw[:, k] - base[:, k]).mean() / steps
    return ig


def alibi_flip(e_p: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float:
    """Minimal L2 move of the probe embedding along (g2-g1) that makes i_2nd outrank i* (closed form)."""
    d = g1 - g2
    return float(max(0.0, e_p @ d) / (np.linalg.norm(d) + 1e-9))


def occlusion_saliency(fb, aligned: Image.Image, target: np.ndarray, patch=None, stride=None) -> np.ndarray:
    if patch is None:  # heavier real backends get a coarser grid (49 forward passes)
        patch, stride = (40, 24) if getattr(fb, "is_real", False) else (20, 10)
    a = np.asarray(aligned.convert("RGB"), dtype=np.float32)
    base = float(fb.embed(aligned)[0] @ target)
    heat = np.zeros((112, 112)); cnt = np.zeros((112, 112)) + 1e-6
    fill = a.mean((0, 1))
    spots, imgs = [], []
    for y in range(0, 112 - patch + 1, stride):
        for x in range(0, 112 - patch + 1, stride):
            b = a.copy()
            b[y:y + patch, x:x + patch] = fill
            spots.append((y, x))
            imgs.append(Image.fromarray(b.astype(np.uint8)))
    E = fb.embed_batch(imgs) if hasattr(fb, "embed_batch") else np.stack([fb.embed(im)[0] for im in imgs])
    for (y, x), e in zip(spots, E):
        heat[y:y + patch, x:x + patch] += max(0.0, base - float(e @ target))
        cnt[y:y + patch, x:x + patch] += 1
    h = heat / cnt
    return h / (h.max() + 1e-9)


def gradcam_overlay(fb, aligned: Image.Image, target: np.ndarray) -> Tuple[Image.Image, str]:
    if getattr(fb, "is_real", False) and hasattr(fb, "gradcam"):
        try:
            cam, label = fb.gradcam(aligned, target), "Grad-CAM (AdaFace body[-1])"
        except Exception:
            cam, label = occlusion_saliency(fb, aligned, target), "Grad-CAM (occlusion fallback)"
    else:
        cam = occlusion_saliency(fb, aligned, target)
        label = ("Grad-CAM (occlusion-sensitivity; ONNX model exposes no gradients)" if getattr(fb, "is_real", False)
                 else "Grad-CAM (occlusion-sensitivity, surrogate backend)")
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
    x = cam[..., None]
    jet = np.concatenate([np.clip(1.5 - np.abs(4 * x - 3), 0, 1), np.clip(1.5 - np.abs(4 * x - 2), 0, 1),
                          np.clip(1.5 - np.abs(4 * x - 1), 0, 1)], -1)
    base = np.asarray(aligned.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
    alpha = 0.25 + 0.35 * x
    out = (1 - alpha) * base + alpha * jet
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)), label


# =============================================================================
# Orchestration of NSFRS Stages 1-3
# =============================================================================
@dataclass
class NSFRSResult:
    e_p: np.ndarray
    q: float
    cos: np.ndarray
    shortlist: List[int]
    vlm_json: List[dict] = field(default_factory=list)
    att: Dict[int, np.ndarray] = field(default_factory=dict)
    beliefs: Dict[int, float] = field(default_factory=dict)
    fusion: Dict[int, dict] = field(default_factory=dict)
    shap: Optional[np.ndarray] = None
    ig: Optional[np.ndarray] = None
    flip: Optional[float] = None
    flip_to: Optional[int] = None
    fragile: bool = False
    cam: Optional[Image.Image] = None
    cam_label: str = ""
    match: List[int] = field(default_factory=list)
    C: int = 0
    asp_atoms: List[str] = field(default_factory=list)
    asp_engine: str = ""
    asp_program: str = ""
    top: Optional[int] = None


def nsfrs_neural(eng: "Engine", aligned: Image.Image, emb=None) -> NSFRSResult:
    """`emb` = (embedding, norm) already computed for this frame in Stage 1 (avoids a second inference)."""
    e_p, q = emb if emb is not None else eng.face_backend.embed(aligned)
    cos = eng.G @ e_p if eng.residents else np.zeros(0)
    order = list(np.argsort(-cos))
    short = [int(i) for i in order if cos[i] >= eng.calib["tau_low"]][: int(CFG["shortlist_k"])]
    res = NSFRSResult(e_p=e_p, q=q, cos=cos, shortlist=short)
    probe_key = hashlib.md5(np.asarray(aligned.resize((32, 32))).tobytes()).hexdigest()
    reg = eng.center_regions(region_raw(aligned))
    for i in short:
        cand = eng.residents[i]
        base = np.array([float(sigmoid((float(reg[k] @ cand.reg[k]) - eng.calib["regions"][k][0]) / eng.calib["regions"][k][1]))
                         for k in ATTRS])
        rows = []
        for b in VLMS:
            j, att = b.compare(probe_key, base, cand, aligned)
            res.vlm_json.append(j)
            rows.append(att)
        res.att[i] = np.stack(rows)
    return res


def nsfrs_xai_fusion(eng: "Engine", res: NSFRSResult, aligned: Image.Image, with_cam: bool = True):
    tau_f = eng.calib["tau_f"]
    for i in res.shortlist:
        res.beliefs[i], res.fusion[i] = fuse_belief(float(res.cos[i]), res.att[i], tau_f)
    order = list(np.argsort(-res.cos)) if len(res.cos) else []
    ranked = sorted(res.beliefs, key=lambda i: -res.beliefs[i]) if res.beliefs else ([int(order[0])] if order else [])
    if not ranked:
        return
    top = ranked[0]
    res.top = top
    second = ranked[1] if len(ranked) > 1 else next((int(i) for i in order if int(i) != top), None)
    if second is not None:
        res.flip = alibi_flip(res.e_p, eng.residents[top].face_e, eng.residents[second].face_e)
        res.flip_to = second
        if res.flip < CFG["frag_delta"] and top in res.beliefs:
            pen = CFG["frag_penalty"] * (1 - res.flip / CFG["frag_delta"])
            floor = res.beliefs.get(second, 0.0)
            res.beliefs[top] = max(floor, res.beliefs[top] - pen)
            res.fragile = True
    if top in res.att:
        res.shap = shapley(float(res.cos[top]), res.att[top], tau_f)
        res.ig = integrated_gradients(float(res.cos[top]), res.att[top], tau_f)
    if with_cam:
        nsfrs_gradcam(eng, res, aligned)


def nsfrs_gradcam(eng: "Engine", res: NSFRSResult, aligned: Image.Image):
    """Grad-CAM / occlusion heatmap for the evidence card. Purely explanatory (it does not feed
    the belief or the gate), so the AAS renders it after the door decision."""
    if res.top is None:
        return
    res.cam, res.cam_label = gradcam_overlay(eng.face_backend, aligned, eng.residents[res.top].face_e)


def nsfrs_symbolic(eng: "Engine", res: NSFRSResult):
    theta = int(round(CFG["theta"]))
    bel_int = {i: int(round(b)) for i, b in res.beliefs.items()}
    facts = "probe(p).\n" + f"theta({theta}).\n" + "".join(f"belief(p,{i},{b}).  % {eng.residents[i].name}\n"
                                                            for i, b in bel_int.items())
    atoms, engine, program = asp_solve(NSFRS_RULES, facts, lambda: py_nsfrs(bel_int, theta))
    res.asp_atoms, res.asp_engine, res.asp_program = atoms, engine, program
    res.match = sorted(int(m.group(1)) for a in atoms for m in [re.match(r"match\(p,(\d+)\)", a)] if m)
    res.C = len(res.match)
