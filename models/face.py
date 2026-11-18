"""Face models: detection/alignment, quality, and the embedding backends.

Backend order (load_face_backend): AdaFace (ADAFACE_REPO + ADAFACE_CKPT) → InsightFace buffalo_l
(auto when installed) → offline HOG surrogate. NEUROGATE_FACE=surrogate forces the surrogate.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from neurogate.core.config import ATTRS, REGIONS
from neurogate.core.utils import hog, l2n

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


class FaceDetector:
    """OpenCV Haar detector with eye-line alignment to a 112x112 crop.
    Production path: MTCNN/RetinaFace 5-point similarity transform (AdaFace protocol)."""

    def __init__(self):
        self.face = self.eye = None
        if cv2 is not None:
            try:
                hc = cv2.data.haarcascades
                fpath = os.path.join(hc, "haarcascade_frontalface_default.xml")
                epath = os.path.join(hc, "haarcascade_eye.xml")
                if os.path.exists(fpath):
                    self.face = cv2.CascadeClassifier(fpath)
                    self.eye = cv2.CascadeClassifier(epath) if os.path.exists(epath) else None
                    if self.face.empty():
                        self.face = None
            except Exception:
                self.face = None

    def detect(self, img: Image.Image) -> List[Tuple[int, int, int, int]]:
        if self.face is None:
            return []
        g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        s = max(1.0, max(g.shape) / 900.0)
        if s > 1:
            g = cv2.resize(g, (int(g.shape[1] / s), int(g.shape[0] / s)))
        boxes = self.face.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48))
        return [tuple(int(v * s) for v in b) for b in boxes] if len(boxes) else []

    def align(self, img: Image.Image, box=None) -> Image.Image:
        img = img.convert("RGB")
        W, H = img.size
        if box is None:  # center square crop fallback
            side = min(W, H)
            return img.crop(((W - side) // 2, (H - side) // 2, (W + side) // 2, (H + side) // 2)).resize((112, 112), Image.BILINEAR)
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        if self.eye is not None:
            roi = cv2.cvtColor(np.asarray(img.crop((x, y, x + w, y + h // 2))), cv2.COLOR_RGB2GRAY)
            eyes = sorted(self.eye.detectMultiScale(roi, 1.1, 6), key=lambda e: -e[2] * e[3])[:2]
            if len(eyes) == 2:
                (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
                ang = math.degrees(math.atan2((y2 + h2 / 2) - (y1 + h1 / 2), (x2 + w2 / 2) - (x1 + w1 / 2)))
                if abs(ang) < 25:
                    img = img.rotate(ang, center=(cx, cy), resample=Image.BILINEAR)
        side = max(w, h) * 1.25
        return img.crop((int(cx - side / 2), int(cy - side / 2 + 0.05 * side),
                         int(cx + side / 2), int(cy + side / 2 + 0.05 * side))).resize((112, 112), Image.BILINEAR)


def quality_proxy(aligned: Image.Image) -> float:
    """Proxy for AdaFace feature norm ||z||_2 (sharpness x contrast, ~0-32)."""
    g = np.asarray(aligned.convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(6 + 26 * math.tanh(lap.var() / 400.0) * min(1.0, g.std() / 40.0))


def region_raw(aligned: Image.Image) -> Dict[str, np.ndarray]:
    g = np.asarray(aligned.convert("L").resize((112, 112)), dtype=np.float32) / 255.0
    out = {k: hog(g[y0:y1, x0:x1]) for k, (y0, y1, x0, x1) in REGIONS.items()}
    small = np.asarray(aligned.convert("L").resize((56, 56)), dtype=np.float32) / 255.0
    rows = small.mean(1); cols = small.mean(0)
    prof = np.concatenate([(rows - rows.mean()) / (rows.std() + 1e-6), (cols - cols.mean()) / (cols.std() + 1e-6)])
    out["craniofacial_proportions"] = np.concatenate([hog(small, cell=8), 0.15 * prof]).astype(np.float32)
    return out


class SurrogateFaceBackend:
    name = "Surrogate HOG→RP-512 (offline demo backend)"
    is_real = False

    def __init__(self):
        rng = np.random.default_rng(1729)
        self.dim_in = 1296 + 256
        self.P = (rng.standard_normal((self.dim_in, 512)) / np.sqrt(512)).astype(np.float32)
        self.mu: Optional[np.ndarray] = None

    def raw(self, aligned: Image.Image) -> np.ndarray:
        gl = aligned.convert("L").resize((112, 112))
        g = np.asarray(gl.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
        s = np.asarray(gl.resize((16, 16)), dtype=np.float32).ravel()
        s = (s - s.mean()) / (s.std() + 1e-6)
        return np.concatenate([hog(g, cell=16), 0.3 * s]).astype(np.float32)

    def set_cohort(self, raws: List[np.ndarray]):
        self.mu = np.mean(raws, axis=0) if len(raws) >= 2 else None

    def embed(self, aligned: Image.Image) -> Tuple[np.ndarray, float]:
        r = self.raw(aligned)
        if self.mu is not None:
            r = r - self.mu
        return l2n(r @ self.P), quality_proxy(aligned)


class AdaFaceBackend:
    """AdaFace IR-101 (github.com/mk-minchul/AdaFace). Needs ADAFACE_REPO (folder with net.py)
    and ADAFACE_CKPT (e.g. adaface_ir101_webface12m.ckpt)."""
    is_real = True

    def __init__(self, repo: str, ckpt: str, arch: str = "ir_101"):
        import torch  # type: ignore
        sys.path.insert(0, repo)
        import net  # type: ignore
        self.torch = torch
        self.model = net.build_model(arch)
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        self.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
        self.model.eval()
        self.name = f"AdaFace {arch.upper()} 512-D ({os.path.basename(ckpt)})"

    def set_cohort(self, raws):
        pass

    def raw(self, aligned):
        return np.zeros(1, np.float32)

    def _t(self, aligned):
        bgr = np.asarray(aligned.convert("RGB").resize((112, 112)), dtype=np.float32)[:, :, ::-1] / 255.0
        return self.torch.tensor(((bgr - 0.5) / 0.5).transpose(2, 0, 1).copy()).unsqueeze(0).float()

    def embed(self, aligned):
        with self.torch.no_grad():
            feat, norm = self.model(self._t(aligned))
        return feat[0].numpy().astype(np.float32), float(norm[0])

    def gradcam(self, aligned, target: np.ndarray) -> np.ndarray:
        acts, grads = {}, {}
        layer = self.model.body[-1]
        h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        x = self._t(aligned).requires_grad_(True)
        feat, _ = self.model(x)
        (feat[0] * self.torch.tensor(target)).sum().backward()
        h1.remove(); h2.remove()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
        return cam / (cam.max() + 1e-9)


class InsightFaceBackend:
    """InsightFace buffalo_l: SCRFD-10GF detector + 5-point norm_crop + ArcFace R50 (WebFace600K), 512-D.
    Loaded automatically when `insightface` is installed (models cache in ~/.insightface)."""
    is_real = True

    def __init__(self):
        from insightface.app import FaceAnalysis  # type: ignore
        from insightface.utils import face_align  # type: ignore
        try:
            import onnxruntime as ort  # type: ignore
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or None
        except Exception:
            providers = None
        self.app = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if providers and "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
        self.rec = self.app.models["recognition"]
        self.face_align = face_align
        self.lock = threading.Lock()   # one ONNX inference at a time (concurrent kiosk sessions)
        try:  # bounded, non-spinning thread pools: keeps the browser/UI responsive on laptops
            import onnxruntime as ort  # type: ignore
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) - 1))
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            for m in (self.rec, self.app.det_model):
                prov = m.session.get_providers()
                m.session = ort.InferenceSession(m.model_file, sess_options=so, providers=prov)
        except Exception as e:
            print("[InsightFace] keeping default ONNX session options:", e)
        self.name = "InsightFace buffalo_l (SCRFD-10GF + ArcFace R50 512-D)"

    def set_cohort(self, raws):
        pass

    def raw(self, aligned):
        return np.zeros(1, np.float32)

    def detect_align(self, img: Image.Image):
        """Returns (aligned 112x112 or None, number of faces)."""
        bgr = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
        with self.lock:
            faces = self.app.det_model.detect(bgr, max_num=0, metric="default")
        bboxes, kpss = faces
        if bboxes is None or len(bboxes) == 0:
            return None, 0
        keep = [i for i in range(len(bboxes)) if bboxes[i][4] >= 0.5]
        if not keep:
            return None, 0
        best = max(keep, key=lambda i: (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]))
        crop = self.face_align.norm_crop(bgr, landmark=kpss[best], image_size=112)
        return Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1])), len(keep)

    def embed(self, aligned: Image.Image):
        bgr = np.ascontiguousarray(np.asarray(aligned.convert("RGB").resize((112, 112)))[:, :, ::-1])
        with self.lock:
            f = self.rec.get_feat(bgr).ravel().astype(np.float32)
        n = float(np.linalg.norm(f))
        return f / (n + 1e-9), n

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """One batched ONNX run (used by the occlusion saliency map)."""
        bgrs = [np.ascontiguousarray(np.asarray(im.convert("RGB").resize((112, 112)))[:, :, ::-1]) for im in images]
        with self.lock:
            F = self.rec.get_feat(bgrs).astype(np.float32)
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def load_face_backend():
    """AdaFace (explicit env) → InsightFace buffalo_l (auto) → surrogate."""
    if os.environ.get("NEUROGATE_FACE", "").lower() == "surrogate":
        return SurrogateFaceBackend()
    repo, ckpt = os.environ.get("ADAFACE_REPO"), os.environ.get("ADAFACE_CKPT")
    if repo and ckpt and os.path.exists(ckpt):
        try:
            return AdaFaceBackend(repo, ckpt, os.environ.get("ADAFACE_ARCH", "ir_101"))
        except Exception as e:
            print("[AdaFace] unavailable:", e)
    try:
        return InsightFaceBackend()
    except Exception as e:
        print("[InsightFace] unavailable, using surrogate:", e)
    return SurrogateFaceBackend()
