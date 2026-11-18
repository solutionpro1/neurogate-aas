"""Voice models: ECAPA-TDNN / surrogate speaker embeddings and server-side ASR."""
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from neurogate.core.config import BASE_DIR, CACHE_DIR
from neurogate.core.utils import l2n, logmel


class SurrogateVoiceBackend:
    name = "Surrogate log-mel→RP-192 (offline demo backend)"
    is_real = False

    def __init__(self):
        rng = np.random.default_rng(4242)
        self.P = (rng.standard_normal((120, 192)) / np.sqrt(192)).astype(np.float32)
        self.mu = None

    def raw(self, x: np.ndarray) -> np.ndarray:
        m = logmel(x)
        e = m.mean(1)
        # speech frames only: above the 35th percentile AND within ~22 dB of the loudest frame
        voiced = m[(e > np.percentile(e, 35)) & (e > e.max() - 5.0)] if len(m) > 8 else m
        if len(voiced) < 4:
            voiced = m
        d = np.abs(np.diff(voiced, axis=0)).mean(0) if len(voiced) > 1 else np.zeros(40)
        return np.concatenate([voiced.mean(0), voiced.std(0), d]).astype(np.float32)

    def set_cohort(self, raws):
        self.mu = np.mean(raws, axis=0) if len(raws) >= 2 else None

    def embed(self, x: np.ndarray) -> np.ndarray:
        r = self.raw(x)
        if self.mu is not None:
            r = r - self.mu
        return l2n(r @ self.P)


class EcapaBackend:
    is_real = True
    name = "SpeechBrain ECAPA-TDNN 192-D (spkrec-ecapa-voxceleb)"

    def __init__(self):
        import torch  # type: ignore
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except Exception:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        self.torch = torch
        extra = {}
        try:  # copy instead of symlink (Windows without admin rights)
            from speechbrain.utils.fetching import LocalStrategy  # type: ignore
            extra["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        repo = "speechbrain/spkrec-ecapa-voxceleb"
        local_dirs = [os.path.join(BASE_DIR, "pretrained_models", "spkrec-ecapa-voxceleb"),
                      os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb"),
                      os.path.join(CACHE_DIR, "ecapa")]
        attempts = [(d, d) for d in local_dirs if os.path.exists(os.path.join(d, "hyperparams.yaml"))]
        attempts.append((repo, local_dirs[0]))
        last = None
        for source, savedir in attempts:
            for offline in (False, True):
                try:
                    if offline:
                        os.environ["HF_HUB_OFFLINE"] = "1"
                    self.clf = EncoderClassifier.from_hparams(source=source, savedir=savedir,
                                                              run_opts={"device": "cpu"}, **extra)
                    return
                except TypeError:
                    extra = {}
                except Exception as e:
                    last = e
                finally:
                    if offline:
                        os.environ.pop("HF_HUB_OFFLINE", None)
        raise RuntimeError(f"ECAPA could not be loaded: {last}")

    def set_cohort(self, raws):
        pass

    def raw(self, x):
        return np.zeros(1, np.float32)

    def embed(self, x):
        with self.torch.no_grad():
            e = self.clf.encode_batch(self.torch.tensor(x).unsqueeze(0))
        return l2n(e.squeeze().numpy().astype(np.float32))


class ASR:
    """Local-language ASR. faster-whisper when NEUROGATE_ASR=whisper; otherwise
    the simulator supplies transcripts and live-mic users type the code they spoke."""

    def __init__(self):
        self.model = None
        self._tried = False
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            self.available = os.environ.get("NEUROGATE_ASR", "").lower() != "off"
        except Exception:
            self.available = False
        self.name = ("Browser Web Speech API → faster-whisper fallback (loads on first use)" if self.available
                     else "Browser Web Speech API (pip install faster-whisper for a server-side fallback)")

    def _load(self):
        self._tried = True
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self.model = WhisperModel(os.environ.get("NEUROGATE_ASR_SIZE", "base"), device="cpu", compute_type="int8")
            self.name = "Browser Web Speech API → faster-whisper fallback (loaded)"
        except Exception as e:  # pragma: no cover
            print("[ASR] faster-whisper unavailable:", e)
            self.available = False

    def transcribe(self, x: np.ndarray) -> Optional[str]:
        if self.available and not self._tried:
            self._load()
        if self.model is None:
            return None
        segs, _ = self.model.transcribe(x, beam_size=1)
        return " ".join(s.text for s in segs).strip()


NUM_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def extract_digits(text: str) -> str:
    toks = re.findall(r"[a-z]+|\d", (text or "").lower())
    return "".join(NUM_WORDS.get(t, t) for t in toks if t.isdigit() or t in NUM_WORDS)


def load_voice_backend():
    """SpeechBrain ECAPA-TDNN (auto) → surrogate. NEUROGATE_VOICE=surrogate forces offline."""
    if os.environ.get("NEUROGATE_VOICE", "").lower() == "surrogate":
        return SurrogateVoiceBackend()
    try:
        return EcapaBackend()
    except Exception as e:
        print("[ECAPA] unavailable, using surrogate:", e)
    return SurrogateVoiceBackend()
