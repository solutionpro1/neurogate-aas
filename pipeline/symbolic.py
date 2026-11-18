"""Symbolic layer: Clingo ASP programs (NSFRS cardinality + AAS gate), exact Python
fallback evaluator, the CoT reasoner (explains, never decides) and the voice persona
(phrases spoken feedback; also never decides)."""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
from typing import Callable, Dict, List, Optional, Tuple

from neurogate.core.config import CFG

try:
    import clingo  # type: ignore
    HAVE_CLINGO = True
except Exception:  # pragma: no cover
    clingo = None
    HAVE_CLINGO = False


# =============================================================================
# Clingo ASP programs — Dr. Fasina decision table (unchanged)
# =============================================================================
NSFRS_RULES = """% NSFRS Stage 3 — symbolic cardinality over fused beliefs
match(P,I) :- belief(P,I,B), theta(T), B >= T.
card(P,C)  :- probe(P), C = #count{ I : match(P,I) }.
#show match/2.
#show card/2."""


GATE_RULES = """% AAS Stage 4 — cross-modal symbolic gate (Dr. Fasina decision table)
card(P,C) :- probe(P), C = #count{ I : match(P,I) }.
face_unique(P,I) :- match(P,I), card(P,1).
voice_ok(P,I) :- voice_top(P,I), voice_score(P,I,S), tau_v(T), S >= T,
                 voice_margin(P,M), margin_min(MM), M >= MM.

% Row 1 — the exact rule
allow(P,I) :- face_unique(P,I), voice_top(P,I), voice_ok(P,I),
              live(P), quality_ok(P), not anomaly(P).
% Row 4 — C>1 look-alikes resolved by a confident voice inside the face set
resolved(P,I) :- card(P,C), C > 1, match(P,I), voice_top(P,I), voice_ok(P,I),
                 live(P), quality_ok(P), not anomaly(P).
allow(P,I) :- resolved(P,I).
granted(P) :- allow(P,_).

decision(P,alert)    :- probe(P), not live(P).                                   % Row 5
decision(P,queue)    :- probe(P), live(P), anomaly(P).                           % Stage 1 scene check
decision(P,retry)    :- probe(P), live(P), not anomaly(P), not quality_ok(P).    % quality floor
decision(P,allow)    :- granted(P).                                              % Rows 1/4
decision(P,retry)    :- probe(P), live(P), not anomaly(P), quality_ok(P),
                        card(P,1), not granted(P).                               % Row 2
decision(P,host)     :- probe(P), live(P), not anomaly(P), quality_ok(P), card(P,0).        % Row 3
decision(P,escalate) :- probe(P), live(P), not anomaly(P), quality_ok(P),
                        card(P,C), C > 1, not granted(P).                        % Row 4 unresolved
#show allow/2.
#show decision/2.
#show card/2.
#show voice_ok/2."""


def run_clingo(program: str) -> Optional[List[str]]:
    if not HAVE_CLINGO:
        return None
    atoms: List[str] = []
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    ctl.solve(on_model=lambda m: atoms.extend(str(s) for s in m.symbols(shown=True)))
    return sorted(atoms)


def py_nsfrs(beliefs: Dict[int, int], theta: int) -> List[str]:
    m = sorted(i for i, b in beliefs.items() if b >= theta)
    return sorted([f"match(p,{i})" for i in m] + [f"card(p,{len(m)})"])


def py_gate(f: dict) -> List[str]:
    F, C = f["match"], len(f["match"])
    j, live, q, anom = f["voice_top"], f["live"], f["quality_ok"], f["anomaly"]
    vok = j is not None and f["voice_score"] >= f["tau_v"] and f["voice_margin"] >= f["margin_min"]
    out = [f"card(p,{C})"]
    if vok:
        out.append(f"voice_ok(p,{j})")
    allow = None
    if live and q and not anom and vok and j in F and C >= 1:
        allow = j
        out.append(f"allow(p,{j})")
    if not live:
        d = "alert"
    elif anom:
        d = "queue"
    elif not q:
        d = "retry"
    elif allow is not None:
        d = "allow"
    elif C == 1:
        d = "retry"
    elif C == 0:
        d = "host"
    else:
        d = "escalate"
    out.append(f"decision(p,{d})")
    return sorted(out)


def asp_solve(rules: str, facts: str, fallback: Callable[[], List[str]]) -> Tuple[List[str], str, str]:
    program = facts + "\n\n" + rules
    try:
        atoms = run_clingo(program)
        if atoms is not None:
            return atoms, f"clingo {clingo.__version__}", program
    except Exception as e:  # pragma: no cover
        print("[clingo] error, using Python evaluator:", e)
    return fallback(), "Python ASP evaluator (pip install clingo for the real solver)", program


def solver_name() -> str:
    return f"clingo {clingo.__version__}" if HAVE_CLINGO else "Python ASP evaluator"


# =============================================================================
# CoT reasoner (explains; the ASP gate decides)
# =============================================================================
class CoTReasoner:
    def __init__(self):
        self.model_id = os.environ.get("NEUROGATE_COT_MODEL")
        self._pipe = None
        self.name = (f"LLM CoT ({self.model_id})" if self.model_id
                     else "Template CoT (set NEUROGATE_COT_MODEL=Qwen/Qwen3-8B or microsoft/Phi-4-mini-instruct)")

    def _llm(self, facts: str, decision: str) -> Optional[str]:
        if not self.model_id:
            return None
        try:
            if self._pipe is None:
                from transformers import pipeline  # type: ignore
                self._pipe = pipeline("text-generation", model=self.model_id, device_map="auto")
            msg = [{"role": "system", "content": "You explain door-access decisions to residents and examiners in 2 plain sentences. "
                                                  "Never change the decision."},
                   {"role": "user", "content": f"Facts:\n{facts}\nSymbolic gate decision: {decision}. Explain why."}]
            out = self._pipe(msg, max_new_tokens=120)[0]["generated_text"]
            return (out[-1]["content"] if isinstance(out, list) else str(out)).strip()
        except Exception as e:  # pragma: no cover
            print("[CoT] LLM unavailable:", e)
            self.model_id = None
            return None

    def reason(self, d: str, ctx: dict, residents) -> Tuple[List[str], str, str]:
        R = residents
        nm = lambda i: R[i].name if i is not None and 0 <= i < len(R) else "unknown"  # noqa: E731
        F, C, j = ctx["match"], ctx["C"], ctx["voice_top"]
        first = lambda n: n.split()[0]  # noqa: E731
        steps = [
            f"Face (NSFRS): C = {C}" + (f" → {', '.join(nm(i) for i in F)}" if F else " → nobody above θ"),
            f"Voice: top j* = {nm(j)} (s = {ctx['voice_score']:.2f}, margin Δv = {ctx['voice_margin']:.2f}, τ_v = {ctx['tau_v']:.2f})",
            f"Liveness: {'passed' if ctx['live'] else 'FAILED — ' + '; '.join(ctx['live_reasons'])}",
            f"Quality ‖z‖₂ = {ctx['q']:.1f} (floor {CFG['quality_floor']:.0f}); occupancy = {ctx['occupancy']}",
        ]
        light, sound = ctx.get("light"), ctx.get("sound")
        if light is not None:
            steps.append(f"Environment: {light.label} (luma {light.luma:.0f}" +
                         (f" → {light.luma_after:.0f}" if light.corrected else "") + ")" +
                         (f"; audio {sound.label}" if sound is not None else ""))
        if d == "allow":
            who = ctx["allow_id"]
            if C > 1:
                steps.append(f"Face set is ambiguous ({C} look-alikes), but the live voice uniquely selects {nm(who)} inside that set.")
                summary = f"Look-alike faces resolved by voice: {nm(who)} confirmed by face set + live voiceprint."
            else:
                steps.append(f"Face and voice agree on {nm(who)}; liveness and quality hold; no anomaly.")
                summary = f"Face and live voice both identify {nm(who)}. All gate conditions hold."
            tts = f"Welcome home, {first(nm(who))}. The door is open."
        elif d == "retry":
            if light is not None and not light.usable:
                steps.append("Doorway too dark to recognise anyone even after exposure correction → quality_ok fails → one fresh capture.")
                summary = "Too dark to see a face reliably. Asking the visitor to step into the light for one fresh capture."
            elif not ctx["quality_ok"]:
                steps.append("Capture quality is below the AdaFace floor → one fresh capture.")
                summary = "Face image too poor to trust. Asking for one fresh capture."
            elif j is not None and F and j not in F:
                steps.append(f"Conflict: face says {nm(F[0])}, voice says {nm(j)}. Never resolve a conflict in favour of access.")
                summary = f"Identity conflict — face matches {nm(F[0])} but the voice matches {nm(j)}. Denied; one fresh capture allowed."
            else:
                steps.append("Face is unique but the voice evidence is too weak to confirm it.")
                summary = "Voice confirmation too weak. Denied; one fresh capture allowed."
            tts = "Sorry, I couldn't confirm who you are. Please look at the camera and say your code again."
        elif d == "escalate":
            if ctx.get("retry_exhausted"):
                steps.append("Second capture still fails the gate → retry budget exhausted → human guard.")
                summary = "Still conflicting after one retry. Door stays locked; a security officer has been asked to verify in person."
            else:
                steps.append(f"{C} residents match the face and the voice cannot single one out → human guard.")
                summary = f"Ambiguous look-alikes ({', '.join(nm(i) for i in F)}) that voice could not resolve. Escalated to a guard."
            tts = "I need a security officer to confirm your identity. Please wait a moment."
        elif d == "host":
            steps.append("C = 0: not a resident. The name spoken is only a hint and never opens the door.")
            claim = ctx.get("claim")
            summary = "Unrecognised visitor — not in the resident set. Routing to a human host" + (f" (visitor said: “{claim}”)." if claim else ".")
            tts = "Hello! You're not on the resident list. I'm calling your host now — please wait."
        elif d == "alert":
            steps.append("Liveness failed: possible recorded or synthetic voice. Deny regardless of face match and raise an alert.")
            summary = "Voice liveness failed (" + "; ".join(ctx["live_reasons"]) + "). Access denied; security alerted."
            tts = "Access denied. Security has been notified."
        elif d == "queue":
            steps.append("More than one person in the door zone → face and voice may come from different bodies.")
            summary = f"{ctx['occupancy']} people detected at the door. Asking them to approach one at a time."
            tts = "Please step forward one at a time."
        else:
            summary, tts = "Decision unavailable.", "Please wait."
        llm = self._llm("\n".join(steps), d)
        tts = PERSONA.decision(d, ctx, R, fallback=tts)
        return steps, (llm or summary), tts


# =============================================================================
# Voice persona — human-like, varied spoken feedback (wording only)
# =============================================================================
class VoicePersona:
    """Turns gate decisions into natural speech: time-of-day greetings, first names,
    last-visit awareness, environment-aware advice and varied phrasing so the kiosk
    does not sound like a recording. Security-relevant lines (alerts) stay short and
    reveal nothing about why a check failed."""

    def __init__(self):
        self.last_visit: Callable[[str], Optional[str]] = lambda name: None

    @staticmethod
    def daypart(now: Optional[dt.datetime] = None) -> str:
        h = (now or dt.datetime.now()).hour
        return "Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 17 else "Good evening"

    @staticmethod
    def _pick(options: List[str]) -> str:
        return random.choice(options)

    @staticmethod
    def spoken_digits(code: str) -> str:
        return ", ".join(code)

    def prompt(self, code: str, light=None) -> str:
        lead = ""
        if light is not None and light.corrected and light.usable:
            lead = self._pick(["It's a little dim here, so I've brightened the camera. ",
                               "The light is low, but I can still see you. "])
        ask = self._pick([f"{self.daypart()}. Please state your name and say code {self.spoken_digits(code)}.",
                          f"Hello. Please state your name and say code {self.spoken_digits(code)}.",
                          f"{self.daypart()}! Tell me your name, then say the code {self.spoken_digits(code)}."])
        return lead + ask

    def _welcome(self, name: str, ambiguous: bool) -> str:
        first = name.split()[0]
        prev = None
        try:
            prev = self.last_visit(name)
        except Exception:
            pass
        mins = None
        if prev:
            try:
                mins = (dt.datetime.now() - dt.datetime.strptime(prev, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            except Exception:
                mins = None
        lead = f"Thanks, {first} — I recognised your voice. " if ambiguous else ""
        if mins is not None and mins < 15:
            line = self._pick([f"Back again, {first}? The door is open.", f"Welcome back, {first}. Door's open."])
        elif mins is not None and mins < 18 * 60:
            line = self._pick([f"Welcome back, {first}. The door is open.", f"Hi again, {first}. Come on in."])
        elif mins is None:
            line = self._pick([f"{self.daypart()}, {first}. Welcome home — the door is open.",
                               f"{self.daypart()}, {first}. The door is open."])
        else:
            line = self._pick([f"{self.daypart()}, {first}. Good to see you again — the door is open.",
                               f"{self.daypart()}, {first}. Welcome home."])
        return lead + line

    def decision(self, d: str, ctx: dict, residents, fallback: str = "") -> str:
        light, sound = ctx.get("light"), ctx.get("sound")
        noisy = sound is not None and sound.condition in ("noisy", "very_noisy")
        if d == "allow" and ctx.get("allow_id") is not None:
            return self._welcome(residents[ctx["allow_id"]].name, ctx.get("C", 1) > 1)
        if d == "retry":
            if light is not None and not light.usable:
                return self._pick(["It's too dark for me to see your face. Please step closer to the light and look at the camera.",
                                   "I can't see you clearly in this light. Please move closer to the light and try again."])
            if light is not None and light.condition == "backlit":
                return "There's bright light behind you. Please step a little closer and face the camera."
            if not ctx.get("quality_ok", True):
                return self._pick(["I didn't get a clear look at you. Please hold still and look at the camera.",
                                   "Sorry, that picture was blurry. Please look straight at the camera for a moment."])
            extra = " It's a bit noisy here, so please speak up." if noisy else ""
            return self._pick(["Sorry, I couldn't confirm who you are. Let's try once more — please look at the camera and say your code again.",
                               "I'm not quite sure it's you. Please look at the camera and say your code again."]) + extra
        if d == "escalate":
            return self._pick(["I'm sorry for the wait. A security officer will come to confirm your identity shortly.",
                               "I need a security officer to confirm your identity. Please wait a moment — they're on the way."])
        if d == "host":
            return self._pick(["Hello, and welcome! You're not on the resident list, so I'm letting your host know you're here. Please wait a moment.",
                               "Welcome! I don't have you on the resident list yet. I'm calling your host now — please wait."])
        if d == "alert":
            return "I'm sorry, I can't open the door. Security has been notified."
        if d == "queue":
            return self._pick(["Hello, everyone. Please come forward one at a time so I can check each of you.",
                               "One at a time, please. I need to see each person on their own."])
        if d == "tailgate":
            return "Only one person may enter per unlock. Please step back — security has been notified."
        return fallback or "Please wait."


PERSONA = VoicePersona()
COT = CoTReasoner()
