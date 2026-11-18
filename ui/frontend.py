"""Kiosk renderers (pure functions of pipeline state → HTML) and the premium CSS layer."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from neurogate.core.config import ATTRS, ATTR_LABEL, CFG, CSS, VERSION
from neurogate.core.utils import esc, img_b64
from neurogate.pipeline.symbolic import COT, solver_name
from neurogate.ui.sentry import SENTRY_CSS


# =============================================================================
# Stage stepper, lock banner, evidence card
# =============================================================================
STAGES = [
    (1, "Presence", "LWM + VLA", ["Detect person", "Scene check", "Capture face", "Open dialogue"]),
    (2, "Voice & Liveness", "Large voice model", ["Greet", "Transcribe", "Voiceprint", "Liveness"]),
    (3, "NSFRS Face AI", "AdaFace · 3 VLMs · XAI · Clingo", ["0 Enrolment", "1 Neural", "2 XAI fusion", "3 Symbolic"]),
    (4, "Symbolic Gate", "CoT + ASP gate", ["Compare IDs", "Check quality", "Reason (CoT)", "Gate rule"]),
    (5, "Door & Anti-Tailgate", "VLA", ["Unlock door", "Verify open", "Anti-tailgate", "Log and speak"]),
]


BADGE = {"idle": ("", ""), "active": ("RUNNING", "ng-b-active"), "pass": ("PASS", "ng-b-pass"),
         "fail": ("FAIL", "ng-b-fail"), "warn": ("HOLD", "ng-b-warn"), "skip": ("SKIPPED", "ng-b-skip")}


def stepper_html(st) -> str:
    cards = []
    for n, title, expert, subs in STAGES:
        s = st[n]
        lab, cls = BADGE[s["status"]]
        chips = "".join(
            f'<span class="ng-chip ng-chip-{s["sub"][k]}">{"✓ " if s["sub"][k] == "pass" else "✕ " if s["sub"][k] == "fail" else "! " if s["sub"][k] == "warn" else ""}{esc(t)}</span>'
            for k, t in enumerate(subs))
        cards.append(f"""
        <div class="ng-step ng-step-{s['status']}">
          <div class="ng-step-head"><span class="ng-num">{n}</span>
            <div><div class="ng-step-title">{esc(title)}</div><div class="ng-step-exp">{esc(expert)}</div></div>
            {f'<span class="ng-badge {cls}">{lab}</span>' if lab else ''}</div>
          <div class="ng-step-sum">{esc(s['summary'])}</div>
          <div class="ng-chips">{chips}</div>
        </div>""")
    done = sum(1 for n, *_ in STAGES if st[n]["status"] in ("pass", "fail", "warn", "skip"))
    active = any(st[n]["status"] == "active" for n, *_ in STAGES)
    pct = min(100, 20 * done + (10 if active else 0))
    rail = f'<div class="ng-rail"><div style="width:{pct}%"></div></div>'
    return rail + '<div class="ng-stepper">' + '<div class="ng-arrow">›</div>'.join(cards) + "</div>"


BANNER_ICON = {"locked": "🔒", "scanning": "📡", "unlocked": "🔓", "alert": "🚨", "hold": "🔒"}


def banner_html(state, title, sub, cot, tts, challenge, who, say="", say_id="", final=False, retry=False, session="",
                env: Optional[dict] = None) -> str:
    chal = (f'<div class="ng-chal">Say your name and code <b>{" ".join(challenge)}</b></div>' if challenge and state == "scanning" else "")
    cot_html = f'<div class="ng-cot"><span>Reasoner</span>{esc(cot)}</div>' if cot else ""
    tts_html = f'<div class="ng-tts">🔊 “{esc(tts)}”</div>' if tts else ""
    return f"""
    <div class="ng-banner ng-{state}" data-state="{state}" data-say="{esc(say)}" data-say-id="{say_id}"
         data-final="{1 if final else 0}" data-retry="{1 if retry else 0}" data-session="{esc(session)}">
      <div class="ng-lock">{BANNER_ICON.get(state, '🔒')}</div>
      <div class="ng-banner-body">
        <div class="ng-banner-title">{esc(title)}</div>
        <div class="ng-banner-sub">{esc(sub)}</div>
        {chal}{cot_html}{tts_html}{env_chips_html(env or {})}
      </div>
    </div>"""


def compare_idle_html() -> str:
    return """<div class="ng-compare ng-compare-idle">Evidence card appears here once the NSFRS has a candidate:
    enrolled photo vs live probe with Grad-CAM, SHAP attribute credit and the Alibi flip distance.</div>"""


def compare_html(eng, res, probe_aligned: Image.Image) -> str:
    R = eng.residents
    if res.top is None:
        return compare_idle_html()
    top = R[res.top]
    in_set = res.top in res.match
    left_cap = f"{esc(top.name)} · {esc(top.unit)}" if in_set else f"Nearest: {esc(top.name)} (below θ)"
    bel = res.beliefs.get(res.top)
    bars = ""
    if res.shap is not None:
        tot = np.abs(res.shap).sum() + 1e-9
        for k, v in zip(ATTRS, res.shap):
            pct = 100 * abs(v) / tot
            bars += (f'<div class="ng-bar"><span class="ng-bar-l">{ATTR_LABEL[k]}</span>'
                     f'<span class="ng-bar-t"><span class="ng-bar-f {"ng-neg" if v < 0 else ""}" style="width:{pct:.0f}%"></span></span>'
                     f'<span class="ng-bar-v">{"+" if v >= 0 else "−"}{pct:.0f}%</span></div>')
    flip = ""
    if res.flip is not None and res.flip_to is not None:
        flip = (f'<div class="ng-flip {"ng-fragile" if res.fragile else ""}"><b>Alibi flip distance δ = {res.flip:.3f}</b> '
                f'to switch to {esc(R[res.flip_to].name)} — {"FRAGILE (belief penalised)" if res.fragile else "robust"}</div>')
    return f"""
    <div class="ng-compare">
      <div class="ng-pair">
        <figure><img src="{img_b64(top.photo, 200)}"/><figcaption>Enrolled · {left_cap}</figcaption></figure>
        <div class="ng-vs">vs</div>
        <figure><img src="{img_b64(res.cam or probe_aligned, 200)}"/><figcaption>Live probe · {esc(res.cam_label.split(' (')[0])}</figcaption></figure>
      </div>
      <div class="ng-metrics">
        <div class="ng-kpis">
          <div><span>AdaFace cos</span><b>{res.cos[res.top]:.3f}</b></div>
          <div><span>‖z‖₂</span><b>{res.q:.1f}</b></div>
          <div><span>Belief</span><b>{'—' if bel is None else f'{bel:.0f}'}</b></div>
          <div><span>Cardinality C</span><b>{res.C}</b></div>
        </div>
        <div class="ng-shap-title">SHAP attribute credit</div>{bars or '<div class="ng-muted">No shortlisted candidate.</div>'}
        {flip}
      </div>
    </div>"""


def env_chips_html(env: dict) -> str:
    light, sound = env.get("light"), env.get("sound")
    chips = []
    if light is not None:
        cls = "ok" if light.condition == "good" else ("bad" if not light.usable else "warn")
        chips.append(f'<span class="ng-env {cls}" title="frame luma {light.luma:.0f}">{light.icon} {esc(light.label)}</span>')
    if sound is not None:
        cls = "ok" if sound.condition == "quiet" else ("bad" if sound.condition in ("silent", "very_noisy") else "warn")
        chips.append(f'<span class="ng-env {cls}">{sound.icon} {esc(sound.label)}</span>')
    return f'<div class="ng-envrow">{"".join(chips)}</div>' if chips else ""


def render_view(eng, view):
    """(banner, stepper, evidence) HTML for a pipeline View."""
    banner = banner_html(**view.banner, env=view.env)
    if view.evidence is not None:
        res, aligned = view.evidence
        evidence = compare_html(eng, res, aligned)
    else:
        evidence = compare_idle_html()
    return banner, stepper_html(view.st), evidence


def idle_view_html():
    from neurogate.pipeline.engine import View
    v = View()
    return banner_html(**v.banner), stepper_html(v.st), compare_idle_html()


# =============================================================================
# Calibration summary (threshold accordion + enrolment tab)
# =============================================================================
def calib_markdown(eng) -> str:
    c = eng.calib
    if not c:
        return "_No calibration yet._"
    lines = [
        f"**Closed set:** N = {c['N']} residents · {c['n_imp_pairs']} resident impostor pairs + "
        f"{c.get('cohort', (0, 0))[0]} cohort faces / {c.get('cohort', (0, 0))[1]} cohort voices",
        f"**Face backend:** {eng.face_backend.name}  \n**Voice backend:** {eng.voice_backend.name}",
        f"**ASR:** {eng.asr.name}  \n**CoT:** {COT.name}  \n**ASP:** {solver_name()}"
        f"  \n**Graph fusion:** {'PyTorch Geometric available' if eng.pyg else 'numpy message passing (PyG not installed)'}",
        "",
        "| Quantity | Face | Voice |", "|---|---|---|",
        f"| System FAR budget | {CFG['far_face']:.0e} | {CFG['far_voice']:.0e} |",
        f"| Per-pair rate (budget / N) | f = {c['f']:.2e} | v = {c['v']:.2e} |",
        f"| Impostor median ± robust σ | {c['imp_face'][0]:.3f} ± {c['imp_face'][1]:.3f} | {c['imp_voice'][0]:.3f} ± {c['imp_voice'][1]:.3f} |",
        f"| Genuine median ± robust σ | {c['gen_face'][0]:.3f} ± {c['gen_face'][1]:.3f} | {c['gen_voice'][0]:.3f} ± {c['gen_voice'][1]:.3f} |",
        f"| Threshold | τ_f = {c['tau_f']:.3f} (shortlist τ_low = {c['tau_low']:.3f}) | τ_v = {c['tau_v']:.3f} |",
        "",
        f"**Joint independent system FAR = N × f × v = {c['N']} × {c['f']:.2e} × {c['v']:.2e} = {c['system_far']:.2e}**",
        f"\nBelief threshold θ = {CFG['theta']:.0f} · quality floor ‖z‖₂ ≥ {CFG['quality_floor']:.1f} · voice margin ≥ {CFG['voice_margin']:.2f}",
    ]
    if c["notes"]:
        lines += ["", "⚠️ " + "  \n⚠️ ".join(c["notes"])]
    return "\n".join(lines)


def header_html(eng) -> str:
    c = eng.calib or {}
    pills = [
        ("Face", eng.face_backend.name.split(" (")[0]),
        ("Voice", eng.voice_backend.name.split(" (")[0]),
        ("ASP", solver_name()),
        ("Closed set", f"N = {c.get('N', 0)}"),
        ("System FAR", f"{c.get('system_far', 0):.1e}"),
    ]
    chips = "".join(f'<span class="ng-hpill"><i>{esc(k)}</i>{esc(v)}</span>' for k, v in pills)
    return f"""<div id="ng-header">
      <div class="ng-brand"><div class="logo">🛡️</div>
        <div><h1>NeuroGate <span>AAS</span></h1>
        <p>Explainable multimodal neuro-symbolic access control · v{VERSION}</p></div></div>
      <div class="ng-hpills">{chips}</div></div>"""


# =============================================================================
# Premium CSS layer (brand: red · black · white)
# =============================================================================
PREMIUM_CSS = """
:root{--ng-red:#d61f2c;--ng-red-d:#9f1520;--ng-ink:#0b0f17;--ng-ink2:#161c27;--ng-line:rgba(15,23,42,.08);
  --ng-glass:rgba(255,255,255,.72);--ng-shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06)}
body,.gradio-container{background:radial-gradient(1200px 600px at 100% -10%,rgba(214,31,44,.07),transparent 60%),
  radial-gradient(900px 500px at -10% 0%,rgba(15,23,42,.05),transparent 60%),#f5f6f8!important}
#ng-header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;padding:14px 18px;margin:4px 0 6px;
  border-radius:20px;background:linear-gradient(120deg,var(--ng-ink) 0%,var(--ng-ink2) 55%,#3a0c11 100%);color:#fff;box-shadow:var(--ng-shadow)}
#ng-header *{color:#fff!important}
#ng-header .ng-brand{display:flex;align-items:center;gap:14px}
#ng-header .logo{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,var(--ng-red),var(--ng-red-d));
  display:grid;place-items:center;font-size:22px;box-shadow:0 6px 18px rgba(214,31,44,.45)}
#ng-header h1{font-size:20px;margin:0;font-weight:800;letter-spacing:-.02em}#ng-header h1 span{color:#ff6b74!important}
#ng-header p{margin:0;font-size:12px;opacity:.65}
.ng-hpills{display:flex;gap:6px;flex-wrap:wrap}
.ng-hpill{font-size:11.5px;padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12)}
.ng-hpill i{font-style:normal;opacity:.55;margin-right:6px;text-transform:uppercase;font-size:9.5px;letter-spacing:.1em}
.tab-nav button,button[role=tab]{font-weight:600!important;border-radius:12px 12px 0 0!important}
button[role=tab][aria-selected=true]{color:var(--ng-red)!important}
.ng-banner{border:1px solid rgba(255,255,255,.08);box-shadow:0 18px 40px rgba(2,6,23,.22)!important}
.ng-locked{background:linear-gradient(135deg,#111827,#0b0f17)!important}
.ng-hold{background:linear-gradient(135deg,#7c2d12,#1c1917)!important}
.ng-scanning{background:linear-gradient(135deg,#1d4ed8,#0b1530)!important}
.ng-unlocked{background:linear-gradient(135deg,#059669,#053b2c)!important}
.ng-alert{background:linear-gradient(135deg,var(--ng-red),#3b0509)!important}
.ng-envrow{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.ng-env{font-size:12px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.18)}
.ng-env.warn{background:rgba(251,191,36,.22)}.ng-env.bad{background:rgba(239,68,68,.30)}
.ng-rail{height:4px;border-radius:999px;background:rgba(15,23,42,.07);margin-top:14px;overflow:hidden}
.ng-rail>div{height:100%;background:linear-gradient(90deg,var(--ng-red),#f97316);border-radius:999px;transition:width .5s ease}
.ng-stepper{margin-top:10px!important}
.ng-step{background:var(--ng-glass)!important;backdrop-filter:blur(10px);box-shadow:var(--ng-shadow);border-color:var(--ng-line)!important}
.ng-step-active{border-color:#3b82f6!important;background:#eff6ff!important}
.ng-step-pass{border-color:#10b981!important;background:#f0fdf6!important}
.ng-step-fail{border-color:#ef4444!important;background:#fff5f5!important}
.ng-step-warn{border-color:#f59e0b!important;background:#fffcf0!important}
.ng-compare{box-shadow:var(--ng-shadow);border-color:var(--ng-line)!important;border-radius:22px!important}
.ng-scn,.ng-scn button{background:#fff!important;border:1px solid var(--ng-line)!important;box-shadow:var(--ng-shadow)!important;
  border-left:4px solid var(--ng-red)!important;transition:transform .15s ease,box-shadow .15s ease!important}
.ng-scn:hover{transform:translateY(-1px);box-shadow:0 10px 26px rgba(15,23,42,.10)!important}
.ng-scn-env,.ng-scn-env button{border-left-color:#6366f1!important}
.ng-zone-title{font-weight:700;color:#475569}
.ng-cam{border:1px solid rgba(255,255,255,.06)}
#ng-sentry-btn{box-shadow:0 8px 22px rgba(2,6,23,.18)}
#ng-sentry-btn.armed{background:linear-gradient(135deg,var(--ng-red),var(--ng-red-d))!important}
footer{opacity:.5}
"""


def build_css() -> str:
    return CSS + SENTRY_CSS + PREMIUM_CSS
