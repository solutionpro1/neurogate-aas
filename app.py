#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGate AAS — Explainable Multimodal Neuro-Symbolic Access Control
====================================================================
Entry point: Gradio Blocks, event wiring and launch.

Dr. E. P. Fasina's two-tier architecture:
  Outer loop   Autonomous Access System (AAS), Stages 1-5     → neurogate/pipeline/engine.py
  Inner module Neuro-Symbolic Face Recognition (NSFRS), 0-3   → neurogate/pipeline/nsfrs.py
  ASP gate + CoT + voice persona                              → neurogate/pipeline/symbolic.py
  Lighting / acoustic awareness                               → neurogate/pipeline/environment.py
  Hands-free browser sentry (camera, TTS, 3.5 s mic)          → neurogate/ui/sentry.py

Run (either works):
    python neurogate/app.py
    python -m neurogate.app

Open http://localhost:7860 in Chrome or Edge, press "Start Auto-Sentry" once.
"""
from __future__ import annotations

import os
import sys
import time

if __package__ in (None, ""):  # allow `python neurogate/app.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "neurogate"

import gradio as gr  # noqa: E402

from neurogate.core.config import CFG  # noqa: E402
from neurogate.core.utils import seed_demo_gallery  # noqa: E402
from neurogate.pipeline.engine import View, get_engine, run_session  # noqa: E402
from neurogate.pipeline.symbolic import solver_name  # noqa: E402
from neurogate.ui.frontend import build_css, calib_markdown, header_html, idle_view_html, render_view  # noqa: E402
from neurogate.ui.scenarios import (ENV_SCENARIOS, SCENARIOS, build_camera, build_env_scenario,  # noqa: E402
                                    build_live, build_scenario, build_walkup)
from neurogate.ui.sentry import AUTONOMOUS_SENTRY_JS, HUD_HTML  # noqa: E402


def resident_choices(eng):
    return [r.name for r in eng.residents]


def kiosk_session(eng, req, err: str):
    """Adapts pipeline.run_session() states into the kiosk outputs (banner, stepper, evidence, audit)."""
    keep = gr.update()
    if req is None:
        view = View()
        view.banner.update(state="locked", title="DOOR LOCKED", sub=err or "Nothing to process", final=True)
        yield (*render_view(eng, view), keep)
        return
    for view, final, done in run_session(eng, req):
        # the audit table only refreshes when an attempt is logged
        yield (*render_view(eng, view), eng.db.audit() if done else keep)


def build_ui(eng):
    gr_major = int(gr.__version__.split(".")[0])
    theme = gr.themes.Soft(primary_hue="red", neutral_hue="slate")
    kw = {"title": "NeuroGate AAS — Smart Door Kiosk"}
    if gr_major < 6:
        kw.update(css=build_css(), theme=theme)

    with gr.Blocks(**kw) as demo:
        header = gr.HTML(header_html(eng))
        with gr.Tabs():
            # ---------------- Tab 1: Kiosk ----------------
            with gr.Tab("🚪 Smart Door Kiosk (Autonomous Live View)"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=35, min_width=320):
                        gr.HTML('<div class="ng-zone-title">Live doorway camera · Auto-Sentry</div>')
                        gr.HTML(HUD_HTML)
                        bridge = gr.Textbox(elem_id="ng-bridge", elem_classes=["ng-hidden-bridge"], show_label=False, container=False)
                        bridge_go = gr.Button("bridge", elem_id="ng-bridge-go", elem_classes=["ng-hidden-bridge"])
                        walkup = gr.Dropdown(choices=resident_choices(eng), value=None, label="Simulate Resident Walking Up",
                                             info="Pick a resident — the door wakes up on its own")
                        with gr.Accordion("📁 Manual fallback: upload a photo / recording", open=False):
                            cam = gr.Image(sources=["upload", "webcam"], type="pil", label="Photo (runs automatically on upload)", height=200)
                            mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Recorded voice (optional)")
                            typed = gr.Textbox(label="Code spoken (only if no speech recognition is available)", max_lines=1)
                            gr.Markdown("<small>Without a recording the resident's enrolled voice is paired automatically.</small>")
                        gr.HTML('<div class="ng-zone-title">One-click scenarios</div>')
                        scn_btns = {k: gr.Button(v, elem_classes=["ng-scn"], variant="secondary") for k, v in SCENARIOS.items()}
                        gr.HTML('<div class="ng-zone-title">Environmental awareness</div>')
                        env_btns = {k: gr.Button(v, elem_classes=["ng-scn", "ng-scn-env"], variant="secondary")
                                    for k, v in ENV_SCENARIOS.items()}
                        with gr.Accordion("⚙️ System Thresholds & Calibration", open=False, elem_classes=["ng-settings"]):
                            gr.Markdown("<small>Changes apply to the next person at the door.</small>")
                            s_theta = gr.Slider(30, 95, value=CFG["theta"], step=1, label="θ — belief threshold for match(p, I)")
                            s_q = gr.Slider(4, 28, value=CFG["quality_floor"], step=0.5, label="Face quality floor ‖z‖₂")
                            s_vm = gr.Slider(0, 0.4, value=CFG["voice_margin"], step=0.01, label="Voice margin Δv minimum")
                            with gr.Row():
                                s_ff = gr.Dropdown(["1e-1", "1e-2", "1e-3", "1e-4"], value="1e-2", label="Face FAR budget")
                                s_fv = gr.Dropdown(["1e-1", "1e-2", "1e-3", "1e-4"], value="1e-2", label="Voice FAR budget")
                            s_fd = gr.Slider(0, 0.3, value=CFG["frag_delta"], step=0.01, label="Alibi fragility δ")
                            calib_md = gr.Markdown(calib_markdown(eng))
                    with gr.Column(scale=65, min_width=520):
                        b0, s0, c0 = idle_view_html()
                        banner = gr.HTML(b0, elem_id="ng-banner")
                        stepper = gr.HTML(s0)
                        compare = gr.HTML(c0)

            # ---------------- Tab 2: Enrolment ----------------
            with gr.Tab("👤 Stage 0: Resident Enrolment"):
                with gr.Row():
                    with gr.Column(scale=1):
                        e_name = gr.Textbox(label="Full name", placeholder="e.g. Ngozi Okonjo")
                        e_unit = gr.Textbox(label="Unit / flat", placeholder="e.g. Flat 6A")
                        e_photo = gr.Image(sources=["upload", "webcam"], type="pil", label="Frontal reference photo", height=220)
                        e_audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                           label="Voice sample (3–5 s: “My name is …”)")
                        with gr.Row():
                            e_btn = gr.Button("Enrol resident", variant="primary")
                            e_rescan = gr.Button("Re-scan gallery folders")
                        e_status = gr.Markdown()
                    with gr.Column(scale=1):
                        gr.Markdown("**Stage 0 pipeline:** capture refs → align 112×112 → AdaFace 512-D + ‖z‖₂ → ECAPA 192-D → "
                                    "store gallery → closed-set calibration.\n\nFiles live in `gallery/` and `gallery_audio/` "
                                    "with matching names (e.g. `Ngozi_Okonjo.jpg` + `Ngozi_Okonjo.wav`).")
                        e_calib = gr.Markdown(calib_markdown(eng))
                        e_dir = gr.Dataframe(value=eng.db.directory(), label="Enrolled closed set", interactive=False)

            # ---------------- Tab 3: Audit ----------------
            with gr.Tab("📋 Stage 5: Audit Logs & Resident Directory (SQLite)"):
                with gr.Row():
                    a_refresh = gr.Button("Refresh")
                    a_clear = gr.Button("Clear audit log", variant="stop")
                audit_df = gr.Dataframe(value=eng.db.audit(), label="Audit log (neurogate_aas.db · newest first)", interactive=False, wrap=True)
                dir_df = gr.Dataframe(value=eng.db.directory(), label="Resident directory", interactive=False)

        outs = [banner, stepper, compare, audit_df]

        # ---- zero-click wiring ----
        def on_walkup(name):
            yield from kiosk_session(eng, *build_walkup(name))

        def on_camera(img, mic_audio, code):
            yield from kiosk_session(eng, *build_camera(img, mic_audio, code or ""))

        def make_scn(key):
            def fn():
                yield from kiosk_session(eng, *build_scenario(key))
            return fn

        def on_live(payload):
            yield from kiosk_session(eng, *build_live(payload))

        # hidden bridge: the browser sentry stores its payload on window and clicks this button;
        # the js hook hands that payload to Python as the textbox value.
        bridge_go.click(on_live, [bridge], outs, js="(x) => [window.__ngPayload || '']")
        demo.load(None, None, None, js=AUTONOMOUS_SENTRY_JS)

        walkup.input(on_walkup, [walkup], outs)
        cam_event = getattr(cam, "input", None) or cam.change
        cam_event(on_camera, [cam, mic, typed], outs)
        mic.stop_recording(on_camera, [cam, mic, typed], outs)
        for k, b in scn_btns.items():
            b.click(make_scn(k), None, outs)

        def make_env(key):
            def fn():
                yield from kiosk_session(eng, *build_env_scenario(key))
            return fn

        for k, b in env_btns.items():
            b.click(make_env(k), None, outs)

        # ---- thresholds ----
        def set_cfg(theta, q, vm, ff, fv, fd):
            CFG.update(theta=float(theta), quality_floor=float(q), voice_margin=float(vm), far_face=float(ff),
                       far_voice=float(fv), frag_delta=float(fd))
            eng.calibrate()          # re-derives τ_f, τ_v, τ_low and the system FAR from the new budgets
            md = calib_markdown(eng)
            return md, md, header_html(eng)
        sliders = [s_theta, s_q, s_vm, s_ff, s_fv, s_fd]
        for comp in sliders:
            comp.change(set_cfg, sliders, [calib_md, e_calib, header])

        # ---- enrolment ----
        def do_enrol(name, unit, photo, audio):
            if not name or photo is None:
                return ("⚠️ Name and photo are required.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
            eng.enrol(name, unit or "", photo, audio)
            msg = f"✅ Enrolled **{name}**" + ("" if audio is not None else " — ⚠️ no voiceprint yet (Row 1 needs one).")
            md = calib_markdown(eng)
            d = eng.db.directory()
            return msg, md, md, d, d, gr.update(choices=resident_choices(eng))

        def do_rescan():
            eng.load_gallery()
            md = calib_markdown(eng)
            d = eng.db.directory()
            return f"🔄 Re-scanned: {len(eng.residents)} residents.", md, md, d, d, gr.update(choices=resident_choices(eng))

        enrol_outs = [e_status, e_calib, calib_md, e_dir, dir_df, walkup]
        e_btn.click(do_enrol, [e_name, e_unit, e_photo, e_audio], enrol_outs)
        e_rescan.click(do_rescan, None, enrol_outs)

        a_refresh.click(lambda: (eng.db.audit(), eng.db.directory()), None, [audit_df, dir_df])

        def clear():
            eng.db.clear_audit()
            return eng.db.audit()
        a_clear.click(clear, None, [audit_df])

    return demo, theme, gr_major


def main():
    seed_demo_gallery()
    eng = get_engine()
    t0 = time.time()
    eng.load_gallery()
    print(f"[NeuroGate] {len(eng.residents)} residents enrolled in {time.time() - t0:.1f}s · "
          f"face={eng.face_backend.name} · voice={eng.voice_backend.name} · ASP={solver_name()}")
    demo, theme, gr_major = build_ui(eng)
    launch = dict(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                  server_port=int(os.environ.get("PORT", 7860)), share=os.environ.get("GRADIO_SHARE") == "1")
    if gr_major >= 6:
        launch.update(css=build_css(), theme=theme)
    demo.queue().launch(**launch)


if __name__ == "__main__":
    main()
