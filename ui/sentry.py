"""Browser-side Autonomous Sentry: WebRTC camera, face lock, spoken challenge,
automatic 3.5 s recording, spoken decisions and the ambient light meter.

The JavaScript runs via demo.load(js=AUTONOMOUS_SENTRY_JS). It talks to Python through the
hidden bridge components (#ng-bridge textbox + #ng-bridge-go button) and listens to the
banner's data-say / data-final / data-session attributes for spoken feedback.
"""

SENTRY_CSS = '\n#ng-hud{position:relative}\n.ng-cam{position:relative;border-radius:20px;overflow:hidden;background:#0b1220;aspect-ratio:4/3;box-shadow:0 8px 24px rgba(0,0,0,.18)}\n.ng-cam video,.ng-cam canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}\n.ng-hud-top{position:absolute;top:10px;left:10px;right:10px;display:flex;justify-content:space-between;gap:6px;z-index:3}\n.ng-pill{background:rgba(15,23,42,.72);color:#e2e8f0!important;font-size:11px;padding:4px 9px;border-radius:999px;backdrop-filter:blur(6px)}\n.ng-pill.on{background:rgba(5,150,105,.85);color:#fff!important}.ng-pill.paused{background:rgba(180,83,9,.85);color:#fff!important}\n.ng-hud-msg{position:absolute;left:12px;right:12px;bottom:80px;text-align:center;z-index:3;color:#fff!important;font-weight:600;font-size:15px;\n  text-shadow:0 2px 8px rgba(0,0,0,.7)}\n.ng-hud-bottom{position:absolute;left:12px;right:12px;bottom:12px;z-index:3;display:flex;flex-direction:column;gap:6px}\n.ng-meter{position:relative;height:22px;border-radius:999px;background:rgba(15,23,42,.65);overflow:hidden}\n.ng-meter>div{position:absolute;inset:0 auto 0 0;width:0;background:linear-gradient(90deg,#10b981,#34d399);transition:width .12s linear}\n.ng-meter>span{position:relative;display:block;text-align:center;font-size:11.5px;line-height:22px;color:#fff!important;font-weight:600}\n#ng-listen{display:none}#ng-listen.on{display:flex;gap:6px;align-items:center}\n#ng-listen .ng-meter{flex:1}#ng-cdfill{background:linear-gradient(90deg,#3b82f6,#60a5fa)!important}\n.ng-level{width:70px;height:22px;border-radius:999px;background:rgba(15,23,42,.65);overflow:hidden;position:relative}\n#ng-levelfill{position:absolute;inset:0 auto 0 0;width:0;background:linear-gradient(90deg,#22c55e,#eab308,#ef4444);transition:width .06s}\n.ng-hud-idle{position:absolute;inset:0;display:grid;place-items:center;color:#94a3b8!important;font-size:13px;text-align:center;padding:20px;z-index:2}\n#ng-sentry-btn{margin-top:10px;width:100%;border:0;border-radius:14px;padding:11px 14px;font-weight:700;font-size:14px;cursor:pointer;\n  background:#111827;color:#fff}\n#ng-sentry-btn.armed{background:#b91c1c}#ng-sentry-btn.paused{background:#047857}\n.ng-hidden-bridge{position:absolute!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:0!important;\n  pointer-events:none!important;padding:0!important;margin:0!important;border:0!important;min-width:0!important}\n'

HUD_HTML = '\n<div id="ng-hud">\n  <div class="ng-cam">\n    <video id="ng-video" autoplay muted playsinline></video>\n    <canvas id="ng-overlay"></canvas>\n    <div class="ng-hud-idle" id="ng-hud-idle"><div><div style="font-size:30px;margin-bottom:6px">📷</div>\n      <div style="font-weight:700;color:#e2e8f0">Doorway camera is off</div>\n      <div style="margin-top:4px;color:#cbd5e1">Press <span style="color:#fff;font-weight:700">Start Auto-Sentry</span> once. Browsers need one click\n      before they allow the camera, microphone and speaker; after that the kiosk runs hands-free.</div></div></div>\n    <div class="ng-hud-top"><span class="ng-pill" id="ng-det">detector: —</span>\n      <span style="display:flex;gap:6px"><span class="ng-pill" id="ng-light">💡 —</span><span class="ng-pill" id="ng-sentry-state">⏻ OFF</span></span></div>\n    <div class="ng-hud-msg" id="ng-hud-msg"></div>\n    <div class="ng-hud-bottom">\n      <div class="ng-meter" id="ng-lockbar"><div id="ng-lockfill"></div><span id="ng-locktxt">Face Lock 0%</span></div>\n      <div id="ng-listen"><div class="ng-meter"><div id="ng-cdfill"></div><span id="ng-cdtxt">🎙️ Listening… 3.5s</span></div>\n        <div class="ng-level"><div id="ng-levelfill"></div></div></div>\n    </div>\n  </div>\n  <button id="ng-sentry-btn" type="button">▶ Start Auto-Sentry</button>\n</div>\n'

# Raw string: no Python escape processing inside the JavaScript.
AUTONOMOUS_SENTRY_JS = r"""
() => {
  if (window.__ng) return [];
  const NG = window.__ng = {armed:false, paused:false, phase:'idle', lock:0, lastSay:null, cooldownUntil:0,
    stream:null, det:null, detKind:'none', prevC:null, bg:null, lastTick:performance.now(), waitingSince:0, nFaces:0};
  const LOCK_MS = 1200, REC_MS = 3500;
  const $ = (id) => document.getElementById(id);
  const setTxt = (id, t) => { const e = $(id); if (e) e.textContent = t; };

  // ---------- speech out ----------
  function pickVoice() {
    const vs = (window.speechSynthesis && speechSynthesis.getVoices()) || [];
    const en = vs.filter(v => /^en/i.test(v.lang));
    const natural = (v) => /natural|neural|online|enhanced|premium|siri/i.test(v.name);
    return en.find(v => /en-NG/i.test(v.lang) && natural(v)) || en.find(v => natural(v) && /en-(GB|US)/i.test(v.lang))
        || en.find(v => /Google UK English Female|Google US English|Samantha|Daniel|Serena/i.test(v.name))
        || en.find(v => /en-NG/i.test(v.lang)) || en.find(v => /en-GB/i.test(v.lang)) || en[0] || null;
  }
  function speak(text, onend) {
    let done = false;
    const finish = () => { if (!done) { done = true; onend && onend(); } };
    if (!text || !('speechSynthesis' in window)) { finish(); return; }
    try {
      const parts = (text.match(/[^.!?]+[.!?]*/g) || [text]).map(s => s.trim()).filter(Boolean);
      const v = pickVoice();
      parts.forEach((p, i) => {
        const u = new SpeechSynthesisUtterance(p);
        u.rate = 0.97; u.pitch = 1.03; u.volume = 1.0;
        if (v) u.voice = v;
        if (i === parts.length - 1) { u.onend = finish; u.onerror = finish; }
        speechSynthesis.speak(u);
      });
      setTimeout(finish, 2500 + text.length * 95);   // safety net: some browsers never fire onend
    } catch (e) { finish(); }
  }
  NG.speak = speak;

  // ---------- banner watcher: speaks every new decision / prompt from Python ----------
  setInterval(() => {
    const b = document.querySelector('#ng-banner .ng-banner');
    if (!b) return;
    const id = b.dataset.sayId;
    if (NG.lastSay === null) { NG.lastSay = id || ''; }
    else if (id && id !== NG.lastSay) {
      NG.lastSay = id;
      speak(b.dataset.say, () => checkDone());
    }
    checkDone();
    if (NG.phase === 'waiting' && performance.now() - NG.waitingSince > 90000) onDecisionSpoken(false);
  }, 250);

  function checkDone() {
    const b = document.querySelector('#ng-banner .ng-banner');
    if (!b || NG.phase !== 'waiting') return;
    const speaking = window.speechSynthesis && speechSynthesis.speaking;
    if (b.dataset.final === '1' && b.dataset.session === NG.session && !speaking) onDecisionSpoken(b.dataset.retry === '1');
  }
  function onDecisionSpoken(isRetry) {
    if (NG.phase !== 'waiting') return;
    NG.phase = 'idle'; NG.lock = 0;
    NG.cooldownUntil = performance.now() + (isRetry ? 1500 : 7000);
    hudMsg(isRetry ? '🔁 Fresh capture — look at the camera' : '');
  }

  // ---------- HUD ----------
  function hudMsg(t) { setTxt('ng-hud-msg', t); }
  function setLock(p) {
    const f = $('ng-lockfill'); if (f) f.style.width = Math.round(p * 100) + '%';
    setTxt('ng-locktxt', 'Face Lock ' + Math.round(p * 100) + '%');
  }
  function setState() {
    const s = $('ng-sentry-state'), btn = $('ng-sentry-btn');
    if (!s || !btn) return;
    s.className = 'ng-pill' + (NG.armed ? (NG.paused ? ' paused' : ' on') : '');
    s.textContent = !NG.armed ? '⏻ OFF' : NG.paused ? '⏸ PAUSED' : '● AUTO-SENTRY';
    btn.className = NG.armed ? (NG.paused ? 'paused' : 'armed') : '';
    btn.textContent = !NG.armed ? '▶ Start Auto-Sentry' : NG.paused ? '▶ Resume Auto-Sentry' : '⏸ Pause Auto-Sentry';
  }
  function drawOverlay(boxes, vw, vh) {
    const c = $('ng-overlay'), v = $('ng-video'); if (!c || !v) return;
    const W = c.clientWidth, H = c.clientHeight; if (c.width !== W) c.width = W; if (c.height !== H) c.height = H;
    const g = c.getContext('2d'); g.clearRect(0, 0, W, H);
    // object-fit: cover mapping
    const s = Math.max(W / vw, H / vh), ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
    const col = NG.lock >= 1 ? '#10b981' : NG.lock > 0 ? '#fbbf24' : 'rgba(255,255,255,.75)';
    g.lineWidth = 3; g.setLineDash(NG.lock > 0 ? [] : [10, 8]); g.strokeStyle = col;
    g.beginPath(); g.ellipse(W / 2, H * 0.47, W * 0.2, H * 0.34, 0, 0, Math.PI * 2); g.stroke();
    g.setLineDash([]); g.lineWidth = 2;
    for (const b of boxes) { g.strokeStyle = boxes.length > 1 ? '#ef4444' : col; g.strokeRect(ox + b.x * s, oy + b.y * s, b.w * s, b.h * s); }
  }

  // ---------- face detection: MediaPipe → native FaceDetector → motion-steadiness sentry ----------
  async function initDetector() {
    try {
      const ver = '0.10.14';
      const vision = await Promise.race([
        import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + ver),
        new Promise((_, rej) => setTimeout(() => rej('timeout'), 8000))]);
      const fs = await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + ver + '/wasm');
      NG.det = await vision.FaceDetector.createFromOptions(fs, {
        baseOptions: {modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'},
        runningMode: 'VIDEO', minDetectionConfidence: 0.6});
      NG.detKind = 'mediapipe'; setTxt('ng-det', 'detector: MediaPipe BlazeFace'); return;
    } catch (e) { console.warn('[NeuroGate] MediaPipe unavailable', e); }
    try {
      if ('FaceDetector' in window) { NG.det = new window.FaceDetector({fastMode: true, maxDetectedFaces: 4});
        NG.detKind = 'native'; setTxt('ng-det', 'detector: browser FaceDetector'); return; }
    } catch (e) {}
    NG.detKind = 'motion'; setTxt('ng-det', 'detector: motion sentry (server verifies face)');
  }
  const small = document.createElement('canvas'); small.width = 64; small.height = 48;
  const lux = document.createElement('canvas'); lux.width = 32; lux.height = 24;
  NG.lightAt = 0; NG.darkSince = 0; NG.darkSpokenAt = -1e9;
  function lightCheck(v, faces) {
    const now = performance.now();
    if (now - NG.lightAt < 500) return; NG.lightAt = now;
    const g = lux.getContext('2d', {willReadFrequently: true}); g.drawImage(v, 0, 0, 32, 24);
    const d = g.getImageData(0, 0, 32, 24).data; let s = 0, clip = 0;
    for (let i = 0; i < d.length; i += 4) { const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; s += y; if (y > 250) clip++; }
    const luma = s / (d.length / 4), clipf = clip / (d.length / 4);
    const label = luma < 35 ? '🌑 Very dark' : luma < 60 ? '🌙 Low light' : luma < 85 ? '🌥️ Dim' : clipf > 0.3 ? '☀️ Glare' : '💡 Good light';
    setTxt('ng-light', label); NG.luma = luma;
    if (luma < 35) { NG.darkSince = NG.darkSince || now; } else NG.darkSince = 0;
    if (NG.darkSince && now - NG.darkSince > 3000 && now - NG.darkSpokenAt > 30000 && NG.phase === 'idle'
        && !(window.speechSynthesis && speechSynthesis.speaking) && !NG.paused) {
      NG.darkSpokenAt = now;
      speak("It's quite dark here. Please step closer to the light so I can see you.");
    }
  }
  async function detect(v) {
    const vw = v.videoWidth, vh = v.videoHeight;
    if (NG.detKind === 'mediapipe') {
      const r = NG.det.detectForVideo(v, performance.now());
      return r.detections.map(d => ({x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height}));
    }
    if (NG.detKind === 'native') {
      const fs = await NG.det.detect(v);
      return fs.map(f => ({x: f.boundingBox.x, y: f.boundingBox.y, w: f.boundingBox.width, h: f.boundingBox.height}));
    }
    // motion sentry: something new and steady inside the oval guide vs. a learned empty-door background
    const g = small.getContext('2d', {willReadFrequently: true}); g.drawImage(v, 0, 0, 64, 48);
    const d = g.getImageData(0, 0, 64, 48).data, cur = new Float32Array(64 * 48);
    for (let i = 0; i < cur.length; i++) cur[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
    if (!NG.bg) { NG.bg = cur; NG.prev = cur; return []; }
    let diff = 0, move = 0, n = 0;
    for (let y = 6; y < 42; y++) for (let x = 20; x < 44; x++) { const i = y * 64 + x; diff += Math.abs(cur[i] - NG.bg[i]); move += Math.abs(cur[i] - NG.prev[i]); n++; }
    diff /= n; move /= n; NG.prev = cur;
    if (diff < 10) { for (let i = 0; i < cur.length; i++) NG.bg[i] = 0.95 * NG.bg[i] + 0.05 * cur[i]; return []; }
    if (move > 9) return [{x: vw * 0.3, y: vh * 0.1, w: vw * 0.4, h: vh * 0.75, moving: true}];
    return [{x: vw * 0.3, y: vh * 0.1, w: vw * 0.4, h: vh * 0.75}];
  }

  function busy() {
    const b = document.querySelector('#ng-banner .ng-banner');
    return NG.phase !== 'idle' || (b && b.dataset.state === 'scanning') || (window.speechSynthesis && speechSynthesis.speaking)
           || performance.now() < NG.cooldownUntil;
  }

  async function tick() {
    const v = $('ng-video');
    const now = performance.now(), dt = Math.min(250, now - NG.lastTick); NG.lastTick = now;
    try {
      if (NG.armed && v && v.readyState >= 2 && v.videoWidth) {
        const boxes = NG.paused ? [] : await detect(v);
        try { lightCheck(v, boxes.length); } catch (e) {}
        const vw = v.videoWidth, vh = v.videoHeight;
        const big = boxes.filter(b => b.w > vw * 0.12);
        NG.nFaces = big.length;
        if (NG.paused) { NG.lock = 0; hudMsg('Auto-Sentry paused'); }
        else if (busy()) { NG.lock = 0; if (NG.phase === 'idle' && !speechSynthesis.speaking) hudMsg(performance.now() < NG.cooldownUntil ? 'Next person in a moment…' : 'Processing…'); }
        else if (big.length === 0) { NG.lock = Math.max(0, NG.lock - dt / 600); NG.prevC = null; hudMsg('Waiting for someone to approach…'); }
        else {
          const b = big[0], cx = (b.x + b.w / 2) / vw, cy = (b.y + b.h / 2) / vh;
          const steady = !b.moving && (!NG.prevC || Math.hypot(cx - NG.prevC[0], cy - NG.prevC[1]) < 0.035);
          const centred = Math.abs(cx - 0.5) < 0.22 && Math.abs(cy - 0.47) < 0.28;
          NG.prevC = [cx, cy];
          if (big.length > 1) { NG.lock = Math.min(1, NG.lock + dt / LOCK_MS); hudMsg('👥 ' + big.length + ' people — one at a time, please'); }
          else if (!centred) { NG.lock = Math.max(0, NG.lock - dt / 900); hudMsg('Step into the oval'); }
          else if (!steady) { NG.lock = Math.max(0, NG.lock - dt / 1500); hudMsg('Hold still…'); }
          else { NG.lock = Math.min(1, NG.lock + dt / LOCK_MS); hudMsg(NG.lock < 1 ? 'Face found — hold still' : 'Face locked'); }
          if (NG.lock >= 1) { NG.lock = 1; setLock(1); drawOverlay(big, vw, vh); runCapture(v, big.length); }
        }
        setLock(NG.lock); drawOverlay(big, vw, vh);
      }
    } catch (e) { console.warn('[NeuroGate] tick', e); }
    setTimeout(tick, 110);
  }

  // ---------- capture → spoken challenge → 3.5 s recording → Python ----------
  function grabFrame(v) {
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0); return c.toDataURL('image/jpeg', 0.92);
  }
  function challengeDigits() {
    const a = new Uint32Array(4); crypto.getRandomValues(a); return Array.from(a, x => String(x % 10)).join('');
  }
  async function runCapture(v, nFaces) {
    if (NG.phase !== 'idle') return;
    NG.phase = 'capturing';
    const img = grabFrame(v);
    const detector = {mediapipe: 'MediaPipe BlazeFace', native: 'browser FaceDetector', motion: 'motion sentry'}[NG.detKind] || 'browser';
    if (nFaces > 1) {   // Stage 1 scene check: no dialogue, let the server raise the queue warning
      send({img, wav: '', transcript: '', challenge: '', n_faces: nFaces, detector}); return;
    }
    const code = challengeDigits();
    NG.phase = 'prompt';
    hudMsg('🔊 Say your name and code ' + code.split('').join(' '));
    speak('Hello. Please state your name and say code ' + code.split('').join(', ') + '.', async () => {
      NG.phase = 'listening';
      const rec = await record(REC_MS);
      hudMsg('Thank you — verifying…');
      send({img, wav: rec.wav, transcript: rec.transcript, challenge: code, n_faces: nFaces, detector, rec_seconds: REC_MS / 1000});
    });
  }
  function send(p) {
    NG.phase = 'waiting'; NG.waitingSince = performance.now();
    NG.session = challengeDigits() + challengeDigits() + String(Date.now() % 100000);
    p.nonce = NG.session;
    window.__ngPayload = JSON.stringify(p);
    const host = document.getElementById('ng-bridge-go');
    const btn = host && (host.tagName === 'BUTTON' ? host : host.querySelector('button'));
    if (btn) btn.click(); else { console.error('[NeuroGate] bridge button missing'); NG.phase = 'idle'; }
  }

  async function record(ms) {
    const out = {wav: '', transcript: ''};
    const listen = $('ng-listen'); if (listen) listen.className = 'on';
    let recog = null, finalTxt = '', interim = '';
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) {
      try {
        recog = new SR(); recog.lang = 'en-NG'; recog.interimResults = true; recog.continuous = true; recog.maxAlternatives = 1;
        recog.onresult = (e) => { interim = ''; for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) finalTxt += e.results[i][0].transcript + ' '; else interim += e.results[i][0].transcript; } };
        recog.onerror = () => {};
        recog.start();
      } catch (e) { recog = null; }
    }
    try {
      const ctx = NG.actx || (NG.actx = new (window.AudioContext || window.webkitAudioContext)());
      await ctx.resume();
      const src = ctx.createMediaStreamSource(new MediaStream(NG.stream.getAudioTracks()));
      const an = ctx.createAnalyser(); an.fftSize = 1024;
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const mute = ctx.createGain(); mute.gain.value = 0;
      const chunks = [];
      proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(an); src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
      const buf = new Float32Array(an.fftSize), t0 = performance.now();
      await new Promise(res => {
        const step = () => {
          const el = performance.now() - t0, left = Math.max(0, ms - el);
          an.getFloatTimeDomainData(buf); let r = 0; for (const x of buf) r += x * x; r = Math.sqrt(r / buf.length);
          const lv = $('ng-levelfill'); if (lv) lv.style.width = Math.min(100, r * 400) + '%';
          const cd = $('ng-cdfill'); if (cd) cd.style.width = (100 * left / ms) + '%';
          setTxt('ng-cdtxt', '🎙️ Listening… ' + (left / 1000).toFixed(1) + 's');
          if (el >= ms) res(); else requestAnimationFrame(step);
        };
        step();
      });
      proc.disconnect(); src.disconnect(); mute.disconnect();
      const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n);
      let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
      out.wav = encodeWav(resample(all, ctx.sampleRate, 16000), 16000);
    } catch (e) { console.error('[NeuroGate] recording failed', e); }
    if (recog) {
      await new Promise(res => { let fin = false; const end = () => { if (!fin) { fin = true; res(); } };
        recog.onend = end; try { recog.stop(); } catch (e) { end(); } setTimeout(end, 1500); });
    }
    out.transcript = (finalTxt + ' ' + interim).trim();
    if (listen) listen.className = '';
    return out;
  }
  function resample(x, from, to) {
    if (from === to) return x;
    const n = Math.round(x.length * to / from), y = new Float32Array(n), r = from / to;
    for (let i = 0; i < n; i++) { const p = i * r, k = Math.floor(p), f = p - k; y[i] = (x[k] || 0) * (1 - f) + (x[k + 1] || 0) * f; }
    return y;
  }
  function encodeWav(x, sr) {
    const b = new ArrayBuffer(44 + x.length * 2), v = new DataView(b);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, x.length * 2, true);
    for (let i = 0; i < x.length; i++) { const s = Math.max(-1, Math.min(1, x[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    const bytes = new Uint8Array(b); let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return 'data:audio/wav;base64,' + btoa(bin);
  }

  // ---------- start / pause ----------
  async function start() {
    try {
      NG.stream = await navigator.mediaDevices.getUserMedia({
        video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user'},
        audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}});
    } catch (e) {
      hudMsg('⚠️ Camera/microphone blocked: ' + (e && e.name ? e.name : e) + '. Open the app at http://localhost:7860 (or HTTPS) and allow access.');
      return;
    }
    const v = $('ng-video'); v.srcObject = NG.stream; await v.play().catch(() => {});
    const idle = $('ng-hud-idle'); if (idle) idle.style.display = 'none';
    NG.armed = true; NG.paused = false; setState();
    speak('Auto sentry active.');
    setTxt('ng-det', 'detector: loading…');
    await initDetector();
  }
  document.addEventListener('click', (e) => {
    const t = e.target && e.target.closest && e.target.closest('#ng-sentry-btn');
    if (!t) return;
    e.preventDefault();
    if (!NG.armed) { start(); return; }
    NG.paused = !NG.paused; NG.lock = 0; setState();
    if (NG.paused && window.speechSynthesis) speechSynthesis.cancel();
  });
  if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => {};
  tick();
  return [];
}
"""

SENTRY_JS = AUTONOMOUS_SENTRY_JS  # backwards-compatible name
