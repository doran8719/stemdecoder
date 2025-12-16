(function () {
  const DATA = window.__STEMDECODER_DATA__;
  if (!DATA || !Array.isArray(DATA.stems)) return;

  const apiBase = (DATA.api_base || "").replace(/\/+$/, "");
  const jobName = DATA.active_job || "";

  // DOM
  const playBtn = document.getElementById("sdPlayBtn");
  const trackTitle = document.getElementById("trackTitle");
  const pillBpm = document.getElementById("pillBpm");
  const pillKey = document.getElementById("pillKey");
  const durationText = document.getElementById("durationText");
  const selectionText = document.getElementById("selectionText");
  const stemsRoot = document.getElementById("stemsRoot");

  const downloadZipBtn = document.getElementById("downloadZipBtn");
  const downloadMidiBtn = document.getElementById("downloadMidiBtn");
  const exportBtn = document.getElementById("actionExportSelection");
  const clearBtn = document.getElementById("actionClearSelection");

  // dropdown
  const ddPanel = document.getElementById("ddPanel");
  const ddList = document.getElementById("ddList");
  const ddClose = document.getElementById("ddClose");
  const ddSelectAll = document.getElementById("ddSelectAll");
  const ddClearAll = document.getElementById("ddClearAll");
  const ddDownload = document.getElementById("ddDownload");

  // export dropdown
  const expPanel = document.getElementById("expPanel");
  const expList = document.getElementById("expList");
  const expClose = document.getElementById("expClose");
  const expSelectAll = document.getElementById("expSelectAll");
  const expClearAll = document.getElementById("expClearAll");
  const expExport = document.getElementById("expExport");

  // SVG icons
  const ICONS = {
    play: `<svg viewBox="0 0 24 24"><path d="M9 7l10 5-10 5V7z"/></svg>`,
    pause: `<svg viewBox="0 0 24 24"><path d="M7 6h3v12H7zM14 6h3v12h-3z"/></svg>`,
    vocals: `<svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V7a3 3 0 0 0-6 0v4a3 3 0 0 0 3 3z"/><path d="M19 11a7 7 0 0 1-14 0"/><path d="M12 18v3"/></svg>`,
    drums: `<svg viewBox="0 0 24 24"><path d="M6 9c0 2 3 4 6 4s6-2 6-4-3-4-6-4-6 2-6 4z"/><path d="M6 9v6c0 2 3 4 6 4s6-2 6-4V9"/></svg>`,
    bass: `<svg viewBox="0 0 24 24"><path d="M14 3l7 7"/><path d="M13 4l1 1-9 9-2 6 6-2 9-9 1 1"/></svg>`,
    inst: `<svg viewBox="0 0 24 24"><path d="M4 18h16"/><path d="M6 18V6"/><path d="M10 18V8"/><path d="M14 18V10"/><path d="M18 18V7"/></svg>`
  };

  function labelIcon(label){
    const l = (label||"").toLowerCase();
    if (l.includes("voc")) return ICONS.vocals;
    if (l.includes("drum")) return ICONS.drums;
    if (l.includes("bass")) return ICONS.bass;
    return ICONS.inst;
  }

  function fmtTime(sec){
    sec = Math.max(0, sec || 0);
    const m = Math.floor(sec/60);
    const s = Math.floor(sec%60);
    return `${m}:${String(s).padStart(2,"0")}`;
  }

  function escapeHtml(s){
    return String(s||"")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;");
  }

  function setBtnLoading(btn, loading, labelWhenLoading){
    if (!btn) return;
    btn.disabled = !!loading;
    btn.dataset._orig = btn.dataset._orig || btn.textContent;
    btn.textContent = loading ? (labelWhenLoading || "Working…") : btn.dataset._orig;
    btn.classList.toggle("is-loading", !!loading);
  }

  function triggerDownload(url){
    if (!url) return;
    window.location.href = url;
  }

  // Downloads config
  const downloads = (DATA.downloads || {});
  let midiUrl = downloads.midi_url || "";

  // ========= Spacebar play/pause =========
  function isTypingTarget(el){
    if (!el) return false;
    const t = (el.tagName||"").toLowerCase();
    return t === "input" || t === "textarea" || t === "select" || el.isContentEditable;
  }

  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    if (isTypingTarget(document.activeElement)) return;
    e.preventDefault();
    // toggle play/pause
    playBtn?.click();
  }, { passive:false });

  // ========= Download MIDI =========
  downloadMidiBtn?.addEventListener("click", async () => {
    if (midiUrl) return triggerDownload(midiUrl);
    if (!apiBase) return alert("API not available.");

    try{
      setBtnLoading(downloadMidiBtn, true, "Generating MIDI…");
      const r = await fetch(`${apiBase}/build_midi`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ job: jobName })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "MIDI failed");
      midiUrl = j.midi_url || "";
      if (!midiUrl) throw new Error("No MIDI URL returned");
      setBtnLoading(downloadMidiBtn, false);
      triggerDownload(midiUrl);
    }catch(err){
      setBtnLoading(downloadMidiBtn, false);
      alert(String(err));
    }
  });

  // ========= Stems dropdown (custom) =========
  function openDropdown(){
    ddPanel.hidden = false;
    // close on outside click
    setTimeout(() => {
      const onDoc = (ev) => {
        if (!ddPanel.contains(ev.target) && ev.target !== downloadZipBtn){
          closeDropdown();
          document.removeEventListener("mousedown", onDoc, true);
        }
      };
      document.addEventListener("mousedown", onDoc, true);
    }, 0);
  }

  function closeDropdown(){
    ddPanel.hidden = true;
  }

  downloadZipBtn?.addEventListener("click", () => {
    if (ddPanel.hidden) openDropdown();
    else closeDropdown();
  });

  ddClose?.addEventListener("click", closeDropdown);

  // Build checkbox list from DATA.stems (use file names for API)
  const stemItems = DATA.stems.map(s => ({
    label: s.label,
    file: s.file, // wav filename
  })).filter(x => !!x.file);

  function renderDD(){
    ddList.innerHTML = "";
    stemItems.forEach((it) => {
      const row = document.createElement("div");
      row.className = "dd-item";
      row.innerHTML = `
        <label>
          <input type="checkbox" class="dd-check" data-file="${escapeHtml(it.file)}" checked />
          <span>${escapeHtml(it.label)}</span>
        </label>
      `;
      ddList.appendChild(row);
    });
  }
  renderDD();

  ddSelectAll?.addEventListener("click", () => {
    ddList.querySelectorAll(".dd-check").forEach(ch => ch.checked = true);
  });
  ddClearAll?.addEventListener("click", () => {
    ddList.querySelectorAll(".dd-check").forEach(ch => ch.checked = false);
  });

  function selectedFormat(){
    const el = document.querySelector('input[name="fmt"]:checked');
    return (el && el.value) ? el.value : "wav";
  }

  function selectedFiles(){
    const out = [];
    ddList.querySelectorAll(".dd-check").forEach(ch => {
      if (ch.checked) out.push(ch.getAttribute("data-file"));
    });
    return out.filter(Boolean);
  }

  ddDownload?.addEventListener("click", async () => {
    if (!apiBase) return alert("API not available.");
    const files = selectedFiles();
    if (!files.length) return alert("Select at least one stem.");
    const fmt = selectedFormat();

    try{
      setBtnLoading(ddDownload, true, "Building…");
      const r = await fetch(`${apiBase}/download_stems_custom`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ job: jobName, files, format: fmt })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "Download failed");
      const url = j.zip_url;
      setBtnLoading(ddDownload, false);
      closeDropdown();
      triggerDownload(url);
    }catch(err){
      setBtnLoading(ddDownload, false);
      alert(String(err));
    }
  });

  // ========= Audio engine =========
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();

  const master = ctx.createGain();
  master.gain.value = 1.0;
  master.connect(ctx.destination);

  const stems = DATA.stems.map(s => ({
    label: s.label,
    uri: s.audio_data_uri,
    buffer: null,
    source: null,
    gain: ctx.createGain(),
    vol: 1.0,
    muted: false,
    solo: false,
  }));
  stems.forEach(s => { s.gain.gain.value = 1.0; s.gain.connect(master); });

  let isPlaying = false;
  let startTime = 0;
  let startOffset = 0;
  let durationSec = 0;

  // Selection state
  let selA = null;
  let selB = null;
  let dragging = false;

  function updateMix(){
    const anySolo = stems.some(s => s.solo);
    stems.forEach(s => {
      const hear = anySolo ? s.solo : true;
      const g = (!hear || s.muted) ? 0 : s.vol;
      s.gain.gain.setTargetAtTime(g, ctx.currentTime, 0.01);
    });
  }

  async function decodeAll(){
    for (const s of stems){
      if (s.buffer) continue;
      if (!s.uri) continue;
      const resp = await fetch(s.uri);
      const arr = await resp.arrayBuffer();
      s.buffer = await ctx.decodeAudioData(arr);
    }
    durationSec = stems.reduce((mx, s) => Math.max(mx, s.buffer ? s.buffer.duration : 0), 0);
    durationText.textContent = `Duration: ${durationSec ? fmtTime(durationSec) : "—"}`;
  }

  function stopSources(){
    stems.forEach(s => {
      if (s.source){
        try { s.source.stop(); } catch(e){}
        try { s.source.disconnect(); } catch(e){}
        s.source = null;
      }
    });
  }

  function makeSources(at){
    stopSources();
    stems.forEach(s => {
      if (!s.buffer) return;
      const src = ctx.createBufferSource();
      src.buffer = s.buffer;
      src.connect(s.gain);
      s.source = src;
    });
    startTime = ctx.currentTime;
    startOffset = at || 0;
    stems.forEach(s => { if (s.source) s.source.start(ctx.currentTime, startOffset); });
    isPlaying = true;
    playBtn.innerHTML = ICONS.pause;
    requestAnimationFrame(tick);
  }

  function currentPos(){
    if (!isPlaying) return startOffset || 0;
    return (ctx.currentTime - startTime) + (startOffset || 0);
  }

  function updatePlayheads(sec){
    const pct = durationSec ? (sec/durationSec) : 0;
    document.querySelectorAll(".playhead").forEach(ph => {
      ph.style.left = `${(pct*100).toFixed(4)}%`;
    });
  }

  function updateSelectionText(){
    if (selA == null || selB == null || !durationSec){
      selectionText.textContent = "Selection: —";
      exportBtn.disabled = true;
      clearBtn.disabled = true;
      return;
    }
    const s = Math.min(selA, selB);
    const e = Math.max(selA, selB);
    selectionText.textContent = `Selection: ${fmtTime(s)}–${fmtTime(e)}`;
    const ok = (e - s) > 0.05;
    exportBtn.disabled = !ok;
    clearBtn.disabled = false;
  }

  function renderSelectionOverlay(){
    document.querySelectorAll(".selection").forEach(sel => {
      if (selA == null || selB == null || !durationSec){
        sel.style.display = "none";
        sel.style.left = "0%";
        sel.style.width = "0%";
      } else {
        const s = Math.min(selA, selB);
        const e = Math.max(selA, selB);
        const left = (s / durationSec) * 100;
        const width = ((e - s) / durationSec) * 100;
        sel.style.display = "block";
        sel.style.left = `${left.toFixed(4)}%`;
        sel.style.width = `${width.toFixed(4)}%`;
      }
    });
  }

  function clearSelection(){
    selA = selB = null;
    updateSelectionText();
    renderSelectionOverlay();
  }

  // ========= Export dropdown (selection) =========
  function openExportDropdown(){
    if (!expPanel) return;
    expPanel.hidden = false;
    setTimeout(() => {
      const onDoc = (ev) => {
        if (!expPanel.contains(ev.target) && ev.target !== exportBtn){
          closeExportDropdown();
          document.removeEventListener("mousedown", onDoc, true);
        }
      };
      document.addEventListener("mousedown", onDoc, true);
    }, 0);
  }
  function closeExportDropdown(){
    if (!expPanel) return;
    expPanel.hidden = true;
  }

  exportBtn?.addEventListener("click", () => {
    if (!expPanel) return;
    if (expPanel.hidden) openExportDropdown();
    else closeExportDropdown();
  });
  expClose?.addEventListener("click", closeExportDropdown);

  // Build checkbox list for export (same stems as Download Stems)
  function renderExportList(){
    if (!expList) return;
    expList.innerHTML = "";
    stemItems.forEach((it) => {
      const row = document.createElement("div");
      row.className = "dd-item";
      row.innerHTML = `
        <label>
          <input type="checkbox" class="exp-check" data-file="${escapeHtml(it.file)}" checked />
          <span>${escapeHtml(it.label)}</span>
        </label>
      `;
      expList.appendChild(row);
    });
  }
  renderExportList();

  expSelectAll?.addEventListener("click", () => {
    expList?.querySelectorAll(".exp-check").forEach(ch => ch.checked = true);
  });
  expClearAll?.addEventListener("click", () => {
    expList?.querySelectorAll(".exp-check").forEach(ch => ch.checked = false);
  });

  function selectedExportFormat(){
    const el = document.querySelector('input[name="expfmt"]:checked');
    return (el && el.value) ? el.value : "wav";
  }
  function selectedExportFiles(){
    const out = [];
    expList?.querySelectorAll(".exp-check").forEach(ch => {
      if (ch.checked) out.push(ch.getAttribute("data-file"));
    });
    return out.filter(Boolean);
  }

  expExport?.addEventListener("click", async () => {
    if (selA == null || selB == null || !durationSec) return;
    if (!apiBase) return alert("API not available.");

    const start_s = Math.min(selA, selB);
    const end_s = Math.max(selA, selB);

    const files = selectedExportFiles();
    if (!files.length) return alert("Select at least one stem.");
    const format = selectedExportFormat();

    try{
      setBtnLoading(expExport, true, "Exporting…");
      const r = await fetch(`${apiBase}/export_selection`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ job: jobName, start_s, end_s, files, format })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "Export failed");
      setBtnLoading(expExport, false);
      closeExportDropdown();
      triggerDownload(j.zip_url);
    }catch(err){
      setBtnLoading(expExport, false);
      alert(String(err));
    }
  });

clearBtn?.addEventListener("click", () => clearSelection());

  // Wave visuals
  function barsHtml(count){
    let out = "";
    for (let i=0;i<count;i++){
      const h = 10 + Math.floor((Math.sin(i*0.20)+1)*12 + (i%9));
      out += `<div class="bar" style="height:${h}px"></div>`;
    }
    return out;
  }

  function tick(){
    if (!isPlaying) return;
    const pos = currentPos();
    updatePlayheads(pos);

    if (durationSec && pos >= durationSec - 0.02){
      stopSources();
      isPlaying = false;
      startOffset = 0;
      playBtn.innerHTML = ICONS.play;
      updatePlayheads(0);
      return;
    }
    requestAnimationFrame(tick);
  }

  function buildRow(s) {
    const row = document.createElement("div");
    row.className = "sd-stem";
    row.innerHTML = `
      <div class="sd-rowtop">
        <div class="sd-label">
          <div class="sd-glyph">${labelIcon(s.label)}</div>
          <div class="sd-name">${escapeHtml(s.label)}</div>
        </div>
        <div class="sd-controls">
          <div class="seg">
            <button class="solo" type="button">Solo</button>
            <button class="mute" type="button">Mute</button>
          </div>
          <input class="vol" type="range" min="0" max="1" step="0.01" value="1"/>
        </div>
      </div>
      <div class="wavewrap" tabindex="0">
        <div class="selection" style="display:none"></div>
        <div class="playhead" style="left:0%"></div>
        <div class="bars">${barsHtml(260)}</div>
      </div>
    `;

    const soloBtn = row.querySelector(".solo");
    const muteBtn = row.querySelector(".mute");
    const vol = row.querySelector(".vol");
    const wave = row.querySelector(".wavewrap");

    soloBtn.addEventListener("click", () => {
      s.solo = !s.solo;
      soloBtn.classList.toggle("active", s.solo);
      updateMix();
    });

    muteBtn.addEventListener("click", () => {
      s.muted = !s.muted;
      muteBtn.classList.toggle("active", s.muted);
      updateMix();
    });

    vol.addEventListener("input", () => {
      s.vol = parseFloat(vol.value || "1");
      updateMix();
    });

    // click seek
    wave.addEventListener("click", async (ev) => {
      if (!durationSec) return;
      if (dragging) return;
      await ctx.resume();
      const rect = wave.getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width;
      const sec = Math.max(0, Math.min(durationSec, x * durationSec));
      if (isPlaying) makeSources(sec);
      else { startOffset = sec; updatePlayheads(sec); }
    });

    // drag selection
    wave.addEventListener("pointerdown", async (ev) => {
      if (!durationSec) return;
      await ctx.resume();
      dragging = true;
      const rect = wave.getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width;
      selA = Math.max(0, Math.min(durationSec, x * durationSec));
      selB = selA;
      updateSelectionText();
      renderSelectionOverlay();
      wave.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });

    wave.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const rect = wave.getBoundingClientRect();
      const x = (ev.clientX - rect.left) / rect.width;
      selB = Math.max(0, Math.min(durationSec, x * durationSec));
      updateSelectionText();
      renderSelectionOverlay();
    });

    wave.addEventListener("pointerup", () => { dragging = false; });

    return row;
  }

  // Init UI
  playBtn.innerHTML = ICONS.play;
  stemsRoot.innerHTML = "";
  stems.forEach(s => stemsRoot.appendChild(buildRow(s)));

  if (DATA.track){
    trackTitle.textContent = DATA.track.title || "—";
    pillBpm.textContent = `BPM: ${DATA.track.bpm || "—"}`;
    pillKey.textContent = `Key: ${DATA.track.key || "—"}`;
  }

  playBtn.addEventListener("click", async () => {
    await ctx.resume();
    await decodeAll();
    updateMix();

    if (!isPlaying) {
      makeSources(startOffset || 0);
    } else {
      const pos = currentPos();
      stopSources();
      isPlaying = false;
      playBtn.innerHTML = ICONS.play;
      startOffset = pos;
      updatePlayheads(pos);
    }
  });

  clearSelection();
})();
