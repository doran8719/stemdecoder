import base64
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_multi_stem_player(
    stem_paths: list[Path],
    meta: dict,
    title: str = "Multi-Stem Player"
):
    """
    Fadr-style synchronized mixer:
        • Uses the first stem as master clock
        • Each stem gets Mute/Solo
        • Chord display updates during playback
        • All stems preloaded as base64 data URLs
    """

    # ------------------------------------------------------------------
    # Convert stems to base64
    # ------------------------------------------------------------------
    stems_data = []
    for p in stem_paths:
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            stems_data.append({
                "name": p.stem,
                "data_url": f"data:audio/wav;base64,{b64}",
            })
        except Exception:
            continue

    if not stems_data:
        st.write("No stems available.")
        return

    # ------------------------------------------------------------------
    # Prepare chord data
    # ------------------------------------------------------------------
    chord_segments = meta.get("chords") or meta.get("chord_segments") or []
    chords_json = json.dumps(chord_segments)
    stems_json = json.dumps(stems_data)

    player_id = f"mix_{uuid.uuid4().hex}"

    # ------------------------------------------------------------------
    # HTML + JS player
    # ------------------------------------------------------------------
    html = f"""
<div id="{player_id}" style="border:1px solid #444;padding:12px;border-radius:8px;margin-bottom:12px;font-family:sans-serif;">
  <div style="margin-bottom:6px;font-weight:600;font-size:16px;">{title}</div>

  <!-- Chord display -->
  <div style="margin-bottom:8px;font-size:14px;">
    <strong>Chord:</strong>
    <span id="{player_id}_chord">—</span>
  </div>

  <!-- Transport -->
  <div style="margin-bottom:10px;">
    <button id="{player_id}_play">Play</button>
    <button id="{player_id}_pause" style="margin-left:6px;">Pause</button>
    <button id="{player_id}_stop" style="margin-left:6px;">Stop</button>
  </div>

  <!-- Stem table -->
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead>
      <tr>
        <th style="text-align:left;padding:4px;">Stem</th>
        <th style="text-align:center;padding:4px;">Mute</th>
        <th style="text-align:center;padding:4px;">Solo</th>
      </tr>
    </thead>
    <tbody id="{player_id}_rows"></tbody>
  </table>
</div>

<script>
(function() {{
  const stems = {stems_json};
  const chords = {chords_json};
  const pid = "{player_id}";
  const rows = document.getElementById(pid + "_rows");
  const chordSpan = document.getElementById(pid + "_chord");

  // Create audio elements (one per stem)
  const audios = stems.map((s, idx) => {{
    const a = new Audio();
    a.src = s.data_url;
    a.preload = "auto";
    a.crossOrigin = "anonymous";
    a.dataset.idx = idx;
    return a;
  }});

  // Stem mute/solo state
  const state = stems.map(() => ({{
    mute: false,
    solo: false
  }}));

  function applyVolumes() {{
    const anySolo = state.some(s => s.solo);
    audios.forEach((a, i) => {{
      if (anySolo) {{
        a.muted = !state[i].solo;
      }} else {{
        a.muted = state[i].mute;
      }}
    }});
  }}

  // Build the table
  stems.forEach((stem, i) => {{
    const tr = document.createElement("tr");

    const nameTd = document.createElement("td");
    nameTd.textContent = stem.name;
    nameTd.style.padding = "2px 4px";
    tr.appendChild(nameTd);

    const muteTd = document.createElement("td");
    muteTd.style.textAlign = "center";
    const muteCb = document.createElement("input");
    muteCb.type = "checkbox";
    muteCb.addEventListener("change", () => {{
      state[i].mute = muteCb.checked;
      applyVolumes();
    }});
    muteTd.appendChild(muteCb);
    tr.appendChild(muteTd);

    const soloTd = document.createElement("td");
    soloTd.style.textAlign = "center";
    const soloCb = document.createElement("input");
    soloCb.type = "checkbox";
    soloCb.addEventListener("change", () => {{
      state[i].solo = soloCb.checked;
      applyVolumes();
    }});
    soloTd.appendChild(soloCb);
    tr.appendChild(soloTd);

    rows.appendChild(tr);
  }});

  // Master audio (first stem)
  const master = audios[0];

  // Chord switching logic
  function updateChord(t) {{
    if (!chords || chords.length === 0) {{
      chordSpan.textContent = "—";
      return;
    }}
    let current = "—";
    for (let i = 0; i < chords.length; i++) {{
      const c = chords[i] || {{}};
      const s = typeof c.start === "number" ? c.start : 0;
      const e = typeof c.end === "number" ? c.end : 1e9;
      if (t >= s && t < e) {{
        current = c.chord || c.label || "—";
        break;
      }}
    }}
    chordSpan.textContent = current;
  }}

  master.addEventListener("timeupdate", () => {{
    updateChord(master.currentTime);
  }});

  // Transport functions
  function playAll(reset=false) {{
    if (reset) {{
      audios.forEach(a => {{ a.currentTime = 0; }});
    }}
    applyVolumes();
    audios.forEach(a => a.play().catch(() => {{}}));
  }}

  function pauseAll() {{
    audios.forEach(a => a.pause());
  }}

  function stopAll() {{
    audios.forEach(a => {{
      a.pause();
      a.currentTime = 0;
    }});
    updateChord(0);
  }}

  // Hook up buttons
  document.getElementById(pid + "_play").onclick = () => playAll(false);
  document.getElementById(pid + "_pause").onclick = () => pauseAll();
  document.getElementById(pid + "_stop").onclick = () => stopAll();

}})();
</script>
"""

    components.html(html, height=300 + len(stems_data) * 24, scrolling=False)
