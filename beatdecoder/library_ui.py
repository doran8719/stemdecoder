# library_ui.py

import streamlit as st
from pathlib import Path
import json
from typing import List, Dict, Any
import shutil
from datetime import datetime


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_metadata(job_dir: Path) -> Dict[str, Any]:
    meta_path = job_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def infer_song_name(meta: Dict[str, Any], job_dir: Path) -> str:
    if "song_name" in meta and meta["song_name"]:
        return meta["song_name"]
    # folder name fallback
    name = job_dir.name
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def get_song_folders(jobs_root: Path) -> Dict[str, List[Path]]:
    """
    Returns:
        {
          "Song A": [job1_dir, job2_dir],
          "Song B": [job3_dir ...],
        }
    """
    if not jobs_root.exists():
        return {}

    all_jobs = [
        p for p in jobs_root.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    ]

    songs: Dict[str, List[Path]] = {}

    for job in all_jobs:
        meta = load_metadata(job)
        song_name = infer_song_name(meta, job)

        if song_name not in songs:
            songs[song_name] = []
        songs[song_name].append(job)

    # Sort runs of each song by date
    for s in songs:
        songs[s] = sorted(
            songs[s],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )

    return songs


def human_label(meta: Dict[str, Any], job_dir: Path) -> str:
    label = meta.get("job_label") or job_dir.name
    created = meta.get("created_at") or datetime.fromtimestamp(
        job_dir.stat().st_mtime).isoformat()
    return f"{label} — {created}"


# ---------------------------------------------------------------------------
# RENDER LIBRARY UI
# ---------------------------------------------------------------------------

def render_library_ui(
    workspace_dir: Path,
    on_select_job_callback
):
    """
    Main UI for the Library tab.

    Args:
        workspace_dir: Path to workspace/ root.
        on_select_job_callback: fn(job_dir: Path) called when user selects a run.
    """

    st.header("📚 Library Browser")

    jobs_root = workspace_dir / "jobs"
    song_map = get_song_folders(jobs_root)

    if not song_map:
        st.info("No processed jobs found. Process a track first.")
        return

    # -----------------------------------------
    # Search filters
    # -----------------------------------------
    st.subheader("🔍 Filters")

    search_name = st.text_input("Search by song name:")
    bpm_min = st.number_input("Min BPM", min_value=0, max_value=400, value=0)
    bpm_max = st.number_input("Max BPM", min_value=0, max_value=400, value=400)
    key_list = ["Any", "C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]
    key_choice = st.selectbox("Key:", key_list, index=0)

    st.markdown("---")

    # -----------------------------------------
    # SONG GRID
    # -----------------------------------------
    st.subheader("🎵 Songs")

    # 2-column grid
    cols = st.columns(2)

    col_idx = 0

    # Sort alphabetically
    for song in sorted(song_map.keys()):
        # Apply text filter
        if search_name and search_name.lower() not in song.lower():
            continue

        with cols[col_idx]:
            with st.expander(f"🎧 {song}", expanded=False):
                runs = song_map[song]

                # Render runs
                for run_dir in runs:
                    meta = load_metadata(run_dir)

                    # BPM filter
                    bpm = meta.get("bpm")
                    if isinstance(bpm, (int, float)):
                        if bpm < bpm_min or bpm > bpm_max:
                            continue

                    # Key filter
                    if key_choice != "Any":
                        if meta.get("key") != key_choice:
                            continue

                    run_label = human_label(meta, run_dir)

                    st.write(f"**{run_label}**")
                    c1, c2 = st.columns([2, 1])

                    with c1:
                        st.write(f"BPM: {meta.get('bpm', '?')}")
                        st.write(f"Key: {meta.get('key', '?')} {meta.get('mode', '')}")

                    with c2:
                        if st.button("Load", key=f"load-{run_dir.name}"):
                            on_select_job_callback(run_dir)

                        if st.button("Delete", key=f"del-{run_dir.name}"):
                            try:
                                shutil.rmtree(run_dir)
                                st.warning(f"Deleted run: {run_dir}")
                            except Exception as e:
                                st.error(f"Could not delete: {e}")

                    st.markdown("---")

        col_idx = (col_idx + 1) % 2
