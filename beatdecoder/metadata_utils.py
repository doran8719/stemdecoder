import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import shutil
import time
import requests
import io
import zipfile
import base64
import uuid
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict, Counter
import math
import io  # for in-memory ZIPs



# Add global styles immediately after imports:
def inject_global_styles():
    st.markdown(
        """
        <style>
        /* APP SHELL – light, minimal, Apple-ish */
        .stApp {
            background: radial-gradient(circle at top, #f9fafb 0, #f3f4f6 38%, #e5e7eb 100%);
            color: #111827;
            font-family: -apple-system, BlinkMacSystemFont, system-ui,
                         "SF Pro Text", "SF Pro Display",
                         "Helvetica Neue", Arial, sans-serif;
        }

        .block-container {
            max-width: 1320px;   /* wider page */
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }


        h1, h2, h3, h4 {
            letter-spacing: 0.01em;
            color: #111827;
        }

        /* GLOBAL MAIN CONTENT TEXT COLOR */
        .block-container,
        .block-container p,
        .block-container span,
        .block-container label,
        .block-container li,
        .block-container div {
            color: #111827;
        }

        /* TABS – Process Track(s) / My Songs / Settings */
        div[data-baseweb="tab"],
        button[data-baseweb="tab"],
        div[role="tab"],
        button[role="tab"] {
            color: #4b5563 !important;
            background-color: transparent !important;
            font-weight: 500;
            font-size: 0.9rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px !important;
            margin-right: 0.25rem;
        }

        div[data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"],
        div[role="tab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            background-color: #ffffff !important;
            color: #111827 !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.09);
            border: 1px solid #e5e7eb !important;
            font-weight: 600;
        }

        div[data-baseweb="tab"]:hover,
        button[data-baseweb="tab"]:hover,
        div[role="tab"]:hover,
        button[role="tab"]:hover {
            background-color: rgba(15, 23, 42, 0.03) !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] * {
            color: #111827 !important;
            font-size: 0.9rem;
        }

        /* PAGE SUBTITLE UNDER MAIN HEADER */
        .bd-page-subtitle {
            font-size: 0.86rem;
            color: #6b7280;
            margin-top: -0.25rem;
            margin-bottom: 0.8rem;
        }

        /* SECTION CARDS – Player / Filters / Analytics */
        .bd-section {
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e5e5ea;
            padding: 14px 16px 12px 16px;
            margin-bottom: 16px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
        }

        .bd-section-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .bd-section-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #111827;
        }

        .bd-section-caption {
            font-size: 0.75rem;
            color: #6b7280;
        }

        /* SONG CARDS */
        .bd-card {
            position: relative;
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e5e5ea;
            padding: 14px 16px 11px 16px;
            margin-bottom: 14px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            transition: transform 120ms ease-out, box-shadow 120ms ease-out,
                        border-color 120ms ease-out;
        }

        .bd-card::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 16px;
            border-top: 2px solid rgba(96, 165, 250, 0.35);
            opacity: 0;
            pointer-events: none;
            transition: opacity 140ms ease-out;
        }

        .bd-card:hover {
            transform: translateY(-1.5px);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
            border-color: #d4d4dd;
        }

        .bd-card:hover::before {
            opacity: 1;
        }

        .bd-card-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 4px;
        }

        .bd-song-title {
            font-size: 1.02rem;
            font-weight: 600;
            color: #111827;
        }

        .bd-song-subtitle {
            font-size: 0.78rem;
            color: #6b7280;
        }

        .bd-pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .bd-pill-status-Idea { background: #eef2ff; color: #4f46e5; }
        .bd-pill-status-In-Progress { background: #fffbeb; color: #d97706; }
        .bd-pill-status-Ready { background: #ecfdf3; color: #16a34a; }
        .bd-pill-status-Final { background: #fff7ed; color: #ea580c; }


        .bd-pill-fav {
            margin-left: 6px;
            background: #fef9c3;
            color: #ca8a04;
        }

        .bd-tags-row {
            margin-top: 4px;
            font-size: 0.7rem;
            color: #4b5563;
        }

        .bd-tag {
            display: inline-flex;
            align-items: center;
            padding: 1px 7px;
            border-radius: 999px;
            background: #edfafe;
            color: #0369a1;
            margin-right: 4px;
            margin-top: 2px;
            border: 1px solid #e0f2fe;
        }

        .bd-metadata-row {
            margin-top: 6px;
            font-size: 0.74rem;
            color: #6b7280;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .bd-meta-label {
            font-weight: 500;
            color: #4b5563;
        }

        /* NOW-PLAYING HEADER IN PLAYER */
        .bd-now-playing {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 4px;
        }

        .bd-now-meta {
            display: flex;
            flex-direction: column;
        }

        .bd-now-title {
            font-size: 0.98rem;
            font-weight: 600;
            color: #111827;
        }

        .bd-now-path {
            font-size: 0.7rem;
            color: #9ca3af;
        }

        .bd-now-pill {
            padding: 4px 10px;
            border-radius: 999px;
            background: #f1f5f9;
            font-size: 0.72rem;
            color: #475569;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        /* BUTTONS – normal */
        .stButton > button {
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.36rem 1.25rem !important;
            border: 1px solid #d1d5db !important;
            background: #ffffff !important;
            color: #111827 !important;
        }

        .stButton > button:hover {
            border-color: #007aff !important;
            box-shadow: 0 0 0 1px rgba(0,122,255,0.22);
        }

        /* DOWNLOAD BUTTONS (e.g. MIDI) */
        .stDownloadButton > button {
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.36rem 1.25rem !important;
            border: 1px solid #d1d5db !important;
            background: #ffffff !important;
            color: #111827 !important;
        }

        .stDownloadButton > button:hover {
            border-color: #007aff !important;
            box-shadow: 0 0 0 1px rgba(0,122,255,0.22);
            background: #f9fafb !important;
        }

        /* METRICS (BPM / KEY) */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 999px;
            border: 1px solid #e5e5ea;
            padding: 4px 10px;
        }

        [data-testid="stMetricLabel"] {
            color: #6b7280 !important;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        [data-testid="stMetricValue"] {
            color: #111827 !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }

        /* FORM LABELS & FILTER TEXT */
        label,
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stCheckbox label,
        .stSlider label {
            color: #111827 !important;
            font-size: 0.78rem;
        }

        /* Selected value inside selectboxes (Version, Sort by, Key filter, etc.) */
        div[data-baseweb="select"] div {
            color: #111827 !important;
        }

        /* NUMBER INPUT VALUES (Min BPM, Max BPM, Page) */
        .stNumberInput input {
            color: #111827 !important;
        }

        input[type="number"],
        input[type="text"] {
            color: #111827 !important;
        }

        /* TEXT INPUTS / AREAS / SELECTS styling */
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            border-radius: 999px !important;
            border: 1px solid #d4d4dd !important;
            background: #f9fafb !important;
        }

        .stTextArea textarea {
            border-radius: 14px !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus,
        .stSelectbox > div > div:focus,
        .stMultiSelect > div > div:focus {
            border-color: #007aff !important;
            box-shadow: 0 0 0 1px rgba(0,122,255,0.18);
        }

        /* EXPANDERS AS CARDS + DARK TEXT INSIDE */
        details[data-testid="stExpander"] {
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            padding: 3px 10px 7px 10px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.04);
        }

        details[data-testid="stExpander"] * {
            color: #111827 !important;
        }

        summary {
            font-weight: 500;
        }

        /* HORIZONTAL RULES */
        hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 0.5rem 0 0.25rem 0;
        }

        /* SONG GRID: tiles that fill 4 columns */
/* SONG GRID: perfectly uniform square tiles */
.song-card .stButton > button {
    width: 100%;
    aspect-ratio: 1 / 1;              /* ⬅️ makes each tile a square */
    border-radius: 18px;
    border: 1px solid #e5e5ea;
    background: #ffffff;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    padding: 14px 16px;
    text-align: left;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;

    white-space: normal;              /* allow wrapping */
    word-wrap: break-word;
    overflow: hidden;                 /* extra text stays inside the square */

    font-size: 0.9rem;
    color: #111827;
    transition: transform 120ms ease-out, box-shadow 120ms ease-out,
                border-color 120ms ease-out;
}

.song-card .stButton > button:hover {
    transform: translateY(-1.5px);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
    border-color: #d4d4dd;
}


.song-card .stButton > button:hover {
    transform: translateY(-1.5px);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
    border-color: #d4d4dd;
}

/* main line: song title stronger */
.song-card-title {
    font-weight: 600;
    font-size: 0.95rem;
}

/* second line: bpm + key a bit softer */
.song-card-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 4px;
}

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_styles()

# ---------------------------------------------------------------------
# BASIC PATHS / DEFAULTS
# ---------------------------------------------------------------------

WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

DEFAULT_DEMUCS_MODEL = "htdemucs"
DEFAULT_STEMS_FOR_MIDI = ["bass", "other"]
DEFAULT_DEMUCS_DEVICE = "auto"

DEFAULT_WORKER_URL = "http://localhost:8000"

from processor import process_song  # your existing processor


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]
NOTE_INDEX = {n: i for i, n in enumerate(NOTE_NAMES)}


def format_time(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        return "0:00"
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}:{s:02d}"


def transpose_chord_name(chord_name: str, semitones: int) -> str:
    """
    Very simple chord transposer for chords in the format 'C major'/'C minor'.
    """
    if not chord_name:
        return chord_name
    parts = chord_name.split()
    if len(parts) < 2:
        return chord_name
    root = parts[0]
    quality = " ".join(parts[1:])

    if root not in NOTE_INDEX:
        return chord_name
    idx = NOTE_INDEX[root]
    new_idx = (idx + semitones) % 12
    new_root = NOTE_NAMES[new_idx]
    return f"{new_root} {quality}"


def parse_key_to_index(key: Optional[str]) -> Optional[int]:
    if not key:
        return None
    k = key.strip().upper()
    # normalize possible "B♭" etc if needed later; for now assume standard NOTE_NAMES
    if k in NOTE_INDEX:
        return NOTE_INDEX[k]
    # try things like "Bb"
    k = k.replace("B♭", "A#").replace("E♭", "D#")
    return NOTE_INDEX.get(k)


# ---------------------------------------------------------------------
# FILE / JOB HELPERS
# ---------------------------------------------------------------------

def clean_ableton_guide_text(raw: str) -> str:
    """
    Remove noisy sections like:
      - 'Estimated Chords (rough):'
      - 'Loudness-based Sections (rough):'
    from the Ableton import guide before displaying it.
    """
    lines = raw.splitlines()
    out_lines = []
    skip = False

    for line in lines:
        lower = line.strip().lower()

        # Start skipping when we hit one of the rough headings
        if (
            "estimated chords (rough" in lower
            or "loudness-based sections (rough" in lower
        ):
            skip = True
            continue

        if skip:
            # Stop skipping at a blank line or a new heading
            if lower == "" or lower.startswith("#") or lower.endswith(":"):
                skip = False
            # Either way, don't include the skipped lines
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


def save_uploaded_file(uploaded_file) -> Path:
    temp_dir = WORKSPACE_DIR / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = temp_dir / safe_name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def list_jobs() -> List[Path]:
    jobs_root = WORKSPACE_DIR / "jobs"
    if not jobs_root.exists():
        return []
    return sorted(
        [p for p in jobs_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_metadata(job_dir: Path) -> Dict[str, Any]:
    meta_path = job_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    # New: ensure markers list exists so we can store time-based notes
    meta.setdefault("markers", [])

    return meta



def infer_song_name(job_dir: Path, meta: Dict[str, Any]) -> str:
    if meta.get("song_name"):
        return meta["song_name"]
    name = job_dir.name
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def infer_created_at(job_dir: Path, meta: Dict[str, Any]) -> str:
    if meta.get("created_at"):
        return meta["created_at"]
    return datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat()


def load_index() -> List[Dict[str, Any]]:
    index_path = WORKSPACE_DIR / "index.json"
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def build_index_from_jobs() -> List[Dict[str, Any]]:
    jobs = list_jobs()
    entries: List[Dict[str, Any]] = []
    for job_dir in jobs:
        meta = load_metadata(job_dir)
        song_name = infer_song_name(job_dir, meta)
        created_at = infer_created_at(job_dir, meta)
        entry = {
            "job_id": meta.get("job_id", job_dir.name),
            "job_path": str(job_dir.resolve()),
            "song_name": song_name,
            "job_label": meta.get("job_label") or job_dir.name,
            "bpm": meta.get("bpm"),
            "key": meta.get("key"),
            "mode": meta.get("mode"),
            "created_at": created_at,
            "notes": meta.get("notes", ""),
            "tags": meta.get("tags", []),
            "status": meta.get("status", "Idea"),
            "favorite": meta.get("favorite", False),
        }
        entries.append(entry)

    index_path = WORKSPACE_DIR / "index.json"
    try:
        with open(index_path, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass
    return entries


def make_zip_for_job(job_dir: Path) -> Path:
    """
    Used for single-job downloads (Process tab, and selected job in global player).
    """
    zip_path = job_dir / f"{job_dir.name}.zip"
    if not zip_path.exists():
        base_name = job_dir / job_dir.name
        shutil.make_archive(str(base_name), "zip", root_dir=job_dir)
    return zip_path


def generate_smart_job_label(
    song_name: Optional[str],
    bpm: Optional[float],
    key: Optional[str],
    mode: Optional[str],
    model_name: Optional[str],
    fallback: str,
) -> str:
    """
    Build something like: 'Song Name [128 BPM, Am, htdemucs]'
    """
    parts = []
    if song_name:
        parts.append(song_name)
    else:
        parts.append(fallback)

    bracket_bits = []
    if bpm:
        bracket_bits.append(f"{bpm:.2f} BPM")
    if key:
        kmode = f"{key}{' ' + mode if mode else ''}".strip()
        bracket_bits.append(kmode)
    if model_name:
        bracket_bits.append(model_name)

    if bracket_bits:
        return f"{' '.join(parts)} [{' , '.join(bracket_bits)}]"

    return " ".join(parts)


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

def init_session_state():
    if "demucs_model" not in st.session_state:
        st.session_state["demucs_model"] = DEFAULT_DEMUCS_MODEL
    if "stems_for_midi" not in st.session_state:
        st.session_state["stems_for_midi"] = DEFAULT_STEMS_FOR_MIDI.copy()
    if "run_serum_analysis" not in st.session_state:
        st.session_state["run_serum_analysis"] = True
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if "demucs_device" not in st.session_state:
        st.session_state["demucs_device"] = DEFAULT_DEMUCS_DEVICE
    if "processing_backend" not in st.session_state:
        st.session_state["processing_backend"] = "Remote GPU worker"
    if "worker_url" not in st.session_state:
        st.session_state["worker_url"] = DEFAULT_WORKER_URL
    if "last_gpu_result" not in st.session_state:
        st.session_state["last_gpu_result"] = None
    if "current_player_job" not in st.session_state:
        st.session_state["current_player_job"] = None
    if "current_player_song_name" not in st.session_state:
        st.session_state["current_player_song_name"] = None


def log_to_session(msg: str):
    st.session_state["logs"].append(msg)


# ---------------------------------------------------------------------
# CHORDS + SECTION DETECTION (BACKEND ONLY, FOR PLAYER)
# ---------------------------------------------------------------------

def _pick_best_audio_for_analysis(job_root: Path, original_path: Optional[Path]) -> Optional[Path]:
    vocals = job_root / "vocals.wav"
    other = job_root / "other.wav"

    if vocals.exists() and other.exists():
        return None  # we'll mix them in memory

    candidates: List[Path] = []
    if original_path and original_path.exists():
        candidates.append(original_path)

    for name in ["original.wav", "mixture.wav", "original.mp3", "mixture.mp3"]:
        p = job_root / name
        if p.exists():
            candidates.append(p)

    wavs = sorted(job_root.glob("*.wav"))
    mp3s = sorted(job_root.glob("*.mp3"))
    candidates.extend(wavs)
    candidates.extend(mp3s)

    return candidates[0] if candidates else None


def _compute_sections_from_energy(y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
    duration = len(y) / sr
    if duration < 10.0:
        return [{"start": 0.0, "end": float(duration), "label": "Song"}]

    hop_length = 1024
    frame_times = librosa.frames_to_time(
        np.arange(0, 1 + len(y) // hop_length),
        sr=sr,
        hop_length=hop_length,
    )

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms = librosa.util.normalize(rms)

    S = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop_length))
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-6)
    flux = np.sqrt(((np.diff(S_norm, axis=1).clip(min=0)) ** 2).sum(axis=0))
    flux = np.concatenate([[0.0], flux])
    flux = librosa.util.normalize(flux)

    activity = 0.7 * rms + 0.3 * flux

    win = 5
    kernel = np.ones(win) / win
    activity_smooth = np.convolve(activity, kernel, mode="same")

    target_segments = 8
    frame_count = len(activity_smooth)
    seg_size = max(1, frame_count // target_segments)

    segments = []
    for i in range(0, frame_count, seg_size):
        j = min(frame_count, i + seg_size)
        seg_act = float(activity_smooth[i:j].mean())
        start_t = float(frame_times[i])
        end_t = float(frame_times[j - 1]) if j - 1 < len(frame_times) else float(duration)
        segments.append(
            {"start": start_t, "end": end_t, "activity": seg_act}
        )

    if not segments:
        return [{"start": 0.0, "end": float(duration), "label": "Song"}]

    sorted_idx = sorted(range(len(segments)), key=lambda k: segments[k]["activity"], reverse=True)
    top_idx = sorted_idx[: max(1, len(sorted_idx) // 3)]

    labels = [""] * len(segments)
    for i in range(len(segments)):
        if i == 0:
            labels[i] = "Intro"
        elif i == len(segments) - 1:
            labels[i] = "Outro"

    for idx in top_idx:
        center_pos = (segments[idx]["start"] + segments[idx]["end"]) / 2.0
        frac = center_pos / duration
        if frac < 0.35:
            label = "Chorus"
        elif frac > 0.7:
            label = "Drop"
        else:
            label = "Chorus"
        labels[idx] = label

    for i in range(len(segments)):
        if labels[i]:
            continue
        prev_act = segments[i - 1]["activity"] if i > 0 else segments[i]["activity"]
        next_act = segments[i + 1]["activity"] if i < len(segments) - 1 else segments[i]["activity"]
        this_act = segments[i]["activity"]
        if this_act < 0.6 * max(prev_act, next_act):
            labels[i] = "Break"
        elif next_act > this_act:
            labels[i] = "Build"
        else:
            labels[i] = "Verse"

    merged: List[Dict[str, Any]] = []
    cur_label = labels[0]
    cur_start = segments[0]["start"]
    cur_end = segments[0]["end"]

    for i in range(1, len(segments)):
        if labels[i] == cur_label:
            cur_end = segments[i]["end"]
        else:
            merged.append({"start": cur_start, "end": cur_end, "label": cur_label})
            cur_label = labels[i]
            cur_start = segments[i]["start"]
            cur_end = segments[i]["end"]
    merged.append({"start": cur_start, "end": cur_end, "label": cur_label})

    return merged


def analyze_chords_and_sections_from_audio(
    job_root: Path,
    original_audio: Optional[Path],
    max_duration_sec: float = 240.0,
) -> Dict[str, Any]:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    analysis_path = _pick_best_audio_for_analysis(job_root, original_audio)

    vocals = job_root / "vocals.wav"
    other = job_root / "other.wav"
    use_mix_stems = vocals.exists() and other.exists()

    if use_mix_stems:
        y_v, sr = librosa.load(str(vocals), sr=22050, mono=True)
        y_o, _ = librosa.load(str(other), sr=22050, mono=True)
        y = 0.7 * y_v + 0.3 * y_o
    else:
        if analysis_path is None or not analysis_path.exists():
            return {"chord_summary": "", "chords": [], "sections": []}
        y, sr = librosa.load(str(analysis_path), sr=22050, mono=True)

    duration = len(y) / sr
    if duration > max_duration_sec:
        y = y[: int(max_duration_sec * sr)]
        duration = max_duration_sec

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    if chroma.size == 0:
        sections = _compute_sections_from_energy(y, sr)
        return {"chord_summary": "", "chords": [], "sections": sections}

    major_int = [0, 4, 7]
    minor_int = [0, 3, 7]

    templates = []
    chord_names = []

    for root in range(12):
        tpl = np.zeros(12)
        for iv in major_int:
            tpl[(root + iv) % 12] = 1.0
        templates.append(tpl)
        chord_names.append(f"{NOTE_NAMES[root]} major")

    for root in range(12):
        tpl = np.zeros(12)
        for iv in minor_int:
            tpl[(root + iv) % 12] = 1.0
        templates.append(tpl)
        chord_names.append(f"{NOTE_NAMES[root]} minor")

    templates = np.array(templates)
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-6)
    scores = templates @ chroma_norm
    best_idx = scores.argmax(axis=0)

    chords_segments = []
    last_idx = None
    seg_start = 0
    n_frames = chroma.shape[1]

    for i in range(n_frames):
        idx = int(best_idx[i])
        if last_idx is None:
            last_idx = idx
            seg_start = i
            continue
        if idx != last_idx:
            s_t = float(librosa.frames_to_time(seg_start, sr=sr))
            e_t = float(librosa.frames_to_time(i, sr=sr))
            chord_name = chord_names[last_idx]
            chords_segments.append(
                {"start": s_t, "end": e_t, "chord": chord_name}
            )
            last_idx = idx
            seg_start = i

    if last_idx is not None:
        s_t = float(librosa.frames_to_time(seg_start, sr=sr))
        e_t = float(duration)
        chord_name = chord_names[last_idx]
        chords_segments.append(
            {"start": s_t, "end": e_t, "chord": chord_name}
        )

    seen = []
    for seg in chords_segments:
        ch = seg["chord"]
        if ch not in seen:
            seen.append(ch)
    chord_summary = " → ".join(seen)

    sections = _compute_sections_from_energy(y, sr)

    return {
        "chord_summary": chord_summary,
        "chords": chords_segments,
        "sections": sections,
    }


def compute_and_store_chords(
    job_root: Path,
    original_audio: Optional[Path] = None,
) -> Dict[str, Any]:
    if not job_root.exists():
        return {}

    try:
        chord_data = analyze_chords_and_sections_from_audio(job_root, original_audio)
    except Exception as e:
        log_to_session(f"Chord/section analysis error: {e}")
        return {}

    meta = load_metadata(job_root)
    meta.update(
        {
            "chord_summary": chord_data.get("chord_summary", ""),
            "chords": chord_data.get("chords", []),
            "sections": chord_data.get("sections", []),
        }
    )

    meta_path = job_root / "metadata.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        log_to_session(f"Failed writing updated metadata: {e}")

    return {
        "chord_summary": meta.get("chord_summary", ""),
        "chords": meta.get("chords", []),
        "sections": meta.get("sections", []),
    }


# ---------------------------------------------------------------------
# EXPORT PRESETS
# ---------------------------------------------------------------------

def build_export_preset_zip(job_dir: Path, meta: Dict[str, Any], preset: str) -> io.BytesIO:
    """
    Build an in-memory ZIP for an export preset.
    preset: "Ableton Template" or "Remix Pack"
    """
    stems = [p for p in job_dir.iterdir() if p.suffix.lower() == ".wav"]
    midi_dir = job_dir / "midi"
    midi_files = (
        [p for p in midi_dir.iterdir() if p.suffix.lower() == ".mid"]
        if midi_dir.exists()
        else []
    )
    guide_path = job_dir / "ABLETON_IMPORT.txt"

    song_name = meta.get("song_name") or job_dir.name
    safe_song = song_name.replace(" ", "_")

    if preset == "Ableton Template":
        root_folder = f"{safe_song}_AbletonTemplate"
    else:
        root_folder = f"{safe_song}_RemixPack"

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # stems
        for stem in stems:
            if preset == "Remix Pack":
                # simple categorization
                name_low = stem.stem.lower()
                if any(k in name_low for k in ["drum", "kick", "snare", "perc"]):
                    sub = "Stems/Drums"
                elif "bass" in name_low:
                    sub = "Stems/Bass"
                elif any(k in name_low for k in ["voc", "voice"]):
                    sub = "Stems/Vocals"
                else:
                    sub = "Stems/Other"
            else:
                sub = "Stems"

            arcname = f"{root_folder}/{sub}/{stem.name}"
            zf.write(stem, arcname=arcname)

        # midi
        for midi in midi_files:
            arcname = f"{root_folder}/MIDI/{midi.name}"
            zf.write(midi, arcname=arcname)

        # guide
        if guide_path.exists():
            zf.write(guide_path, arcname=f"{root_folder}/Guide/ABLETON_IMPORT.txt")

        # simple metadata
        meta_lines = [
            f"Song: {song_name}",
            f"BPM: {meta.get('bpm', 'Unknown')}",
            f"Key: {meta.get('key', 'Unknown')} {meta.get('mode', '')}".strip(),
            f"Export preset: {preset}",
            "",
            "Generated by BeatDecoder",
        ]
        meta_bytes = "\n".join(meta_lines).encode("utf-8")
        zf.writestr(f"{root_folder}/metadata.txt", meta_bytes)

    mem_zip.seek(0)
    return mem_zip


# ---------------------------------------------------------------------
# MINI PLAYER (USED GLOBALLY + PER-JOB WHEN NEEDED)
# ---------------------------------------------------------------------

def render_multi_stem_player(
    stem_files: List[Path],
    meta: Dict[str, Any],
    title: str = "Stems Mixer",
):
    """
    Multi-stem mini mixer with:

      - Single Play/Pause toggle (no Stop button)
      - Mute / solo per stem
      - Chord display from meta["chords"] / meta["chord_segments"]
      - SoundCloud-style waveform per stem:
          * Playhead line that moves with playback
          * Played vs unplayed waveform colors
          * Click to seek (accurate to mouse position)
          * Click-and-drag to select a region (shared across stems)
      - Export selected region:
          * Uses JSZip in the browser
          * Respects solo/mute (only audible stems exported)
      - Visual markers:
          * meta["markers"] entries are drawn as red triangle markers
    """
    if not stem_files:
        st.info("No stems found to play.")
        return

    chords = meta.get("chords") or meta.get("chord_segments") or []
    markers = meta.get("markers", [])

    stems_data = []
    for p in sorted(stem_files):
        is_mix = any(
            kw in p.stem.lower()
            for kw in ("mix", "mixture", "full", "master", "song")
        )
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            data_url = f"data:audio/wav;base64,{b64}"
        except Exception:
            data_url = ""
        stems_data.append(
            {
                "name": p.stem,
                "data_url": data_url,
                "is_full_mix": is_mix,
            }
        )

    if not stems_data:
        st.info("No stems available for mixer.")
        return

    player_id = f"player_{uuid.uuid4().hex[:8]}"
    stems_json = json.dumps(stems_data)
    chords_json = json.dumps(chords)
    markers_json = json.dumps(markers)

    html = f"""
<div style="border:1px solid #e5e5ea;border-radius:12px;padding:12px;margin-bottom:16px;background:#ffffff;color:#111827;font-family:-apple-system,BlinkMacSystemFont,system-ui,'SF Pro Text','Helvetica Neue',Arial,sans-serif;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <button id="{player_id}_toggle" style="padding:4px 10px;border-radius:4px;border:none;background:#22c55e;color:#000;font-weight:600;cursor:pointer;font-size:13px;">
      ▶ Play
    </button>
    <div style="margin-left:12px;font-size:13px;">
      Chord: <span id="{player_id}_chord" style="font-weight:bold;">—</span>
    </div>
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:8px;">
    <thead>
      <tr style="border-bottom:1px solid #333;">
        <th style="text-align:left;padding:4px;">Stem</th>
        <th style="text-align:center;padding:4px;">Solo</th>
        <th style="text-align:center;padding:4px;">Mute</th>
      </tr>
    </thead>
    <tbody id="{player_id}_stem_rows"></tbody>
  </table>

  <div style="margin-top:10px;font-size:13px;font-weight:600;">
    Waveforms (click to seek, drag to select):
  </div>
  <div id="{player_id}_waveforms" style="margin-top:4px;"></div>
  <div id="{player_id}_selection_info"
       style="font-size:11px;color:#ccc;margin-top:4px;">
    Selected region: none (click + drag on any waveform)
  </div>

  <div style="margin-top:6px;">
    <button id="{player_id}_export"
            style="padding:4px 10px;border-radius:4px;border:none;background:#38bdf8;color:#000;font-weight:600;cursor:pointer;font-size:12px;">
      Export selected region (audible stems)
    </button>
  </div>

  <div style="font-size:11px;color:#999;margin-top:4px;">
    • Click a waveform to jump the playhead<br/>
    • Click + drag to highlight a region<br/>
    • Red arrows = saved favorite sections<br/>
    • Export respects solo/mute (only stems you can hear are exported)
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>

<script>
(function() {{
  const stems = {stems_json};
  const chords = {chords_json};
  const markers = {markers_json};
  const id = "{player_id}";

  const chordSpan = document.getElementById(id + "_chord");
  const toggleBtn = document.getElementById(id + "_toggle");
  const exportBtn = document.getElementById(id + "_export");
  const rowsBody = document.getElementById(id + "_stem_rows");
  const waveContainer = document.getElementById(id + "_waveforms");
  const selectionInfo = document.getElementById(id + "_selection_info");

  if (!stems || stems.length === 0) {{
    if (chordSpan) chordSpan.textContent = "No stems";
    return;
  }}

  function formatTime(sec) {{
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    const sInt = Math.floor(s);
    const t = Math.floor((s - sInt) * 10);
    return m + ":" + (sInt < 10 ? "0" + sInt : sInt) + "." + t;
  }}

  // Shared selection across stems
  let selectionStartSec = null;
  let selectionEndSec = null;

  function updateSelectionDisplay() {{
    if (!selectionInfo) return;
    if (
      selectionStartSec == null ||
      selectionEndSec == null ||
      selectionEndSec <= selectionStartSec + 0.01
    ) {{
      selectionInfo.textContent =
        "Selected region: none (click + drag on any waveform)";
    }} else {{
      selectionInfo.textContent =
        "Selected region: " +
        formatTime(selectionStartSec) +
        " → " +
        formatTime(selectionEndSec);
    }}
  }}

  // Audio per stem
  const audios = stems.map((s, idx) => {{
    const a = new Audio();
    a.src = s.data_url;
    a.preload = "auto";
    a.crossOrigin = "anonymous";
    a.dataset.index = idx;
    return a;
  }});

  const stemState = stems.map(() => ({{ mute: false, solo: false }}));

  function applyVolumes() {{
    const anySolo = stemState.some(s => s.solo);
    audios.forEach((a, idx) => {{
      if (anySolo) {{
        a.muted = !stemState[idx].solo;
      }} else {{
        a.muted = stemState[idx].mute;
      }}
    }});
  }}

  function getAudibleIndices() {{
    const anySolo = stemState.some(s => s.solo);
    const out = [];
    stemState.forEach((st, idx) => {{
      if (anySolo) {{
        if (st.solo) out.push(idx);
      }} else {{
        if (!st.mute) out.push(idx);
      }}
    }});
    return out;
  }}

  // Build rows
  stems.forEach((stem, idx) => {{
    const tr = document.createElement("tr");
    tr.style.borderBottom = "1px solid #222";

    const nameTd = document.createElement("td");
    nameTd.style.padding = "4px";
    nameTd.textContent = stem.name + (stem.is_full_mix ? " (mix)" : "");
    tr.appendChild(nameTd);

    const soloTd = document.createElement("td");
    soloTd.style.textAlign = "center";
    const soloCb = document.createElement("input");
    soloCb.type = "checkbox";
    soloCb.addEventListener("change", () => {{
      stemState[idx].solo = soloCb.checked;
      applyVolumes();
    }});
    soloTd.appendChild(soloCb);
    tr.appendChild(soloTd);

    const muteTd = document.createElement("td");
    muteTd.style.textAlign = "center";
    const muteCb = document.createElement("input");
    muteCb.type = "checkbox";
    muteCb.addEventListener("change", () => {{
      stemState[idx].mute = muteCb.checked;
      applyVolumes();
    }});
    muteTd.appendChild(muteCb);
    tr.appendChild(muteTd);

    rowsBody.appendChild(tr);
  }});

  // Waveforms
  const waveforms = [];   // per idx: {{ canvas, peaks, maxPeak, duration, audioBuffer }}
  const waveDurations = [];

  function findChordAtTime(t) {{
    if (!chords || chords.length === 0) return null;
    for (let i = 0; i < chords.length; i++) {{
      const c = chords[i] || {{}};
      const start = c.start ?? 0;
      const end = c.end ?? 0;
      if (t >= start && t < end) return c.chord || c.label || null;
    }}
    return null;
  }}

  function updateChordDisplay(t) {{
    if (!chordSpan) return;
    const c = findChordAtTime(t);
    chordSpan.textContent = c || "—";
  }}

  function getGlobalCurrentTime() {{
    if (!audios.length) return 0;
    let maxT = 0;
    audios.forEach(a => {{
      if (!isNaN(a.currentTime)) maxT = Math.max(maxT, a.currentTime);
    }});
    return maxT;
  }}

  function drawSingleWaveform(idx, currentTime) {{
    const wf = waveforms[idx];
    if (!wf || !wf.canvas) return;

    const canvas = wf.canvas;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const peaks = wf.peaks || [];
    const maxPeak = wf.maxPeak || 1.0;
    const duration = wf.duration || 1.0;

    ctx.clearRect(0, 0, W, H);

    // Light background
    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(0, 0, W, H);

    if (!peaks.length) return;

    const midY = H / 2;
    const samples = peaks.length;
    const progress = duration > 0 ? currentTime / duration : 0;
    const playX = Math.min(Math.max(progress * W, 0), W);

    // waveform
    for (let i = 0; i < samples; i++) {{
      const amp = peaks[i] / maxPeak;
      const barHeight = Math.max(1, amp * (H * 0.8 / 2));
      const x = (i / samples) * W;
      const isPlayed = (i / samples) <= progress;
      // Waveform bars: blue for played, light grey for unplayed
      ctx.strokeStyle = isPlayed ? "#0ea5e9" : "#d1d5db";
      ctx.beginPath();
      ctx.moveTo(x, midY - barHeight);
      ctx.lineTo(x, midY + barHeight);
      ctx.stroke();
    }}

    // selection overlay
    if (
      selectionStartSec != null &&
      selectionEndSec != null &&
      selectionEndSec > selectionStartSec + 0.01
    ) {{
      const selStartRatio = selectionStartSec / duration;
      const selEndRatio = selectionEndSec / duration;
      const selX1 = Math.max(0, Math.min(W, selStartRatio * W));
      const selX2 = Math.max(0, Math.min(W, selEndRatio * W));
      // Selection overlay: subtle blue tint
      ctx.fillStyle = "rgba(59, 130, 246, 0.12)";
      ctx.fillRect(Math.min(selX1, selX2), 0, Math.abs(selX2 - selX1), H);
    }}

    // markers (favorite sections)
    if (markers && markers.length && duration > 0) {{
      ctx.save();
      // Markers (favorite sections): clean red
      ctx.fillStyle = "#ef4444";
      markers.forEach(m => {{
        let t = 0;
        if (typeof m.time === "number") {{
          t = m.time;
        }} else if (typeof m.time === "string") {{
          const parsed = parseFloat(m.time);
          if (!isNaN(parsed)) t = parsed;
        }}
        if (!isFinite(t) || t < 0 || t > duration) return;
        const x = (t / duration) * W;
        const h = 10;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - 4, h);
        ctx.lineTo(x + 4, h);
        ctx.closePath();
        ctx.fill();
      }});
      ctx.restore();
    }}

    // Playhead line: dark grey
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, H);
    ctx.stroke();
  }}

  function redrawAllWaveforms() {{
    const t = getGlobalCurrentTime();
    updateChordDisplay(t);
    for (let i = 0; i < waveforms.length; i++) {{
      drawSingleWaveform(i, t);
    }}
  }}

  function setupCanvasInteractions(canvas, idx) {{
    let isDragging = false;
    let dragStartX = 0;
    let dragStartTime = 0;
    let dragMoved = false;
    const clickThreshold = 3; // px

    function xToTime(localX) {{
      const wf = waveforms[idx];
      if (!wf) return 0;
      const rectWidth = canvas.getBoundingClientRect().width || canvas.width;
      const ratio = Math.min(Math.max(localX / rectWidth, 0), 1);
      return ratio * (wf.duration || 0);
    }}

    canvas.addEventListener("mousedown", (evt) => {{
      const rect = canvas.getBoundingClientRect();
      const localX = evt.clientX - rect.left;
      isDragging = true;
      dragMoved = false;
      dragStartX = localX;
      dragStartTime = xToTime(localX);
    }});

    canvas.addEventListener("mousemove", (evt) => {{
      if (!isDragging) return;
      const rect = canvas.getBoundingClientRect();
      const localX = evt.clientX - rect.left;
      if (Math.abs(localX - dragStartX) > clickThreshold) {{
        dragMoved = true;
      }}
      const curTime = xToTime(localX);
      selectionStartSec = Math.min(dragStartTime, curTime);
      selectionEndSec = Math.max(dragStartTime, curTime);
      updateSelectionDisplay();
      redrawAllWaveforms();
    }});

    canvas.addEventListener("mouseup", (evt) => {{
      if (!isDragging) return;
      isDragging = false;
      const rect = canvas.getBoundingClientRect();
      const localX = evt.clientX - rect.left;
      const localTime = xToTime(localX);

      if (!dragMoved || Math.abs(localX - dragStartX) <= clickThreshold) {{
        // click → seek
        const targetTime = localTime;
        audios.forEach(a => {{
          try {{
            a.currentTime = Math.min(
              targetTime,
              (a.duration && isFinite(a.duration)) ? (a.duration - 0.01) : targetTime
            );
          }} catch (e) {{}}
        }});
        redrawAllWaveforms();
      }} else {{
        // drag selection already handled
        updateSelectionDisplay();
        redrawAllWaveforms();
      }}
    }});

    canvas.addEventListener("mouseleave", () => {{
      if (isDragging) {{
        isDragging = false;
        updateSelectionDisplay();
        redrawAllWaveforms();
      }}
    }});
  }}

  function buildWaveforms() {{
    waveContainer.innerHTML = "";
    stems.forEach((stem, idx) => {{
      const wrapper = document.createElement("div");
      wrapper.style.marginBottom = "8px";

      const labelDiv = document.createElement("div");
      labelDiv.textContent = stem.name + (stem.is_full_mix ? " (mix)" : "");
      labelDiv.style.fontSize = "11px";
      labelDiv.style.color = "#6b7280";
      labelDiv.style.marginBottom = "2px";
      wrapper.appendChild(labelDiv);

const canvas = document.createElement("canvas");
canvas.width = 1000;
canvas.height = 60;
canvas.style.width = "100%";
canvas.style.height = "60px";
canvas.style.borderRadius = "6px";
canvas.style.background = "#ffffff";
canvas.style.border = "1px solid #e5e7eb";
canvas.style.display = "block";
wrapper.appendChild(canvas);


      waveContainer.appendChild(wrapper);

      const wfObj = {{
        canvas: canvas,
        peaks: [],
        maxPeak: 1.0,
        duration: 0,
        audioBuffer: null,
      }};
      waveforms[idx] = wfObj;
      waveDurations[idx] = 0;

      setupCanvasInteractions(canvas, idx);
    }});

    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();

    stems.forEach((stem, idx) => {{
      const wf = waveforms[idx];
      if (!wf || !wf.canvas) return;

      try {{
        fetch(stem.data_url)
          .then(r => r.arrayBuffer())
          .then(buf => decodeCtx.decodeAudioData(buf))
          .then(audioBuffer => {{
            const data = audioBuffer.getChannelData(0);
            const samples = 800;
            const blockSize = Math.max(1, Math.floor(data.length / samples));
            const peaks = [];
            for (let i = 0; i < samples; i++) {{
              let sum = 0;
              const start = i * blockSize;
              const end = Math.min(start + blockSize, data.length);
              for (let j = start; j < end; j++) {{
                sum += Math.abs(data[j]);
              }}
              peaks.push(sum / (end - start || 1));
            }}
            const maxPeak = Math.max(...peaks) || 1.0;
            const duration = audioBuffer.duration;

            wf.peaks = peaks;
            wf.maxPeak = maxPeak;
            wf.duration = duration;
            wf.audioBuffer = audioBuffer;
            waveDurations[idx] = duration;

            drawSingleWaveform(idx, getGlobalCurrentTime());
          }})
          .catch(err => {{
            console.log("Waveform decode error", err);
          }});
      }} catch (e) {{
        console.log("Waveform setup error", e);
      }}
    }});
  }}

  // Play/pause toggle
  let isPlaying = false;

  function playAll() {{
    isPlaying = true;
    applyVolumes();
    audios.forEach(a => {{
      try {{ a.play(); }} catch (e) {{}}
    }});
    if (toggleBtn) toggleBtn.textContent = "Pause";
  }}

  function pauseAll() {{
    isPlaying = false;
    audios.forEach(a => {{
      try {{ a.pause(); }} catch (e) {{}}
    }});
    if (toggleBtn) toggleBtn.textContent = "▶ Play";
    redrawAllWaveforms();
  }}

  if (toggleBtn) {{
    toggleBtn.addEventListener("click", () => {{
      if (!isPlaying) {{
        playAll();
      }} else {{
        pauseAll();
      }}
    }});
  }}

  audios.forEach(a => {{
    a.addEventListener("timeupdate", () => {{
      const t = getGlobalCurrentTime();
      updateChordDisplay(t);
      redrawAllWaveforms();
    }});
    a.addEventListener("ended", () => {{
      // If everything has basically hit the end, reset button state
      const t = getGlobalCurrentTime();
      const maxDur = Math.max(...waveDurations.filter(d => isFinite(d) && d > 0), 0);
      if (maxDur && t >= maxDur - 0.05) {{
        isPlaying = false;
        if (toggleBtn) toggleBtn.textContent = "▶ Play";
      }}
    }});
  }});

  // WAV encoder (16-bit PCM)
  function encodeWAV(channels, sampleRate) {{
    const numChannels = channels.length;
    const numSamples = channels[0].length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    let offset = 0;

    function writeString(s) {{
      for (let i = 0; i < s.length; i++) {{
        view.setUint8(offset++, s.charCodeAt(i));
      }}
    }}
    function writeUint32(v) {{
      view.setUint32(offset, v, true);
      offset += 4;
    }}
    function writeUint16(v) {{
      view.setUint16(offset, v, true);
      offset += 2;
    }}

    writeString("RIFF");
    writeUint32(36 + dataSize);
    writeString("WAVE");
    writeString("fmt ");
    writeUint32(16);
    writeUint16(1);
    writeUint16(numChannels);
    writeUint32(sampleRate);
    writeUint32(byteRate);
    writeUint16(blockAlign);
    writeUint16(16);
    writeString("data");
    writeUint32(dataSize);

    for (let i = 0; i < numSamples; i++) {{
      for (let ch = 0; ch < numChannels; ch++) {{
        let sample = channels[ch][i];
        sample = Math.max(-1, Math.min(1, sample));
        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        view.setInt16(offset, intSample, true);
        offset += 2;
      }}
    }}
    return new Uint8Array(buffer);
  }}

  if (exportBtn) {{
    exportBtn.addEventListener("click", async () => {{
      if (!window.JSZip) {{
        alert("JSZip not loaded; cannot export ZIP.");
        return;
      }}
      if (
        selectionStartSec == null ||
        selectionEndSec == null ||
        selectionEndSec <= selectionStartSec + 0.01
      ) {{
        alert("Select a region on the waveform first (click + drag).");
        return;
      }}

      const audibleIndices = getAudibleIndices();
      if (!audibleIndices.length) {{
        alert("No audible stems. Use solo/mute to choose which stems to export.");
        return;
      }}

      const zip = new JSZip();
      const startSec = selectionStartSec;
      const endSec = selectionEndSec;

      for (const idx of audibleIndices) {{
        const wf = waveforms[idx];
        if (!wf || !wf.audioBuffer) continue;
        const audioBuffer = wf.audioBuffer;
        const sr = audioBuffer.sampleRate;
        const numChannels = audioBuffer.numberOfChannels;

        const startSample = Math.max(0, Math.floor(startSec * sr));
        const endSample = Math.min(
          audioBuffer.length,
          Math.max(startSample + 1, Math.floor(endSec * sr))
        );
        const numSamples = endSample - startSample;
        if (numSamples <= 0) continue;

        const channels = [];
        for (let ch = 0; ch < numChannels; ch++) {{
          const src = audioBuffer.getChannelData(ch);
          const segment = src.slice(startSample, endSample);
          channels.push(segment);
        }}

        const wavBytes = encodeWAV(channels, sr);
        const stemName = (stems[idx].name || ("stem_" + idx)).replace(/\\s+/g, "_");
        const startLabel = Math.floor(startSec * 1000);
        const endLabel = Math.floor(endSec * 1000);
        const fileName = stemName + "_" + startLabel + "ms_" + endLabel + "ms.wav";
        zip.file(fileName, wavBytes);
      }}

      const blob = await zip.generateAsync({{ type: "blob" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "beatdecoder_selection.zip";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {{
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }}, 1000);
    }});
  }}

  buildWaveforms();
  updateSelectionDisplay();
}})();
</script>
"""
    components.html(html, height=380 + 60 * len(stems_data), scrolling=False)



# ---------------------------------------------------------------------
# MY SONGS VIEW (PAGINATED + GLOBAL PLAYER + ANALYTICS)
# ---------------------------------------------------------------------



def load_all_job_infos() -> List[Dict[str, Any]]:
    """
    Load all job folders and return a list of info dicts.

    This version restores all the fields the My Songs UI expects,
    including job_label, original_file, model_name, status, favorite, etc.
    """
    infos: List[Dict[str, Any]] = []

    for job_dir in list_jobs():
        meta = load_metadata(job_dir)
        song_name = infer_song_name(job_dir, meta)
        created_at = infer_created_at(job_dir, meta)

        info = {
            "job_dir": job_dir,
            "meta": meta,
            "song_name": song_name,
            "created_at": created_at,
            # These fields are required by the My Songs UI:
            "job_id": meta.get("job_id", job_dir.name),
            "job_label": meta.get("job_label") or job_dir.name,
            "bpm": meta.get("bpm"),
            "key": meta.get("key"),
            "mode": meta.get("mode"),
            "notes": meta.get("notes", ""),
            "original_file": meta.get("original_file_name", "unknown"),
            "model_name": meta.get("model_name", "unknown"),
            "tags": meta.get("tags", []),
            "status": meta.get("status", "Idea"),
            "favorite": meta.get("favorite", False),
            "rating": meta.get("rating", None),
        }

        infos.append(info)

    # My Songs view expects created_at to be sortable, so we sort here
    infos_sorted = sorted(infos, key=lambda x: x["created_at"], reverse=True)
    return infos_sorted



def render_global_player_from_session(prefix: str = "songs"):
    job_path = st.session_state.get("current_player_job")
    song_name = st.session_state.get("current_player_song_name")

    if not job_path:
        st.info("Select a song below to load it into the player.")
        return

    job_dir = Path(job_path)
    if not job_dir.exists():
        st.info("Selected job folder no longer exists on disk.")
        return

    meta = load_metadata(job_dir)

    # Collect stems
    stems = [p for p in job_dir.iterdir() if p.suffix.lower() == ".wav"]
    stems = sorted(stems)

    bpm = meta.get("bpm")
    key = meta.get("key")
    mode = meta.get("mode")

    # 🔊 Player UI
    st.markdown("### 🎚 Player")


    top_cols = st.columns([3, 1, 1])
    with top_cols[0]:
        display_name = meta.get("song_name") or song_name or job_dir.name
        path_str = str(job_dir.resolve())
        st.markdown(
            f"""
                <div class="bd-now-playing">
                 <div class="bd-now-meta">
                  <div class="bd-now-title">🎧 {display_name}</div>
                 <div class="bd-now-path">{path_str}</div>
              </div>
                <div class="bd-now-pill">
                 <span style="width:8px;height:8px;border-radius:999px;background:#22c55e;display:inline-block;"></span>
                 <span>Ready to play</span>
                 </div>
              </div>
              """,
        unsafe_allow_html=True,
        )
        st.caption(str(job_dir.resolve()))
    with top_cols[1]:
        st.metric("BPM", f"{bpm:.2f}" if bpm is not None else "–")
    with top_cols[2]:
        key_text = f"{key} {mode}".strip() if key and mode else "–"
        st.metric("Key", key_text)

    if not stems:
        st.write("No stems found for this job.")
        return

    # 🔥 The actual multi-stem waveform player
    render_multi_stem_player(stems, meta, title="Stems Mixer")

    # ------------------------------
    # Stem ZIP export (your custom version)
    # ------------------------------
    with st.expander("📙  Download selected stems as ZIP"):
        st.write("Choose which stems to include and the format:")

        stem_selection = {}
        for stem in stems:
            label = stem.stem
            stem_selection[stem] = st.checkbox(
                label,
                value=True,
                key=f"{prefix}_stemchk_{job_dir.name}_{stem.name}",
            )

        fmt = st.radio(
            "Format",
            options=["wav", "mp3"],
            index=0,
            key=f"{prefix}_fmt_{job_dir.name}",
        )

        if st.button("Download ZIP", key=f"{prefix}_buildzip_{job_dir.name}"):
            chosen_files = []

            for stem, chosen in stem_selection.items():
                if not chosen:
                    continue

                if fmt == "wav":
                    chosen_files.append(stem)
                else:
                    mp3_candidate = stem.with_suffix(".mp3")
                    if mp3_candidate.exists():
                        chosen_files.append(mp3_candidate)

            if not chosen_files:
                st.warning("No stems found that match your selection.")
            else:
                mem_zip = io.BytesIO()
                with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in chosen_files:
                        zf.write(p, arcname=p.name)
                mem_zip.seek(0)

                safe_name = (song_name or job_dir.name).replace(" ", "_")
                st.download_button(
                    "Download selected stems as ZIP",
                    data=mem_zip.getvalue(),
                    file_name=f"{safe_name}_stems_{fmt}.zip",
                    mime="application/zip",
                    key=f"{prefix}_stemszip_{job_dir.name}",
                )

    # ------------------------------
    # MIDI FILES
    # ------------------------------
    midi_dir = job_dir / "midi"
    midi_files = [p for p in midi_dir.iterdir() if p.suffix.lower() == ".mid"] if midi_dir.exists() else []

    st.markdown("#### 🎼 MIDI Files")
    if not midi_files:
        st.write("No MIDI files for this job.")
    else:
        for midi in midi_files:
            with open(midi, "rb") as f:
                data = f.read()
            st.download_button(
                f"Download {midi.name}",
                data=data,
                file_name=midi.name,
                mime="audio/midi",
                key=f"{prefix}_global_midi_{job_dir.name}_{midi.name}",
            )



def render_my_songs_view(prefix: str = "songs"):
    st.header("My Songs")
    st.markdown(
        '<p class="bd-page-subtitle">Pick one of your songs to view it.</p>',
        unsafe_allow_html=True,
    )

    job_infos = load_all_job_infos()
    if not job_infos:
        st.write("No jobs yet. Process some tracks first.")
        return

    # Ensure there is a current job selected for the global player
    if st.session_state.get("current_player_job") is None:
        first = job_infos[0]
        st.session_state["current_player_job"] = str(first["job_dir"])
        st.session_state["current_player_song_name"] = first["song_name"]

    # -------- PLAYER SECTION (card) --------
    with st.container():
        st.markdown(
            """
            <div class="bd-section">
              <div class="bd-section-header">
                <div class="bd-section-title">Player</div>
                <div class="bd-section-caption">Solo stems, drag to select loops, and export sections.</div>
              </div>
            """,
            unsafe_allow_html=True,
        )
        render_global_player_from_session(prefix=prefix)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- FILTERS / SEARCH / SORT (card) --------
    with st.container():
        st.markdown(
            """
            <div class="bd-section">
              <div class="bd-section-header">
                <div class="bd-section-title">Filters</div>
                <div class="bd-section-caption">Search and filter your library.</div>
              </div>
            """,
            unsafe_allow_html=True,
        )

        col_filters1, col_filters2 = st.columns([2, 1])
        with col_filters1:
            text_filter = st.text_input(
                "Search by song, job label, notes, or tags",
                "",
                key=f"{prefix}_search_text",
            )
        with col_filters2:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Newest first",
                    "Oldest first",
                    "BPM (low → high)",
                    "BPM (high → low)",
                    "Key (A → Z)",
                ],
                key=f"{prefix}_sort_by",
            )

        col_filters3, col_filters4 = st.columns(2)
        with col_filters3:
            bpm_min = st.number_input(
                "Min BPM",
                value=0,
                min_value=0,
                max_value=400,
                key=f"{prefix}_bpm_min",
            )
            bpm_max = st.number_input(
                "Max BPM",
                value=400,
                min_value=0,
                max_value=400,
                key=f"{prefix}_bpm_max",
            )
        with col_filters4:
            key_options = ["Any", "C", "C#", "D", "D#", "E", "F",
                           "F#", "G", "G#", "A", "A#", "B"]
            key_choice = st.selectbox(
                "Key filter",
                options=key_options,
                index=0,
                key=f"{prefix}_key_filter",
            )

        col_filters5, col_filters6 = st.columns(2)
        with col_filters5:
            status_options = ["Any", "Idea", "In Progress", "Ready", "Final"]
            status_filter = st.selectbox(
                "Status filter",
                options=status_options,
                index=0,
                key=f"{prefix}_status_filter",
            )
        with col_filters6:
            favorites_only = st.checkbox(
                "Show favorites only",
                value=False,
                key=f"{prefix}_favorites_only",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- GROUP + FILTER SONGS --------
    from collections import defaultdict
    from typing import Dict, List, Any
    import math
    from collections import Counter

    song_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for info in job_infos:
        song_groups[info["song_name"]].append(info)

    grouped_songs: List[Dict[str, Any]] = []
    for song_name, group in song_groups.items():
        group_sorted = sorted(group, key=lambda x: x["created_at"], reverse=True)
        primary = group_sorted[0]

        # Search filter
        if text_filter:
            haystack = song_name.lower()
            for g in group:
                haystack += " " + (g["job_label"] or "").lower()
                haystack += " " + (g["notes"] or "").lower()
                haystack += " " + " ".join([t.lower() for t in g.get("tags", [])])
            if text_filter.lower() not in haystack:
                continue

        # BPM filter
        bpm = primary.get("bpm")
        if bpm is not None and (bpm < bpm_min or bpm > bpm_max):
            continue

        # Key filter
        key_val = primary.get("key")
        if key_choice != "Any" and key_val != key_choice:
            continue

        # Status filter
        status_val = primary.get("status", "Idea")
        if status_filter != "Any" and status_val != status_filter:
            continue

        # Favorites filter
        fav_val = primary.get("favorite", False)
        if favorites_only and not fav_val:
            continue

        grouped_songs.append(
            {
                "song_name": song_name,
                "group": group_sorted,
                "primary": primary,
            }
        )

    def safe_bpm(x):
        b = x["primary"].get("bpm")
        return b if isinstance(b, (int, float)) else -1

    def safe_key(x):
        k = x["primary"].get("key")
        return k if k is not None else "ZZZ"

    if sort_by == "Newest first":
        grouped_songs.sort(key=lambda x: x["primary"]["created_at"], reverse=True)
    elif sort_by == "Oldest first":
        grouped_songs.sort(key=lambda x: x["primary"]["created_at"])
    elif sort_by == "BPM (low → high)":
        grouped_songs.sort(key=lambda x: (safe_bpm(x) if safe_bpm(x) >= 0 else 9999))
    elif sort_by == "BPM (high → low)":
        grouped_songs.sort(
            key=lambda x: (safe_bpm(x) if safe_bpm(x) >= 0 else -1),
            reverse=True,
        )
    elif sort_by == "Key (A → Z)":
        grouped_songs.sort(key=lambda x: safe_key(x))

    total_songs = len(grouped_songs)
    if total_songs == 0:
        st.write("No songs match the current filters.")
        return

    # -------- PAGINATION --------
    page_size = 12   # 4 columns * 3 rows per page
    total_pages = max(1, math.ceil(total_songs / page_size))
    col_page1, col_page2, _ = st.columns([1, 1, 2])
    with col_page1:
        current_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{prefix}_page",
        )
    with col_page2:
        st.write(f"of {total_pages} page(s), {total_songs} song(s) total.")

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    page_songs = grouped_songs[start_idx:end_idx]

    # -------- SONG GRID: 4 COLUMNS, CLICKABLE TILES --------
    for row_start in range(0, len(page_songs), 4):
        row_songs = page_songs[row_start:row_start + 4]
        cols = st.columns(4)

        for i, (col, song_entry) in enumerate(zip(cols, row_songs)):
            global_idx = start_idx + row_start + i
            song_name = song_entry["song_name"]
            primary = song_entry["primary"]

            # Nicely truncated title for display
            display_title = song_name
            if len(display_title) > 32:
                display_title = display_title[:29] + "..."

            bpm_val = primary.get("bpm")
            bpm_text = f"{bpm_val:.2f} BPM" if isinstance(bpm_val, (int, float)) else "BPM —"

            key_val = primary.get("key")
            mode_val = primary.get("mode") or ""
            key_text = f"{key_val} {mode_val}".strip() if key_val else "Key —"

            job_dir = primary["job_dir"]

            # Two-line label inside the square
            label = f"{display_title}\n{bpm_text}  •  {key_text}"


            with col:
                st.markdown('<div class="song-card">', unsafe_allow_html=True)
                clicked = st.button(
                    label,
                    key=f"{prefix}_song_card_{global_idx}",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if clicked:
                    st.session_state["current_player_job"] = str(job_dir)
                    st.session_state["current_player_song_name"] = song_name
                    st.rerun()

    # (Optional) library analytics can stay below if you still want it.


    # -------- LIBRARY ANALYTICS (card) --------
    with st.container():
        st.markdown(
            """
            <div class="bd-section">
              <div class="bd-section-header">
                <div class="bd-section-title">Library Analytics</div>
                <div class="bd-section-caption">High-level view of your jobs, keys, BPMs, and favorites.</div>
              </div>
            """,
            unsafe_allow_html=True,
        )

        all_keys = [info["key"] for info in job_infos if info.get("key")]
        all_bpms = [info["bpm"] for info in job_infos if isinstance(info.get("bpm"), (int, float))]
        all_status = [info.get("status", "Idea") for info in job_infos]
        all_tags = [t for info in job_infos for t in info.get("tags", [])]
        all_favs = [info.get("favorite", False) for info in job_infos]

        total_jobs = len(job_infos)
        unique_songs = len(song_groups)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total jobs", total_jobs)
        with col_b:
            st.metric("Unique songs", unique_songs)
        with col_c:
            st.metric("Favorites", sum(1 for f in all_favs if f))

        if all_bpms:
            st.write(f"- Avg BPM: {sum(all_bpms)/len(all_bpms):.2f}")
        if all_keys:
            key_counts = Counter(all_keys)
            st.write(f"- Most common key: {key_counts.most_common(1)[0][0]}")

        if all_status:
            st.write("- Status distribution:")
            for status, count in Counter(all_status).items():
                st.write(f"  - {status}: {count}")

        if all_tags:
            tag_counts = Counter(all_tags)
            st.write("- Top tags:")
            for tag, count in tag_counts.most_common(10):
                st.write(f"  - {tag}: {count}")

        st.markdown("</div>", unsafe_allow_html=True)







# ---------------------------------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------------------------------

st.set_page_config(page_title="AI Stem Splitter + MIDI Rebuilder", layout="wide")
init_session_state()

st.title("🎧 AI Stem Splitter + MIDI Rebuilder")
st.caption(
    "Demucs stems • BPM/Key • live chord display • My Songs library with producer tools."
)

tab_process, tab_songs, tab_settings = st.tabs(
    ["Process Track(s)", "My Songs", "Settings"]
)

# ---------------------------------------------------------------------
# SETTINGS TAB
# ---------------------------------------------------------------------

with tab_settings:
    st.header("Settings")

    backend = st.radio(
        "Processing backend:",
        ["Remote GPU worker", "Local (this server)"],
        index=["Remote GPU worker", "Local (this server)"].index(
            st.session_state["processing_backend"]
        ),
        help="Remote GPU = FastAPI worker (RunPod/local). Local = this Streamlit process.",
    )
    st.session_state["processing_backend"] = backend

    if backend == "Remote GPU worker":
        worker_url = st.text_input(
            "GPU worker URL",
            value=st.session_state["worker_url"],
            help="For local dev: http://localhost:8000",
        )
        st.session_state["worker_url"] = worker_url.strip()

    st.subheader("Demucs Model")
    model_input = st.text_input("Demucs model name:", st.session_state["demucs_model"])
    st.session_state["demucs_model"] = model_input.strip() or DEFAULT_DEMUCS_MODEL

    st.subheader("Demucs Device (local only)")
    device_choice = st.selectbox(
        "Device:",
        options=["auto", "cpu", "cuda", "mps"],
        index=["auto", "cpu", "cuda", "mps"].index(st.session_state["demucs_device"]),
    )
    st.session_state["demucs_device"] = device_choice

    st.subheader("MIDI Extraction Targets (local only)")
    options = ["bass", "other", "vocals", "drums"]
    chosen = st.multiselect(
        "Which stems should Basic Pitch extract MIDI from?",
        options,
        default=st.session_state["stems_for_midi"],
    )
    st.session_state["stems_for_midi"] = chosen or DEFAULT_STEMS_FOR_MIDI.copy()

    st.subheader("Sound Design / Serum-style Analysis")
    st.session_state["run_serum_analysis"] = st.checkbox(
        "Enable Serum-style analysis on stems",
        value=st.session_state["run_serum_analysis"],
    )

    st.info("Settings will apply to the next processing run.")


# ---------------------------------------------------------------------
# MY SONGS TAB
# ---------------------------------------------------------------------

with tab_songs:
    render_my_songs_view(prefix="songs")


# ---------------------------------------------------------------------
# PROCESS TAB
# ---------------------------------------------------------------------

with tab_process:
    st.header("Process Track(s)")

    st.write(
        "Upload one or multiple WAV/MP3 files. "
        "In **Remote GPU worker** mode, processing happens on your worker API."
    )

    uploaded_files = st.file_uploader(
        "Upload WAV or MP3",
        type=["wav", "mp3"],
        accept_multiple_files=True,
    )

    job_label_input = st.text_input(
        "Optional job name/description:",
        "",
        key="process_job_label",
    )

    backend = st.session_state["processing_backend"]

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Logs")
        st.text_area(
            "Processing log",
            value="\n".join(st.session_state["logs"]),
            height=400,
            key="processing_log_textarea",
        )

    run_clicked = st.button("Process Song(s)", key="process_button")

    if uploaded_files and run_clicked:
        st.session_state["logs"] = []
        demucs_model = st.session_state["demucs_model"]
        stems_for_midi = st.session_state["stems_for_midi"]
        run_serum = st.session_state["run_serum_analysis"]
        demucs_device = st.session_state["demucs_device"]

        # ---------------- GPU BACKEND ----------------
        if backend == "Remote GPU worker":
            with col_left:
                worker_url = st.session_state["worker_url"].rstrip("/")
                if not worker_url:
                    st.error("GPU worker URL is empty. Set it in Settings.")
                else:
                    if len(uploaded_files) != 1:
                        st.warning("GPU worker mode currently supports one file at a time.")
                    uploaded_file = uploaded_files[0]

                    try:
                        st.info("Starting GPU job...")
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                        }
                        data = {
                            "model_name": demucs_model,
                            "demucs_device": "cuda",
                            "run_serum_analysis": "true" if run_serum else "false",
                            "job_label": job_label_input or "",
                            "stems_for_midi": ",".join(stems_for_midi),
                        }
                        resp = requests.post(
                            f"{worker_url}/start_job",
                            files=files,
                            data=data,
                            timeout=60,
                        )
                        resp.raise_for_status()
                        start_data = resp.json()
                        job_id = start_data["job_id"]

                        st.success(f"GPU job started. Job ID: `{job_id}`")

                        status_placeholder = st.empty()
                        poll_log_placeholder = st.empty()

                        def show_logs_from_status(job_status: Dict[str, Any]):
                            log_lines = job_status.get("log", []) or []
                            st.session_state["logs"] = log_lines
                            poll_log_placeholder.code(
                                "\n".join(log_lines) if log_lines else "(no logs yet)",
                                height=300,
                            )

                        with st.spinner("Processing on GPU worker..."):
                            status = "queued"
                            result_data = None
                            max_polls = 200
                            for _ in range(max_polls):
                                time.sleep(3)
                                s_resp = requests.get(
                                    f"{worker_url}/job_status",
                                    params={"job_id": job_id},
                                    timeout=30,
                                )
                                s_resp.raise_for_status()
                                status_json = s_resp.json()
                                status = status_json["status"]
                                status_placeholder.write(f"Status: **{status}**")
                                show_logs_from_status(status_json)

                                if status == "done":
                                    result_data = status_json.get("result", {})
                                    break
                                if status == "error":
                                    err_msg = status_json.get("error", "Unknown error")
                                    st.error(f"GPU job failed: {err_msg}")
                                    st.stop()

                            if status != "done":
                                st.error("GPU job did not finish in time.")
                                st.stop()

                        st.info("Fetching ZIP from GPU worker...")
                        z_resp = requests.get(
                            f"{worker_url}/job_zip",
                            params={"job_id": job_id},
                            timeout=120,
                        )
                        z_resp.raise_for_status()
                        zip_bytes = z_resp.content

                        jobs_root = WORKSPACE_DIR / "jobs"
                        jobs_root.mkdir(parents=True, exist_ok=True)
                        job_root = jobs_root / job_id

                        if job_root.exists() and job_root.is_dir():
                            shutil.rmtree(job_root)
                        job_root.mkdir(parents=True, exist_ok=True)

                        zip_path = job_root / f"{job_id}.zip"
                        with open(zip_path, "wb") as fz:
                            fz.write(zip_bytes)

                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(job_root)

                        meta_path = job_root / "metadata.json"
                        if not meta_path.exists():
                            # fallback metadata if worker didn't write one
                            song_name = result_data.get("song_name") or uploaded_file.name
                            bpm = result_data.get("bpm")
                            key = result_data.get("key")
                            mode = result_data.get("mode")
                            key_conf = result_data.get("key_confidence")

                            auto_label = generate_smart_job_label(
                                song_name=song_name,
                                bpm=bpm,
                                key=key,
                                mode=mode,
                                model_name=demucs_model,
                                fallback=job_id,
                            )

                            meta_data = {
                                "song_name": song_name,
                                "bpm": bpm,
                                "key": key,
                                "mode": mode,
                                "key_confidence": key_conf,
                                "job_label": job_label_input or auto_label,
                                "job_id": job_id,
                                "created_at": datetime.utcnow().isoformat(),
                                "model_name": demucs_model,
                                "original_file_name": uploaded_file.name,
                                "tags": [],
                                "status": "Idea",
                                "favorite": False,
                                "rating": None,
                                "markers": [],
                            }
                            try:
                                with open(meta_path, "w") as fm:
                                    json.dump(meta_data, fm, indent=2)
                            except Exception:
                                pass

                        chord_meta = compute_and_store_chords(job_root)
                        if result_data is None:
                            result_data = {}
                        result_data.update(chord_meta)

                        build_index_from_jobs()

                        bpm = result_data.get("bpm")
                        key = result_data.get("key")
                        mode = result_data.get("mode")
                        key_conf = result_data.get("key_confidence")
                        song_name = result_data.get("song_name") or uploaded_file.name

                        st.session_state["last_gpu_result"] = {
                            "song_name": song_name,
                            "bpm": bpm,
                            "key": key,
                            "mode": mode,
                            "key_confidence": key_conf,
                            "zip_bytes": zip_bytes,
                            "job_id": job_id,
                        }

                        # also update global player to this new job
                        st.session_state["current_player_job"] = str(job_root)
                        st.session_state["current_player_song_name"] = song_name

                        st.success("Processing complete on GPU!")
                        st.subheader("Output Overview (GPU)")
                        st.write(f"**Job folder:** `{job_root.resolve()}`")

                        st.markdown("### 🎼 Track Analysis")
                        st.write(f"**Estimated BPM:** {bpm if bpm is not None else 'Unknown'}")
                        if key and mode:
                            conf_str = (
                                f" (confidence {key_conf:.2f})"
                                if isinstance(key_conf, (int, float))
                                else ""
                            )
                            st.write(f"**Estimated Key:** {key} {mode}{conf_str}")
                        else:
                            st.write("**Estimated Key:** Unknown")

                    except Exception as e:
                        st.error(f"Error during GPU processing: {e}")

        # ---------------- LOCAL BACKEND ----------------
        else:
            with col_left:
                if len(uploaded_files) == 1:
                    uploaded_file = uploaded_files[0]
                    with st.spinner("Processing locally..."):
                        try:
                            saved = save_uploaded_file(uploaded_file)
                            result = process_song(
                                saved,
                                WORKSPACE_DIR,
                                model_name=demucs_model,
                                stems_for_midi=stems_for_midi,
                                run_serum_analysis=run_serum,
                                log_fn=log_to_session,
                                job_label=job_label_input or None,
                                demucs_device=demucs_device,
                            )

                            output_root = result["output_root"]
                            stems_dir = result["stems_dir"]
                            midi_dir = result["midi_dir"]
                            serum_analysis = result.get("serum_analysis", [])
                            bpm = result.get("bpm")
                            key = result.get("key")
                            mode = result.get("mode")
                            key_conf = result.get("key_confidence")

                            chord_meta = compute_and_store_chords(output_root, saved)
                            result.update(chord_meta)

                            build_index_from_jobs()

                            st.success("Processing complete (local)!")
                            st.subheader("Output Overview")
                            st.write(f"**Output folder:** `{output_root.resolve()}`")

                            st.markdown("### 🎼 Track Analysis")
                            st.write(f"**Estimated BPM:** {bpm if bpm is not None else 'Unknown'}")
                            if key and mode:
                                conf_str = (
                                    f" (confidence {key_conf:.2f})"
                                    if isinstance(key_conf, (int, float))
                                    else ""
                                )
                                st.write(f"**Estimated Key:** {key} {mode}{conf_str}")
                            else:
                                st.write("**Estimated Key:** Unknown")

                            st.markdown("### 📦 Download Entire Job")
                            zip_path = make_zip_for_job(output_root)
                            with open(zip_path, "rb") as f_zip:
                                st.download_button(
                                    "Download all (ZIP)",
                                    data=f_zip,
                                    file_name=zip_path.name,
                                    mime="application/zip",
                                    key="zip-latest-local",
                                )

                            if midi_dir.exists():
                                midi_files = [
                                    p for p in midi_dir.iterdir() if p.suffix.lower() == ".mid"
                                ]
                            else:
                                midi_files = []

                            if midi_files:
                                st.markdown("### 🎼 MIDI Files")
                                for midi in midi_files:
                                    rel = midi.relative_to(output_root)
                                    with open(midi, "rb") as f:
                                        data = f.read()
                                    st.download_button(
                                        label=f"Download {rel}",
                                        data=data,
                                        file_name=str(rel),
                                        mime="audio/midi",
                                        key=f"midi-{rel}",
                                    )

                            if run_serum and serum_analysis:
                                st.markdown("### 🔬 Sound Design / Serum Hints")
                                for analysis in serum_analysis:
                                    st.markdown(f"**Stem:** `{analysis['stem']}`")
                                    st.code(analysis["description"])
                                    if "serum_patch" in analysis:
                                        st.markdown("**Serum patch suggestion:**")
                                        st.code(json.dumps(analysis["serum_patch"], indent=2))

                            guide_path = output_root / "ABLETON_IMPORT.txt"
                            if guide_path.exists():
                                st.markdown("### 🎛 Ableton Import Guide")
                                with open(guide_path, "r") as f_guide:
                                    st.text(f_guide.read())

                            # update global player to this new job
                            st.session_state["current_player_job"] = str(output_root)
                            st.session_state["current_player_song_name"] = result.get("song_name", uploaded_file.name)

                        except Exception as e:
                            st.error(f"Error during processing (local): {e}")
                else:
                    with st.spinner("Processing batch locally..."):
                        job_summaries = []
                        for up in uploaded_files:
                            try:
                                saved = save_uploaded_file(up)
                                per_file_label = job_label_input or up.name
                                result = process_song(
                                    saved,
                                    WORKSPACE_DIR,
                                    model_name=demucs_model,
                                    stems_for_midi=stems_for_midi,
                                    run_serum_analysis=run_serum,
                                    log_fn=log_to_session,
                                    job_label=per_file_label,
                                    demucs_device=demucs_device,
                                )
                                output_root = result["output_root"]

                                chord_meta = compute_and_store_chords(output_root, saved)
                                result.update(chord_meta)

                                job_summaries.append(
                                    {
                                        "file": up.name,
                                        "job_folder": result["output_root"],
                                        "song_name": result.get("song_name", up.name),
                                        "job_label": per_file_label,
                                        "bpm": result.get("bpm"),
                                        "key": result.get("key"),
                                        "mode": result.get("mode"),
                                    }
                                )
                            except Exception as e:
                                st.error(f"Error processing {up.name}: {e}")

                        build_index_from_jobs()

                        if job_summaries:
                            st.success(f"Batch complete (local). Processed {len(job_summaries)} file(s).")
                            st.markdown("### Summary")
                            for js in job_summaries:
                                key_str = (
                                    f"{js['key']} {js['mode']}"
                                    if js.get("key") and js.get("mode")
                                    else "Unknown"
                                )
                                st.write(
                                    f"- **File:** {js['file']} → **Song:** {js['song_name']} "
                                    f"→ **BPM:** {js.get('bpm', 'Unknown')} → **Key:** {key_str} "
                                    f"→ **Job folder:** `{js['job_folder']}`"
                                )
                            st.info("Use the My Songs tab to inspect jobs and use the global player.")

    gpu_res = st.session_state.get("last_gpu_result")
    if gpu_res:
        st.markdown("---")
        st.subheader("Last GPU Result")
        st.write(f"**Song:** {gpu_res.get('song_name', 'Unknown')}")
        bpm = gpu_res.get("bpm")
        st.write(f"**Estimated BPM:** {bpm if bpm is not None else 'Unknown'}")
        key = gpu_res.get("key")
        mode = gpu_res.get("mode")
        key_conf = gpu_res.get("key_confidence")
        if key and mode:
            conf_str = (
                f" (confidence {key_conf:.2f})"
                if isinstance(key_conf, (int, float))
                else ""
            )
            st.write(f"**Estimated Key:** {key} {mode}{conf_str}")
        else:
            st.write("**Estimated Key:** Unknown")

        st.download_button(
            "Download last GPU job (ZIP)",
            data=gpu_res["zip_bytes"],
            file_name=f"{gpu_res.get('song_name','song')}_job.zip",
            mime="application/zip",
            key="zip-gpu-latest",
        )
