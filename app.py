import os
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests
import streamlit as st

from processor import update_index

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

# GPU worker URL (RunPod)
WORKER_URL = os.environ.get(
    "BEATDECODER_WORKER_URL",
    "https://iblbtqx9zvpxqw-8000.proxy.runpod.net",
)

DEFAULT_DEMUCS_MODEL = "htdemucs"
DEFAULT_STEMS_FOR_MIDI = ["bass", "other"]
DEFAULT_DEMUCS_DEVICE = "cuda"  # default to GPU on worker


# -----------------------------------------------------------------------------
# Helpers: file saving, jobs, metadata, index
# -----------------------------------------------------------------------------

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
                return json.load(f)
        except Exception:
            return {}
    return {}


def infer_song_name(job_dir: Path, meta: Dict[str, Any]) -> str:
    if "song_name" in meta and meta["song_name"]:
        return meta["song_name"]
    name = job_dir.name
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def infer_created_at(job_dir: Path, meta: Dict[str, Any]) -> str:
    if "created_at" in meta and meta["created_at"]:
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


def log_to_session(msg: str):
    st.session_state["logs"].append(msg)


def make_zip_for_job(job_dir: Path) -> Path:
    zip_path = job_dir / f"{job_dir.name}.zip"
    if not zip_path.exists():
        base_name = job_dir / job_dir.name
        shutil.make_archive(str(base_name), "zip", root_dir=job_dir)
    return zip_path


# -----------------------------------------------------------------------------
# Streamlit layout
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Decode Audio – Stem Splitter + MIDI Rebuilder", layout="wide")
init_session_state()

st.title("🎧 Decode Audio – AI Stem Splitter + MIDI Rebuilder")
st.caption(
    "Demucs + Basic Pitch + BPM/Key + chords/sections + Serum-style analysis, "
    "now powered by a remote GPU worker."
)

tab_process, tab_history, tab_library, tab_settings = st.tabs(
    ["Process Track(s)", "History", "Library", "Settings"]
)

# -----------------------------------------------------------------------------
# Settings tab
# -----------------------------------------------------------------------------

with tab_settings:
    st.header("Settings")

    st.subheader("Demucs Model")
    model_input = st.text_input("Demucs model name:", st.session_state["demucs_model"])
    st.session_state["demucs_model"] = model_input.strip() or DEFAULT_DEMUCS_MODEL

    st.subheader("Demucs Device (on worker)")
    device_choice = st.selectbox(
        "Device:",
        options=["cuda", "cpu"],
        index=["cuda", "cpu"].index(st.session_state["demucs_device"]),
        help="For the GPU worker, 'cuda' uses GPU, 'cpu' forces CPU on the worker.",
    )
    st.session_state["demucs_device"] = device_choice

    st.subheader("MIDI Extraction Targets")
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

    st.subheader("Processing Mode")
    st.radio(
        "Mode:",
        ["Remote GPU worker (current)", "Local (deprecated)"],
        index=0,
        help="This UI currently uses the remote GPU worker API for heavy processing.",
    )

    st.info("Settings will be used for the next processing run.")


# -----------------------------------------------------------------------------
# History tab
# -----------------------------------------------------------------------------

with tab_history:
    st.header("History")

    jobs = list_jobs()
    if not jobs:
        st.write("No jobs yet. Process some tracks first in the Process tab.")
    else:
        job_infos = []
        for job in jobs:
            meta = load_metadata(job)
            song_name = infer_song_name(job, meta)
            created_at = infer_created_at(job, meta)
            label = meta.get("job_label") or job.name
            job_infos.append(
                {
                    "dir": job,
                    "meta": meta,
                    "song_name": song_name,
                    "created_at": created_at,
                    "label": label,
                }
            )

        song_names = sorted({info["song_name"] for info in job_infos})
        song_choice = st.selectbox("Select song:", options=song_names)

        runs_for_song = [info for info in job_infos if info["song_name"] == song_choice]
        run_labels = [
            f"{info['label']} (created {info['created_at']})"
            for info in runs_for_song
        ]
        run_idx = st.selectbox(
            "Select run:",
            options=list(range(len(runs_for_song))),
            format_func=lambda i: run_labels[i],
        )

        chosen_job = runs_for_song[run_idx]
        job_dir = chosen_job["dir"]
        meta = chosen_job["meta"]

        st.write(f"**Job folder:** `{job_dir.resolve()}`")
        if meta.get("job_label"):
            st.write(f"**Job name:** {meta['job_label']}")
        st.write(f"**Original file:** {meta.get('original_file_name', 'unknown')}")
        st.write(f"**Demucs model:** {meta.get('model_name', 'unknown')}")

        bpm = meta.get("bpm")
        key = meta.get("key")
        mode = meta.get("mode")
        key_conf = meta.get("key_confidence")

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

        stems = [p for p in job_dir.iterdir() if p.suffix.lower() == ".wav"]
        midi_dir = job_dir / "midi"
        midi_files = (
            [p for p in midi_dir.iterdir() if p.suffix.lower() == ".mid"]
            if midi_dir.exists()
            else []
        )

        st.markdown("### 📦 Download Entire Job")
        zip_path = make_zip_for_job(job_dir)
        with open(zip_path, "rb") as f_zip:
            st.download_button(
                "Download all (ZIP)",
                data=f_zip,
                file_name=zip_path.name,
                mime="application/zip",
                key=f"zip-{job_dir.name}",
            )

        st.markdown("### 🎚 Stems")
        if not stems:
            st.write("No stems found.")
        else:
            for stem in stems:
                rel = stem.relative_to(job_dir)
                with open(stem, "rb") as f:
                    data = f.read()
                    st.audio(data, format="audio/wav")
                    st.download_button(
                        label=f"Download {rel}",
                        data=data,
                        file_name=str(rel),
                        mime="audio/wav",
                        key=f"hist-stem-{job_dir.name}-{rel}",
                    )

        st.markdown("### 🎼 MIDI Files")
        if not midi_files:
            st.write("No MIDI files.")
        else:
            for midi in midi_files:
                rel = midi.relative_to(job_dir)
                with open(midi, "rb") as f:
                    data = f.read()
                    st.download_button(
                        label=f"Download {rel}",
                        data=data,
                        file_name=str(rel),
                        mime="audio/midi",
                        key=f"hist-midi-{job_dir.name}-{rel}",
                    )

        serum_json_path = job_dir / "serum_patches.json"
        if serum_json_path.exists():
            st.markdown("### 🎛 Serum Patch Suggestions")
            try:
                with open(serum_json_path, "r") as f_serum:
                    serum_data = json.load(f_serum)
                for entry in serum_data:
                    st.markdown(f"**Stem:** `{entry.get('stem', '')}`")
                    st.code(json.dumps(entry.get("serum_patch", {}), indent=2))
            except Exception as e:
                st.write(f"Could not read serum_patches.json: {e}")

        guide_path = job_dir / "ABLETON_IMPORT.txt"
        if guide_path.exists():
            st.markdown("### 🎛 Ableton Import Guide")
            with open(guide_path, "r") as f_guide:
                st.text(f_guide.read())

        if st.button("Delete This Job"):
            shutil.rmtree(job_dir)
            st.success("Job deleted. Refresh to update.")


# -----------------------------------------------------------------------------
# Library tab
# -----------------------------------------------------------------------------

with tab_library:
    st.header("Library / Search")

    index = load_index()
    if not index:
        st.write("No indexed jobs yet. Process tracks first.")
    else:
        col_filters, col_results = st.columns([1, 2])

        with col_filters:
            text_filter = st.text_input(
                "Search by song name or job label (contains):",
                "",
            )
            bpm_min = st.number_input("Min BPM", value=0, min_value=0, max_value=400)
            bpm_max = st.number_input("Max BPM", value=400, min_value=0, max_value=400)
            key_options = ["Any", "C", "C#", "D", "D#", "E", "F",
                           "F#", "G", "G#", "A", "A#", "B"]
            key_choice = st.selectbox("Key:", options=key_options, index=0)

        filtered = []
        for entry in index:
            if text_filter:
                hay = f"{entry.get('song_name','')} {entry.get('job_label','')}".lower()
                if text_filter.lower() not in hay:
                    continue
            bpm = entry.get("bpm")
            if isinstance(bpm, (int, float)):
                if bpm < bpm_min or bpm > bpm_max:
                    continue
            key_val = entry.get("key")
            if key_choice != "Any" and key_val != key_choice:
                continue
            filtered.append(entry)

        with col_results:
            st.write(f"Found {len(filtered)} matching job(s).")
            for e in sorted(filtered, key=lambda x: x.get("created_at", ""), reverse=True):
                st.markdown(
                    f"- **Song:** {e.get('song_name','Unknown')} | "
                    f"**Job:** {e.get('job_label') or e.get('job_id')} | "
                    f"**BPM:** {e.get('bpm','?')} | "
                    f"**Key:** {e.get('key','?')} {e.get('mode','') or ''} | "
                    f"`{e.get('job_path','')}`"
                )


# -----------------------------------------------------------------------------
# Process tab – now calling GPU worker
# -----------------------------------------------------------------------------

with tab_process:
    st.header("Process Track(s)")

    st.write(
        "Upload one or multiple WAV/MP3 files. Processing is done on a remote GPU worker, "
        "then results are synced back here."
    )

    uploaded_files = st.file_uploader(
        "Upload WAV or MP3",
        type=["wav", "mp3"],
        accept_multiple_files=True,
    )

    job_label_input = st.text_input(
        "Optional job name/description (single-file run, or used per-file in batch):",
        "",
    )

    if uploaded_files:
        st.write(f"**Selected:** {len(uploaded_files)} file(s).")

        if st.button("Process Song(s)"):
            st.session_state["logs"] = []

            demucs_model = st.session_state["demucs_model"]
            stems_for_midi = st.session_state["stems_for_midi"]
            run_serum = st.session_state["run_serum_analysis"]
            demucs_device = st.session_state["demucs_device"]

            col_left, col_right = st.columns([2, 1])

            with col_left:
                if len(uploaded_files) == 1:
                    uploaded_file = uploaded_files[0]
                    with st.spinner("Processing on GPU worker..."):
                        try:
                            # Save locally (for record) and stream to worker
                            saved = save_uploaded_file(uploaded_file)
                            log_to_session("Uploading file to GPU worker...")

                            with open(saved, "rb") as f_in:
                                files = {
                                    "file": (uploaded_file.name, f_in, uploaded_file.type or "audio/wav")
                                }
                                data = {
                                    "model_name": demucs_model,
                                    "stems_for_midi": ",".join(stems_for_midi),
                                    "run_serum_analysis": str(run_serum).lower(),
                                    "demucs_device": demucs_device or "cuda",
                                }
                                resp = requests.post(
                                    f"{WORKER_URL}/process",
                                    files=files,
                                    data=data,
                                    timeout=1800,
                                )
                            resp.raise_for_status()
                            result = resp.json()
                            job_dir_name = result["job_dir_name"]
                            log_to_session(f"Worker job complete: {job_dir_name}")

                            # Download ZIP of job from worker
                            log_to_session("Downloading job ZIP from worker...")
                            zip_resp = requests.get(
                                f"{WORKER_URL}/job_zip",
                                params={"job_dir": job_dir_name},
                                timeout=1800,
                            )
                            zip_resp.raise_for_status()

                            # Extract into local workspace
                            jobs_root = WORKSPACE_DIR / "jobs"
                            jobs_root.mkdir(parents=True, exist_ok=True)
                            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                                zf.extractall(jobs_root)

                            job_dir = jobs_root / job_dir_name
                            if not job_dir.exists():
                                raise RuntimeError("Job folder not found after extracting ZIP.")

                            # Load metadata & update local index
                            meta = load_metadata(job_dir)
                            if not meta:
                                meta = {
                                    "job_id": result.get("job_id"),
                                    "job_label": job_label_input or None,
                                    "song_name": result.get("song_name"),
                                    "original_file_name": uploaded_file.name,
                                    "created_at": datetime.now().isoformat(),
                                    "model_name": demucs_model,
                                    "stems_for_midi": stems_for_midi,
                                    "bpm": result.get("bpm"),
                                    "key": result.get("key"),
                                    "mode": result.get("mode"),
                                    "key_confidence": result.get("key_confidence"),
                                }
                                with open(job_dir / "metadata.json", "w") as f_meta:
                                    json.dump(meta, f_meta, indent=2)

                            update_index(WORKSPACE_DIR, meta, job_dir)

                            output_root = job_dir
                            stems_dir = job_dir
                            bpm = meta.get("bpm")
                            key = meta.get("key")
                            mode = meta.get("mode")
                            key_conf = meta.get("key_confidence")

                            st.success("Processing complete!")
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
                                    key="zip-latest",
                                )

                            st.markdown("### 🎚 Stems")
                            stem_files = [
                                p for p in stems_dir.iterdir() if p.suffix.lower() == ".wav"
                            ]
                            if not stem_files:
                                st.write("No stems.")
                            else:
                                for stem in stem_files:
                                    rel = stem.relative_to(output_root)
                                    with open(stem, "rb") as f:
                                        data = f.read()
                                        st.audio(data, format="audio/wav")
                                        st.download_button(
                                            label=f"Download {rel}",
                                            data=data,
                                            file_name=str(rel),
                                            mime="audio/wav",
                                            key=f"stem-{rel}",
                                        )

                            st.markdown("### 🎼 MIDI Files")
                            midi_dir = output_root / "midi"
                            if midi_dir.exists():
                                midi_files = [
                                    p for p in midi_dir.iterdir() if p.suffix.lower() == ".mid"
                                ]
                            else:
                                midi_files = []

                            if not midi_files:
                                st.write("No MIDI files generated.")
                            else:
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

                            serum_json_path = output_root / "serum_patches.json"
                            if serum_json_path.exists():
                                st.markdown("### 🔬 Sound Design / Serum Hints")
                                try:
                                    with open(serum_json_path, "r") as f_serum:
                                        serum_data = json.load(f_serum)
                                    for analysis in serum_data:
                                        st.markdown(f"**Stem:** `{analysis.get('stem','')}`")
                                        st.code(analysis.get("description", ""))
                                        if "serum_patch" in analysis:
                                            st.markdown("**Serum patch suggestion:**")
                                            st.code(json.dumps(analysis["serum_patch"], indent=2))
                                except Exception as e:
                                    st.write(f"Could not read serum_patches.json: {e}")

                            guide_path = output_root / "ABLETON_IMPORT.txt"
                            if guide_path.exists():
                                st.markdown("### 🎛 Ableton Import Guide")
                                with open(guide_path, "r") as f_guide:
                                    st.text(f_guide.read())

                        except Exception as e:
                            st.error(f"Error during processing: {e}")

                else:
                    with st.spinner("Processing batch on GPU worker..."):
                        job_summaries = []
                        jobs_root = WORKSPACE_DIR / "jobs"
                        jobs_root.mkdir(parents=True, exist_ok=True)

                        for up in uploaded_files:
                            try:
                                saved = save_uploaded_file(up)
                                per_file_label = job_label_input or up.name
                                log_to_session(f"Uploading '{up.name}' to GPU worker...")

                                with open(saved, "rb") as f_in:
                                    files = {
                                        "file": (up.name, f_in, up.type or "audio/wav")
                                    }
                                    data = {
                                        "model_name": demucs_model,
                                        "stems_for_midi": ",".join(stems_for_midi),
                                        "run_serum_analysis": str(run_serum).lower(),
                                        "demucs_device": demucs_device or "cuda",
                                    }
                                    resp = requests.post(
                                        f"{WORKER_URL}/process",
                                        files=files,
                                        data=data,
                                        timeout=1800,
                                    )
                                resp.raise_for_status()
                                result = resp.json()
                                job_dir_name = result["job_dir_name"]

                                zip_resp = requests.get(
                                    f"{WORKER_URL}/job_zip",
                                    params={"job_dir": job_dir_name},
                                    timeout=1800,
                                )
                                zip_resp.raise_for_status()

                                with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                                    zf.extractall(jobs_root)

                                job_dir = jobs_root / job_dir_name
                                if not job_dir.exists():
                                    raise RuntimeError(
                                        f"Job folder not found after extracting ZIP for {up.name}."
                                    )

                                meta = load_metadata(job_dir)
                                if not meta:
                                    meta = {
                                        "job_id": result.get("job_id"),
                                        "job_label": per_file_label,
                                        "song_name": result.get("song_name", up.name),
                                        "original_file_name": up.name,
                                        "created_at": datetime.now().isoformat(),
                                        "model_name": demucs_model,
                                        "stems_for_midi": stems_for_midi,
                                        "bpm": result.get("bpm"),
                                        "key": result.get("key"),
                                        "mode": result.get("mode"),
                                        "key_confidence": result.get("key_confidence"),
                                    }
                                    with open(job_dir / "metadata.json", "w") as f_meta:
                                        json.dump(meta, f_meta, indent=2)

                                update_index(WORKSPACE_DIR, meta, job_dir)

                                job_summaries.append(
                                    {
                                        "file": up.name,
                                        "song_name": meta.get("song_name", up.name),
                                        "job_label": per_file_label,
                                        "bpm": meta.get("bpm"),
                                        "key": meta.get("key"),
                                        "mode": meta.get("mode"),
                                        "job_folder": job_dir,
                                    }
                                )

                            except Exception as e:
                                st.error(f"Error processing {up.name}: {e}")

                        if job_summaries:
                            st.success(f"Batch complete. Processed {len(job_summaries)} file(s).")
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
                            st.info("Use the History or Library tabs to inspect jobs and download ZIPs.")

            with col_right:
                st.subheader("Logs")
                st.text_area(
                    "Processing log:",
                    value="\n".join(st.session_state["logs"]),
                    height=400,
                )
    else:
        st.info("Upload one or more WAV/MP3 files to begin.")
