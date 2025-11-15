import subprocess
from pathlib import Path
import shutil
import uuid
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
import json

import numpy as np
import librosa

from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

LogFn = Optional[Callable[[str], None]]


def _log(msg: str, log_fn: LogFn = None):
    if log_fn:
        log_fn(msg)
    print(msg)


def run_demucs(
    audio_file: Path,
    model_name: str,
    device: Optional[str] = None,
    log_fn: LogFn = None,
) -> Path:
    """
    Run Demucs on the given audio file using the specified model and device.

    Demucs is run from the project root, writing into:
        ./separated/<model_name>/<song_name>/

    Before running, we delete any existing folder for this song name so we
    never reuse stale stems. Returns the path to the fresh stems directory.
    """
    _log(f"Running Demucs with model '{model_name}'...", log_fn)

    project_root = Path.cwd()
    separated_root = project_root / "separated"
    song_name = audio_file.stem

    # Clean old stems for this song
    if separated_root.exists():
        for model_folder in separated_root.iterdir():
            if not model_folder.is_dir():
                continue
            candidate = model_folder / song_name
            if candidate.exists():
                _log(f"Removing old stems at {candidate}", log_fn)
                shutil.rmtree(candidate)

    # Build demucs command
    cmd = ["demucs", "-n", model_name]
    if device and device.lower() not in ("auto", ""):
        cmd += ["-d", device]
    cmd.append(str(audio_file))

    # Try with requested device; fallback to CPU if that fails
    try:
        subprocess.run(cmd, check=True, cwd=str(project_root))
    except subprocess.CalledProcessError:
        if device and device.lower() not in ("cpu", "auto", ""):
            _log("Demucs failed on requested device. Falling back to CPU...", log_fn)
            cmd_cpu = ["demucs", "-n", model_name, "-d", "cpu", str(audio_file)]
            subprocess.run(cmd_cpu, check=True, cwd=str(project_root))
        else:
            raise

    # Locate fresh stems
    if separated_root.exists():
        for model_folder in separated_root.iterdir():
            if not model_folder.is_dir():
                continue
            candidate = model_folder / song_name
            if candidate.exists():
                _log(f"Found new stems at {candidate}", log_fn)
                return candidate

    raise RuntimeError("Could not find Demucs output directory for this job.")


def run_basic_pitch_on_stems(
    stems_dir: Path,
    output_root: Path,
    stems_for_midi: List[str],
    log_fn: LogFn = None,
):
    """
    Run Basic Pitch on selected stems. Writes MIDI into output_root / 'midi'.
    """
    midi_dir = output_root / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    stem_map = {
        "bass": "bass.wav",
        "other": "other.wav",
        "vocals": "vocals.wav",
        "drums": "drums.wav",
    }

    for stem_key in stems_for_midi:
        stem_name = stem_map.get(stem_key)
        if not stem_name:
            continue

        stem_path = stems_dir / stem_name
        if not stem_path.exists():
            _log(f"Stem {stem_name} missing, skipping MIDI.", log_fn)
            continue

        _log(f"Running Basic Pitch on {stem_name}...", log_fn)

        predict_and_save(
            [str(stem_path)],
            str(midi_dir),
            True,
            False,
            False,
            False,
            ICASSP_2022_MODEL_PATH,
        )


def prepare_output_folder(stem_dir: Path, workspace: Path) -> Path:
    """
    Copy stems from Demucs output into a clean 'job' output folder.
    Returns the output folder path.
    """
    song_name = stem_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = workspace / "jobs" / f"{song_name}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    for stem_file in stem_dir.iterdir():
        if stem_file.suffix.lower() == ".wav":
            shutil.copy(stem_file, output_root / stem_file.name)

    return output_root


def analyze_global_track(audio_path: Path, log_fn: LogFn = None) -> Dict[str, Any]:
    """
    Analyze full track for BPM and key.
    """
    try:
        _log("Analyzing track for BPM and key...", log_fn)
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if len(y) == 0:
            return {"bpm": None, "key": None, "mode": None, "key_confidence": None}

        # BPM (tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)

        # Key detection via chroma + Krumhansl profiles
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        major_profile /= major_profile.sum()
        minor_profile /= minor_profile.sum()

        chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-9)

        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_key = None
        best_mode = None
        best_score = -1.0
        second_best = -1.0

        for i in range(12):
            maj_score = float(np.dot(chroma_norm, np.roll(major_profile, i)))
            min_score = float(np.dot(chroma_norm, np.roll(minor_profile, i)))

            if maj_score > best_score:
                second_best = best_score
                best_score = maj_score
                best_key = notes[i]
                best_mode = "major"
            elif maj_score > second_best:
                second_best = maj_score

            if min_score > best_score:
                second_best = best_score
                best_score = min_score
                best_key = notes[i]
                best_mode = "minor"
            elif min_score > second_best:
                second_best = min_score

        if best_score > 0:
            confidence = (best_score - max(second_best, 0.0)) / best_score
        else:
            confidence = 0.0

        return {
            "bpm": bpm,
            "key": best_key,
            "mode": best_mode,
            "key_confidence": float(confidence),
        }
    except Exception as e:
        _log(f"Error during BPM/key analysis: {e}", log_fn)
        return {"bpm": None, "key": None, "mode": None, "key_confidence": None}


def analyze_chords_and_sections(audio_path: Path, log_fn: LogFn = None) -> Dict[str, Any]:
    """
    Very rough chord and section analysis based on chroma and energy.
    """
    try:
        _log("Analyzing chords and sections...", log_fn)
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if len(y) == 0:
            return {"chords": [], "sections": []}

        hop_length = 2048
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        triad_templates = {
            "C": [0, 4, 7],
            "C#": [1, 5, 8],
            "D": [2, 6, 9],
            "D#": [3, 7, 10],
            "E": [4, 8, 11],
            "F": [5, 9, 0],
            "F#": [6, 10, 1],
            "G": [7, 11, 2],
            "G#": [8, 0, 3],
            "A": [9, 1, 4],
            "A#": [10, 2, 5],
            "B": [11, 3, 6],
        }

        def best_triad(vec):
            best_name = "N"
            best_score = 0.0
            for name, degrees in triad_templates.items():
                score = float(vec[degrees[0]] + vec[degrees[1]] + vec[degrees[2]])
                if score > best_score:
                    best_score = score
                    best_name = name
            return best_name, best_score

        chords = []
        window = 8
        for i in range(0, chroma.shape[1], window):
            seg = chroma[:, i : i + window]
            if seg.shape[1] == 0:
                continue
            avg = seg.mean(axis=1)
            name, score = best_triad(avg)
            start_t = float(times[i])
            end_idx = min(i + window, len(times) - 1)
            end_t = float(times[end_idx])
            chords.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "chord": name,
                    "score": score,
                }
            )

        rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=2048)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=2048)
        thresh = float(np.median(rms) * 0.7)

        sections = []
        in_section = False
        sec_start = 0.0
        for t, val in zip(rms_times, rms):
            if val >= thresh and not in_section:
                in_section = True
                sec_start = float(t)
            elif val < thresh and in_section:
                in_section = False
                sections.append({"start": sec_start, "end": float(t)})
        if in_section:
            sections.append({"start": sec_start, "end": float(rms_times[-1])})

        return {"chords": chords, "sections": sections}
    except Exception as e:
        _log(f"Error during chord/section analysis: {e}", log_fn)
        return {"chords": [], "sections": []}


def analyze_stem_for_synth(stem_path: Path) -> Dict[str, Any]:
    """
    Heuristic sound-design analysis (Serum-style suggestions).
    """
    try:
        y, sr = librosa.load(str(stem_path), sr=None, mono=True)
        if len(y) == 0:
            return {"stem": stem_path.name, "description": "Audio empty.", "features": {}}

        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
        rolloff = float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())

        high_band = freqs >= 4000
        if np.any(high_band):
            brightness = float(S[high_band].sum() / (S.sum() + 1e-9))
        else:
            brightness = 0.0

        if centroid < 800:
            timbre = "warm / sub / low-mid focused"
            osc_hint = "Osc 1: sine or triangle; Osc 2: subtle saw for harmonics."
        elif centroid < 2000:
            timbre = "mid-focused body tone"
            osc_hint = "Osc 1: saw; Osc 2: square/triangle for thickness."
        else:
            timbre = "bright / airy / presence-heavy"
            osc_hint = "Osc 1: bright saw; Osc 2: noise or slightly detuned saw."

        if brightness > 0.3:
            filter_suggestion = "Use a low-pass filter around ~5–8 kHz; maybe 24 dB slope."
        else:
            filter_suggestion = "Minimal filtering; gentle low-pass only if needed."

        if zcr < 0.05:
            waveform = "Likely sine/triangle (smooth, few transients)."
        elif zcr < 0.15:
            waveform = "Likely saw/square (rich harmonics)."
        else:
            waveform = "Very bright/noisy or strongly transient."

        frame = 2048
        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        if len(rms) > 10:
            start_rms = float(np.mean(rms[:5]))
            mid_rms = float(np.mean(rms[len(rms) // 3 : 2 * len(rms) // 3]))
            end_rms = float(np.mean(rms[-5:]))

            if mid_rms > start_rms * 1.2 and mid_rms > end_rms * 1.2:
                env_shape = "Attack-Decay (pluck-like)."
            elif end_rms > mid_rms * 1.1:
                env_shape = "Rising/swell envelope (pad or build)."
            else:
                env_shape = "Fairly steady sustain."
        else:
            env_shape = "Envelope unclear (short or noisy segment)."

        description = (
            f"Stem '{stem_path.name}' analysis:\n"
            f"- Timbre: {timbre}\n"
            f"- Waveform: {waveform}\n"
            f"- Filter: {filter_suggestion}\n"
            f"- Envelope: {env_shape}\n"
            f"- Spectral centroid: {int(centroid)} Hz\n"
            f"- Spectral rolloff: {int(rolloff)} Hz\n"
            f"- Brightness ratio: {brightness:.3f}\n\n"
            "Serum starting point:\n"
            f"- {osc_hint}\n"
            "- Filter: use MG Low 12/24 with cutoff near the rolloff frequency.\n"
            "- Env 1: tailor attack/decay to match the envelope description.\n"
            "- Add slight unison detune if sound feels too thin."
        )

        return {
            "stem": stem_path.name,
            "description": description,
            "features": {
                "spectral_centroid": centroid,
                "spectral_rolloff": rolloff,
                "zcr": zcr,
                "brightness": brightness,
            },
        }
    except Exception as e:
        return {"stem": stem_path.name, "description": f"Error: {e}", "features": {}}


def build_serum_patch_suggestion(stem_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a structured Serum-style patch suggestion from analysis.
    This is NOT a real .fxp, but a JSON-friendly spec of oscillators, filter, and env.
    """
    features = stem_analysis.get("features", {})
    centroid = features.get("spectral_centroid")
    brightness = features.get("brightness")
    zcr = features.get("zcr")

    osc1_type = "saw"
    osc2_type = "sine"
    if zcr is not None:
        if zcr < 0.05:
            osc1_type = "sine"
            osc2_type = "triangle"
        elif zcr < 0.15:
            osc1_type = "saw"
            osc2_type = "square"
        else:
            osc1_type = "saw"
            osc2_type = "noise"

    filter_type = "lowpass"
    cutoff_hz = centroid if centroid is not None else 8000
    if brightness is not None and brightness < 0.15:
        cutoff_hz *= 1.2

    env = {
        "attack": 0.005,
        "decay": 0.2,
        "sustain": 0.7,
        "release": 0.2,
    }

    return {
        "osc1": {
            "type": osc1_type,
            "unison": 4,
            "detune": 0.12,
        },
        "osc2": {
            "type": osc2_type,
            "unison": 2,
            "detune": 0.07,
        },
        "filter": {
            "type": filter_type,
            "cutoff_hz": cutoff_hz,
            "resonance": 0.2,
        },
        "env_amp": env,
        "notes": "Use this as a starting point in Serum. Manually adjust to taste.",
    }


def write_ableton_guide(
    output_root: Path,
    metadata: Dict[str, Any],
    global_analysis: Dict[str, Any],
    chord_section_analysis: Dict[str, Any],
):
    """
    Write a text guide for recreating the session in Ableton Live.
    """
    guide_path = output_root / "ABLETON_IMPORT.txt"
    midi_dir = output_root / "midi"

    bpm = global_analysis.get("bpm")
    key = global_analysis.get("key")
    mode = global_analysis.get("mode")

    with open(guide_path, "w") as f:
        f.write("Ableton Live Import Guide\n")
        f.write("=========================\n\n")
        f.write(f"Song: {metadata.get('song_name', 'Unknown')}\n")
        f.write(f"Original file: {metadata.get('original_file_name', 'Unknown')}\n")
        f.write(f"Demucs model: {metadata.get('model_name', 'Unknown')}\n")
        f.write(f"BPM (estimated): {bpm if bpm is not None else 'Unknown'}\n")
        if key and mode:
            f.write(f"Key (estimated): {key} {mode}\n")
        else:
            f.write("Key (estimated): Unknown\n")
        f.write("\nSuggested Ableton Setup:\n")
        f.write("1. Set project BPM to the estimated BPM above.\n")
        f.write("2. If key is known, set scale in Ableton's scale device or label the project.\n\n")

        f.write("Audio Tracks (stems):\n")
        for stem_file in sorted(output_root.iterdir()):
            if stem_file.suffix.lower() == ".wav":
                f.write(f"- {stem_file.name}\n")

        if midi_dir.exists():
            f.write("\nMIDI Tracks:\n")
            for midi_file in sorted(midi_dir.iterdir()):
                if midi_file.suffix.lower() == ".mid":
                    f.write(
                        f"- {midi_file.name} (Create a MIDI track, load a synth like Serum, then import this MIDI)\n"
                    )

        chords = chord_section_analysis.get("chords", [])
        sections = chord_section_analysis.get("sections", [])

        if chords:
            f.write("\nEstimated Chords (rough):\n")
            for ch in chords:
                f.write(
                    f"- {ch['start']:.1f}s to {ch['end']:.1f}s → {ch['chord']} (score {ch['score']:.2f})\n"
                )

        if sections:
            f.write("\nLoudness-based Sections (rough):\n")
            for sec in sections:
                f.write(f"- {sec['start']:.1f}s to {sec['end']:.1f}s\n")

        f.write(
            "\nTips:\n"
            "- Use the BPM and key to align new parts and keep everything in tune.\n"
            "- For bass MIDI, load a bass patch in Serum and adjust envelopes/filters.\n"
            "- For other/melody MIDI, try lead or pad patches.\n"
            "- Use the Serum patch suggestions JSON to configure oscillators and filters in Serum.\n"
        )


def update_index(workspace: Path, metadata: Dict[str, Any], job_path: Path):
    """
    Maintain a simple JSON index of all jobs for fast search/filter.
    """
    index_path = workspace / "index.json"
    index: List[Dict[str, Any]] = []
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except Exception:
            index = []

    entry = {
        "job_id": metadata.get("job_id"),
        "job_label": metadata.get("job_label"),
        "song_name": metadata.get("song_name"),
        "original_file_name": metadata.get("original_file_name"),
        "created_at": metadata.get("created_at"),
        "model_name": metadata.get("model_name"),
        "stems_for_midi": metadata.get("stems_for_midi"),
        "bpm": metadata.get("bpm"),
        "key": metadata.get("key"),
        "mode": metadata.get("mode"),
        "key_confidence": metadata.get("key_confidence"),
        "job_path": str(job_path),
    }

    index = [e for e in index if e.get("job_id") != entry["job_id"]]
    index.append(entry)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def process_song(
    uploaded_file_path: Path,
    workspace: Path,
    model_name: str = "htdemucs",
    stems_for_midi: Optional[List[str]] = None,
    run_serum_analysis: bool = True,
    log_fn: LogFn = None,
    job_label: Optional[str] = None,
    demucs_device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function to process a song:
    - BPM + key detection on full track
    - chord / section analysis
    - Demucs separation
    - Copy stems into job folder
    - Basic Pitch MIDI extraction
    - Serum-style sound-design analysis (+ JSON patch suggestions)
    - Ableton import guide
    - metadata.json + index.json for UI/history/search
    """
    if stems_for_midi is None:
        stems_for_midi = ["bass", "other"]

    workspace.mkdir(exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    work_dir = workspace / "work" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    local_audio = work_dir / uploaded_file_path.name
    shutil.copy(uploaded_file_path, local_audio)

    _log(f"Starting job {job_id} for {uploaded_file_path.name}", log_fn)

    global_analysis = analyze_global_track(local_audio, log_fn)
    chord_section_analysis = analyze_chords_and_sections(local_audio, log_fn)

    stem_src_dir = run_demucs(local_audio, model_name=model_name, device=demucs_device, log_fn=log_fn)

    output_root = prepare_output_folder(stem_src_dir, workspace)
    stems_dir = output_root

    run_basic_pitch_on_stems(stems_dir, output_root, stems_for_midi, log_fn)

    serum_results: List[Dict[str, Any]] = []
    if run_serum_analysis:
        for stem_name in ["bass.wav", "other.wav"]:
            stem_path = stems_dir / stem_name
            if stem_path.exists():
                analysis = analyze_stem_for_synth(stem_path)
                patch = build_serum_patch_suggestion(analysis)
                analysis["serum_patch"] = patch
                serum_results.append(analysis)

        serum_json_path = output_root / "serum_patches.json"
        with open(serum_json_path, "w") as f:
            json.dump(serum_results, f, indent=2)

    metadata = {
        "job_id": job_id,
        "job_label": job_label,
        "song_name": stem_src_dir.name,
        "original_file_name": uploaded_file_path.name,
        "created_at": datetime.now().isoformat(),
        "model_name": model_name,
        "stems_for_midi": stems_for_midi,
        "bpm": global_analysis.get("bpm"),
        "key": global_analysis.get("key"),
        "mode": global_analysis.get("mode"),
        "key_confidence": global_analysis.get("key_confidence"),
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    write_ableton_guide(output_root, metadata, global_analysis, chord_section_analysis)
    update_index(workspace, metadata, output_root)

    _log(f"Job {job_id} complete. Output at {output_root}", log_fn)

    return {
        "job_id": job_id,
        "output_root": output_root,
        "stems_dir": stems_dir,
        "midi_dir": output_root / "midi",
        "serum_analysis": serum_results,
        "model_name": model_name,
        "stems_for_midi": stems_for_midi,
        "song_name": metadata["song_name"],
        "job_label": job_label,
        "bpm": metadata["bpm"],
        "key": metadata["key"],
        "mode": metadata["mode"],
        "key_confidence": metadata["key_confidence"],
        "chords": chord_section_analysis.get("chords", []),
        "sections": chord_section_analysis.get("sections", []),
    }

