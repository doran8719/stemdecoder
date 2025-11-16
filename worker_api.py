from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import uuid
import shutil
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from processor import process_song

app = FastAPI(title="BeatDecoder GPU Worker", version="0.2.0")

# All worker data lives under this workspace
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# In-memory job store: job_id -> info
# This will reset if the pod restarts, which is fine for v1.
JOBS: Dict[str, Dict[str, Any]] = {}


def _job_log(job_id: str):
    """
    Create a log function that appends messages to the job log
    and prints them to stdout.
    """
    def log(msg: str):
        print(f"[JOB {job_id}] {msg}")
        job = JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)
    return log


def _ensure_zip_for_job(job_folder: Path) -> Path:
    """
    Create (or reuse) a ZIP archive for a completed job folder.
    """
    zip_path = job_folder.with_suffix(".zip")
    if not zip_path.exists():
        base_name = str(job_folder)
        # This will create base_name + ".zip"
        shutil.make_archive(base_name, "zip", root_dir=job_folder)
    return zip_path


def _run_job(
    job_id: str,
    audio_path: Path,
    model_name: str,
    demucs_device: Optional[str],
    run_serum_analysis: bool,
    job_label: Optional[str],
    stems_for_midi: Optional[List[str]],
):
    """
    Background job: run the full pipeline on the GPU worker.

    NOTE: For now, we DISABLE MIDI extraction on the GPU worker because
    Basic Pitch's CoreML backend only works on macOS and crashes on Linux.
    We'll reintroduce MIDI later with a Linux-friendly backend.
    """
    log = _job_log(job_id)
    JOBS[job_id]["status"] = "running"

    try:
        # TEMP: no MIDI on GPU to avoid CoreML/Linux crash
        safe_stems_for_midi: List[str] = []

        log(f"Starting GPU job for {audio_path.name}")
        result = process_song(
            uploaded_file_path=audio_path,
            workspace=WORKSPACE,
            model_name=model_name,
            stems_for_midi=safe_stems_for_midi,
            run_serum_analysis=run_serum_analysis,
            log_fn=log,
            job_label=job_label,
            demucs_device=demucs_device,
        )

        output_root: Path = result["output_root"]
        zip_path = _ensure_zip_for_job(output_root)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "job_folder": str(output_root),
            "zip_path": str(zip_path),
            "song_name": result.get("song_name"),
            "job_label": job_label,
            "bpm": result.get("bpm"),
            "key": result.get("key"),
            "mode": result.get("mode"),
            "key_confidence": result.get("key_confidence"),
        }
        log(f"Job complete. Output at {output_root}")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        log(f"Job failed: {e}")


@app.post("/start_job")
async def start_job(
    file: UploadFile = File(...),
    model_name: str = Form("htdemucs"),
    demucs_device: str = Form("cuda"),
    run_serum_analysis: bool = Form(True),
    job_label: Optional[str] = Form(None),
    stems_for_midi: Optional[str] = Form(None),
):
    """
    Start a new GPU job.

    Returns quickly with:
    {
        "job_id": "...",
        "status": "queued"
    }

    The heavy work runs in a background thread.
    """
    job_id = uuid.uuid4().hex[:8]
    incoming_dir = WORKSPACE / "incoming" / job_id
    incoming_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file to disk
    audio_path = incoming_dir / file.filename
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Parse optional stems_for_midi list (unused for now, but kept for future)
    stems_list: Optional[List[str]] = None
    if stems_for_midi:
        stems_list = [s.strip() for s in stems_for_midi.split(",") if s.strip()]

    JOBS[job_id] = {
        "status": "queued",
        "log": [f"Job {job_id} created for {file.filename}"],
        "result": None,
        "error": None,
    }

    # Kick off background thread
    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            audio_path,
            model_name,
            demucs_device if demucs_device not in ("", "auto") else None,
            run_serum_analysis,
            job_label,
            stems_list,
        ),
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
        }
    )


@app.get("/job_status")
async def job_status(job_id: str):
    """
    Check the status of a job.

    Response example:
    {
        "job_id": "...",
        "status": "running",
        "log": [...],
        "result": {...}  # only when status == "done"
        "error": "..."   # only when status == "error"
    }
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "log": job.get("log", []),
            "result": job.get("result"),
            "error": job.get("error"),
        }
    )


@app.get("/job_zip")
async def job_zip(job_id: str):
    """
    Download the ZIP archive for a completed job.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    result = job.get("result") or {}
    zip_path_str = result.get("zip_path")
    if not zip_path_str:
        raise HTTPException(status_code=500, detail="ZIP path missing for job")

    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found on worker")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )
