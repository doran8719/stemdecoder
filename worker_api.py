import os
import shutil
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from processor import process_song


# -----------------------------------------------------------------------------
# Workspace selection: RunPod vs Local
# -----------------------------------------------------------------------------
def get_workspace_root() -> Path:
    """
    On RunPod: use /workspace (writable pod volume).
    On local dev: use ./workspace inside the project folder.
    """
    pod_path = Path("/workspace")
    if pod_path.exists() and os.access(pod_path, os.W_OK):
        # Running on RunPod or similar environment
        return pod_path

    # Fallback: local dev - use ./workspace next to this file
    return Path(__file__).resolve().parent / "workspace"


WORKSPACE_ROOT = get_workspace_root()
JOBS_ROOT = WORKSPACE_ROOT / "jobs"
UPLOADS_ROOT = WORKSPACE_ROOT / "uploads"
TMP_ZIPS_ROOT = WORKSPACE_ROOT / "tmp_zips"

for p in [WORKSPACE_ROOT, JOBS_ROOT, UPLOADS_ROOT, TMP_ZIPS_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# In-memory job registry
# -----------------------------------------------------------------------------
# JOBS[job_id] = {
#   "status": "queued" | "running" | "done" | "error",
#   "log": [str, ...],
#   "created_at": datetime,
#   "result": {...} or None,
#   "error": str or None,
#   "job_dir": Path or None,
#   "upload_path": Path or None,
# }
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="BeatDecoder GPU Worker", version="1.0.0")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def log(job_id: str, message: str) -> None:
    """Append a timestamped log message to a job."""
    job = JOBS.get(job_id)
    if not job:
        return
    ts = now_utc().isoformat()
    job["log"].append(f"[{ts}] {message}")


def safe_result_from_process(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only small, JSON-safe pieces of the process_song result
    that the Streamlit frontend cares about.
    """
    return {
        "bpm": result.get("bpm"),
        "key": result.get("key"),
        "mode": result.get("mode"),
        "key_confidence": result.get("key_confidence"),
        "song_name": result.get("song_name"),
    }


def cleanup_job(job_id: str, zip_path: Optional[Path] = None) -> None:
    """
    Delete job_dir, upload_path, optional zip file, and remove job
    from the in-memory registry. This runs as a background task
    after /job_zip returns.
    """
    job = JOBS.get(job_id)
    if not job:
        return

    job_dir: Optional[Path] = job.get("job_dir")
    upload_path: Optional[Path] = job.get("upload_path")

    try:
        if job_dir and job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        traceback.print_exc()

    try:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
    except Exception:
        traceback.print_exc()

    if zip_path is not None:
        try:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
        except Exception:
            traceback.print_exc()

    # Finally, remove from registry
    JOBS.pop(job_id, None)


def run_job(
    job_id: str,
    uploaded_path: Path,
    model_name: str,
    demucs_device: str,
    run_serum_analysis: bool,
    job_label: str,
    stems_for_midi: List[str],
) -> None:
    """
    Background worker that runs process_song and updates job status.
    """
    job = JOBS[job_id]
    job["status"] = "running"
    log(job_id, f"Job started. Model={model_name}, device={demucs_device}")

    try:
        # process_song expects workspace_root similar to your Streamlit app.
        # We pass WORKSPACE_ROOT, which is /workspace on RunPod and ./workspace locally.
        result = process_song(
            uploaded_path,
            WORKSPACE_ROOT,
            model_name=model_name,
            stems_for_midi=stems_for_midi,
            run_serum_analysis=run_serum_analysis,
            log_fn=lambda msg: log(job_id, msg),
            job_label=job_label or None,
            demucs_device=demucs_device,
        )

        # Output root (job_dir) is provided by process_song, same as in app.py
        output_root = result["output_root"]
        if isinstance(output_root, str):
            output_root = Path(output_root)

        job["job_dir"] = output_root
        job["result"] = safe_result_from_process(result)
        job["status"] = "done"
        log(job_id, "Job completed successfully.")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(job_id, f"Job failed with error: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/start_job")
async def start_job(
    file: UploadFile = File(...),
    model_name: str = Form("htdemucs"),
    demucs_device: str = Form("cuda"),
    run_serum_analysis: str = Form("true"),
    job_label: str = Form(""),
    stems_for_midi: str = Form("bass,other"),
) -> Dict[str, Any]:
    """
    Start a new job.

    The Streamlit frontend sends:
      - file (wav/mp3)
      - model_name
      - demucs_device ("cuda" for GPU; on local this will still say "cuda" but run on CPU)
      - run_serum_analysis ("true"/"false")
      - job_label
      - stems_for_midi (comma-separated)
    """

    # unique ID: timestamp + microseconds
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    # parse booleans & lists
    run_serum = run_serum_analysis.lower() == "true"
    stems_list = [s.strip() for s in stems_for_midi.split(",") if s.strip()]

    # store uploaded file under WORKSPACE_ROOT/uploads
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename.replace(" ", "_") if file.filename else "uploaded_audio"
    upload_path = UPLOADS_ROOT / f"{job_id}_{safe_name}"

    with upload_path.open("wb") as f_out:
        content = await file.read()
        f_out.write(content)

    # init job record
    JOBS[job_id] = {
        "status": "queued",
        "log": [],
        "created_at": now_utc(),
        "result": None,
        "error": None,
        "job_dir": None,
        "upload_path": upload_path,
    }

    log(job_id, f"Job created. Upload saved to {upload_path}")

    # start background thread
    t = threading.Thread(
        target=run_job,
        args=(
            job_id,
            upload_path,
            model_name,
            demucs_device,
            run_serum,
            job_label,
            stems_list,
        ),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


@app.get("/job_status")
def job_status(job_id: str = Query(..., description="ID returned from /start_job")) -> Dict[str, Any]:
    """
    Return the status of a job:
      - status: queued | running | done | error
      - log: list of log lines
      - result: small JSON dict with bpm/key/mode/etc when done
      - error: message if status == "error"
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "log": job["log"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.get("/job_zip")
def job_zip(job_id: str = Query(..., description="ID returned from /start_job")):
    """
    Create a ZIP of the job directory and return it.

    After the ZIP is sent, the job directory, upload, and ZIP file
    are automatically cleaned up (auto-clean).
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (status={job['status']})",
        )

    job_dir: Path = job.get("job_dir")
    if not job_dir or not job_dir.exists():
        raise HTTPException(status_code=500, detail="Job output directory missing")

    # create ZIP in TMP_ZIPS_ROOT
    TMP_ZIPS_ROOT.mkdir(parents=True, exist_ok=True)
    zip_base = TMP_ZIPS_ROOT / f"{job_id}"
    # shutil.make_archive adds ".zip"
    zip_file_path_str = shutil.make_archive(str(zip_base), "zip", root_dir=job_dir)
    zip_path = Path(zip_file_path_str)

    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create ZIP file")

    # stream the ZIP and then cleanup job + zip in background
    def iterfile():
        with zip_path.open("rb") as f:
            yield from f

    background = BackgroundTask(cleanup_job, job_id=job_id, zip_path=zip_path)

    return StreamingResponse(
        iterfile(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_job.zip"'},
        background=background,
    )


# Optional: allow local debugging with `python worker_api.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("worker_api:app", host="0.0.0.0", port=8000, reload=False)
