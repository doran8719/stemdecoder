from pathlib import Path
from typing import List

import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from processor import process_song

app = FastAPI(title="BeatDecoder GPU Worker")

# Allow calls from anywhere (Render, local, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace layout inside the pod
WORKSPACE_ROOT = Path("workspace")
WORKSPACE_ROOT.mkdir(exist_ok=True)
UPLOADS_DIR = WORKSPACE_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("htdemucs"),
    stems_for_midi: str = Form("bass,other"),
    run_serum_analysis: bool = Form(True),
    job_label: str = Form(""),
):
    """
    Main processing endpoint for the GPU worker.

    - Saves uploaded file to workspace/uploads
    - Runs processor.process_song using CUDA (Demucs on GPU)
    - DISABLES Basic Pitch (MIDI) on this worker (CoreML is macOS-only)
    - Creates a ZIP of the job folder
    - Returns metadata + job_dir name + zip_name
    """
    try:
        # Save uploaded file
        job_id = uuid.uuid4().hex[:8]
        safe_name = file.filename.replace(" ", "_")
        local_path = UPLOADS_DIR / f"{job_id}_{safe_name}"

        file_bytes = await file.read()
        with local_path.open("wb") as f:
            f.write(file_bytes)

        # Parse stems_for_midi string, but then DISABLE MIDI here
        stems_for_midi_list: List[str] = [
            s.strip() for s in stems_for_midi.split(",") if s.strip()
        ]

        # 🔴 IMPORTANT: Basic Pitch uses CoreML and only works on macOS.
        # This worker runs Linux, so we must disable MIDI here to avoid crashes.
        stems_for_midi_list = []

        log(f"[GPU WORKER] Starting job {job_id} for {file.filename}")

        # Run full processing on GPU for Demucs, etc.
        result = process_song(
            local_path,
            WORKSPACE_ROOT,
            model_name=model_name,
            stems_for_midi=stems_for_midi_list,
            run_serum_analysis=run_serum_analysis,
            log_fn=log,
            job_label=job_label or None,
            demucs_device="cuda",
        )

        output_root = Path(result["output_root"])

        # Pre-create ZIP so /job_zip is cheap and fast
        zip_base = output_root / output_root.name  # e.g. workspace/jobs/Song_xxx/Song_xxx
        zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=output_root)
        zip_name = Path(zip_file).name

        log(f"[GPU WORKER] Job {job_id} complete at {output_root}, zip={zip_name}")

        return {
            "status": "ok",
            "job_id": result["job_id"],
            "job_dir": output_root.name,  # e.g. Dennis Lloyd - Playa (Say That)_20251116_094515
            "zip_name": zip_name,
            "bpm": result.get("bpm"),
            "key": result.get("key"),
            "mode": result.get("mode"),
            "key_confidence": result.get("key_confidence"),
            "serum_analysis": result.get("serum_analysis", []),
        }

    except Exception as e:
        log(f"[GPU WORKER] Error in /process: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/job_zip")
async def job_zip(job_dir: str):
    """
    Return the pre-created ZIP for a given job_dir.
    job_dir should be the folder name under workspace/jobs.
    """
    try:
        jobs_root = WORKSPACE_ROOT / "jobs"
        job_path = jobs_root / job_dir

        if not job_path.exists():
            raise FileNotFoundError(f"Job folder not found: {job_path}")

        # Zip created in /process: <job_dir>/<job_dir>.zip
        zip_path = job_path / f"{job_dir}.zip"

        # Fallback: if for some reason the zip doesn't exist, create it quickly.
        if not zip_path.exists():
            zip_base = job_path / job_path.name
            zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=job_path)
            zip_path = Path(zip_file)

        log(f"[GPU WORKER] Sending ZIP {zip_path}")
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
        )

    except Exception as e:
        log(f"[GPU WORKER] Error in /job_zip: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/")
async def root():
    return {"status": "ok", "message": "BeatDecoder GPU worker is running"}
