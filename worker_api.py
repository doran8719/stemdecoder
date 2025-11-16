from pathlib import Path
from typing import List

import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from processor import process_song

app = FastAPI()

# Dedicated workspace on the GPU worker
WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("htdemucs"),
    stems_for_midi: str = Form("bass,other"),
    run_serum_analysis: bool = Form(True),
    demucs_device: str = Form("cuda"),
):
    """
    GPU worker endpoint.

    - Receives an uploaded audio file
    - Calls process_song() with demucs_device set (default 'cuda')
    - Returns metadata and job folder name
    """

    # 1) Save the uploaded file into this worker's workspace
    uploads_dir = WORKSPACE_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    local_path = uploads_dir / file.filename

    with open(local_path, "wb") as f:
        f.write(await file.read())

    # 2) Parse stems_for_midi: comma-separated -> list
    stems_list: List[str] = [
        s.strip() for s in stems_for_midi.split(",") if s.strip()
    ]

    # 3) Run the existing processor on this GPU box
    result = process_song(
        uploaded_file_path=local_path,
        workspace=WORKSPACE_DIR,
        model_name=model_name,
        stems_for_midi=stems_list,
        run_serum_analysis=run_serum_analysis,
        log_fn=None,
        job_label=None,
        demucs_device=demucs_device or "cuda",
    )

    job_dir_name = result["output_root"].name

    return JSONResponse(
        {
            "job_id": result["job_id"],
            "song_name": result["song_name"],
            "job_label": result["job_label"],
            "bpm": result["bpm"],
            "key": result["key"],
            "mode": result["mode"],
            "key_confidence": result["key_confidence"],
            "job_path": str(result["output_root"]),
            "job_dir_name": job_dir_name,
        }
    )


@app.get("/job_zip")
def job_zip(job_dir: str):
    """
    Return a ZIP of a finished job folder.

    The frontend will:
    - call /process to start work
    - get job_dir_name
    - call /job_zip?job_dir=<job_dir_name> to download the full bundle
    """

    job_path = WORKSPACE_DIR / "jobs" / job_dir
    if not job_path.exists() or not job_path.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    zip_path = job_path / f"{job_dir}.zip"
    if not zip_path.exists():
        # Create ZIP: <job_dir>.zip inside the job folder
        shutil.make_archive(str(job_path / job_dir), "zip", root_dir=job_path)

    return FileResponse(
        path=str(zip_path),
        filename=f"{job_dir}.zip",
        media_type="application/zip",
    )
