from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

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
    - Returns metadata and internal job path (we'll later add real download URLs)
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
    #    Force demucs to use the requested device (default "cuda")
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

    # 4) Return basic metadata + paths (for now)
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
        }
    )
