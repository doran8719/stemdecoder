import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# ---------------------------------------------------------------------
# Workspace root (outer app imports override this)
# You can redefine WORKSPACE_DIR in config.py and import it everywhere.
# ---------------------------------------------------------------------
WORKSPACE_DIR = Path("workspace")


# ---------------------------------------------------------------------
# Ensure workspace directories exist
# ---------------------------------------------------------------------
def init_workspace():
    """
    Ensures the workspace and its subdirectories exist.
    Called automatically by app.py and worker_api.py.
    """
    WORKSPACE_DIR.mkdir(exist_ok=True)
    for sub in ["uploads", "jobs", "tmp_zips"]:
        (WORKSPACE_DIR / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> Path:
    """
    Saves a Streamlit-uploaded file into workspace/uploads/
    Returns the path to the saved file.
    """
    uploads_dir = WORKSPACE_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = uploads_dir / safe_name

    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return out_path


# ---------------------------------------------------------------------
# Job folder creation & ZIP extraction
# ---------------------------------------------------------------------
def create_job_folder(job_id: str) -> Path:
    """
    Creates a workspace/jobs/<job_id> folder.
    """
    job_root = WORKSPACE_DIR / "jobs" / job_id
    if job_root.exists():
        shutil.rmtree(job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    return job_root


def extract_zip_to_job(zip_bytes: bytes, job_id: str) -> Path:
    """
    Takes raw ZIP bytes and extracts into workspace/jobs/<job_id>.
    Returns the job_root path.
    """
    job_root = create_job_folder(job_id)
    zip_path = job_root / f"{job_id}.zip"

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(job_root)

    return job_root


# ---------------------------------------------------------------------
# Job listing & metadata loading
# ---------------------------------------------------------------------
def list_jobs() -> List[Path]:
    """
    Returns a list of job directories sorted newest-first.
    """
    jobs_root = WORKSPACE_DIR / "jobs"
    if not jobs_root.exists():
        return []

    return sorted(
        [p for p in jobs_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_metadata(job_dir: Path) -> Dict[str, Any]:
    """
    Reads metadata.json if present.
    """
    meta_path = job_dir / "metadata.json"

    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}


def write_metadata(job_dir: Path, metadata: Dict[str, Any]):
    """
    Writes metadata.json safely.
    """
    meta_path = job_dir / "metadata.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass


def infer_song_name(job_dir: Path, meta: Dict[str, Any]) -> str:
    """
    Infer title from metadata or job folder name.
    """
    if "song_name" in meta and meta["song_name"]:
        return meta["song_name"]

    name = job_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def infer_created_at(job_dir: Path, meta: Dict[str, Any]) -> str:
    """
    Reads created_at from metadata or falls back to folder timestamp.
    """
    if "created_at" in meta and meta["created_at"]:
        return meta["created_at"]

    return datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat()


# ---------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------
def build_index_from_jobs() -> List[Dict[str, Any]]:
    """
    Creates workspace/index.json for fast searching.
    """
    jobs = list_jobs()
    entries = []

    for job_dir in jobs:
        meta = load_metadata(job_dir)
        entry = {
            "job_id": meta.get("job_id", job_dir.name),
            "job_path": str(job_dir.resolve()),
            "song_name": infer_song_name(job_dir, meta),
            "job_label": meta.get("job_label") or job_dir.name,
            "bpm": meta.get("bpm"),
            "key": meta.get("key"),
            "mode": meta.get("mode"),
            "created_at": infer_created_at(job_dir, meta),
        }
        entries.append(entry)

    index_path = WORKSPACE_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)

    return entries


def load_index() -> List[Dict[str, Any]]:
    """
    Reads index.json if exists.
    """
    index_path = WORKSPACE_DIR / "index.json"
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    return []
