import requests
import time
from typing import Dict, Any, Optional


class GPUWorkerError(Exception):
    """Raised for GPU worker failures."""
    pass


# ----------------------------------------------------------
# Start a job on the GPU worker
# ----------------------------------------------------------
def start_gpu_job(
    worker_url: str,
    file_bytes: bytes,
    filename: str,
    settings: Dict[str, Any],
    timeout: int = 60
) -> str:
    """
    Starts a GPU worker job and returns job_id.

    settings = {
        "model_name": str,
        "demucs_device": str,
        "run_serum_analysis": bool,
        "stems_for_midi": "bass,other,vocals",
        "job_label": str
    }
    """
    worker_url = worker_url.rstrip("/")

    files = {
        "file": (filename, file_bytes, "application/octet-stream"),
    }

    # Convert bool to str for FastAPI
    data = {
        "model_name": settings.get("model_name", "htdemucs"),
        "demucs_device": settings.get("demucs_device", "cuda"),
        "run_serum_analysis": "true" if settings.get("run_serum_analysis") else "false",
        "stems_for_midi": settings.get("stems_for_midi", ""),
        "job_label": settings.get("job_label", ""),
    }

    try:
        resp = requests.post(
            f"{worker_url}/start_job",
            files=files,
            data=data,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        raise GPUWorkerError(f"Failed to start GPU job: {e}")

    js = resp.json()
    if "job_id" not in js:
        raise GPUWorkerError(f"GPU worker returned no job_id: {js}")

    return js["job_id"]


# ----------------------------------------------------------
# Poll status
# ----------------------------------------------------------
def poll_gpu_status(
    worker_url: str,
    job_id: str,
    max_wait_seconds: int = 900,
    poll_interval: int = 3
) -> Dict[str, Any]:
    """
    Polls the GPU worker until job completes or errors.
    Returns the /job_status JSON (including result if success).
    """

    worker_url = worker_url.rstrip("/")
    deadline = time.time() + max_wait_seconds

    last_status = ""

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{worker_url}/job_status",
                params={"job_id": job_id},
                timeout=30,
            )
            resp.raise_for_status()
            js = resp.json()
        except Exception as e:
            raise GPUWorkerError(f"Polling failed: {e}")

        status = js.get("status", "unknown")

        if status != last_status:
            last_status = status

        # ---- Finished successfully ----
        if status == "done":
            return js

        # ---- Error ----
        if status == "error":
            err_msg = js.get("error", "Unknown error")
            raise GPUWorkerError(f"GPU worker error: {err_msg}")

        time.sleep(poll_interval)

    raise GPUWorkerError("GPU job did not finish before timeout.")


# ----------------------------------------------------------
# Download ZIP output
# ----------------------------------------------------------
def download_gpu_zip(
    worker_url: str,
    job_id: str,
    timeout: int = 300
) -> bytes:
    """
    Downloads the processed job ZIP bytes.
    """
    worker_url = worker_url.rstrip("/")
    try:
        resp = requests.get(
            f"{worker_url}/job_zip",
            params={"job_id": job_id},
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        raise GPUWorkerError(f"Failed to download GPU ZIP: {e}")

    return resp.content


# ----------------------------------------------------------
# Convenience: run everything (start → poll → download)
# ----------------------------------------------------------
def fetch_full_result(
    worker_url: str,
    file_bytes: bytes,
    filename: str,
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience helper:
        1. start job
        2. poll until done
        3. download zip
        4. return { job_id, result_json, zip_bytes }
    """

    job_id = start_gpu_job(worker_url, file_bytes, filename, settings)

    status_json = poll_gpu_status(worker_url, job_id)

    zip_bytes = download_gpu_zip(worker_url, job_id)

    return {
        "job_id": job_id,
        "result_json": status_json.get("result", {}),
        "zip_bytes": zip_bytes,
    }
