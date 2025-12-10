import json
from pathlib import Path
from typing import Dict, Any, List

from file_utils import (
    list_job_dirs,
    load_metadata,
)

INDEX_PATH = Path("workspace/index.json")


# -------------------------------------------------------------------
# Save + load index.json
# -------------------------------------------------------------------
def load_index() -> List[Dict[str, Any]]:
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def write_index(entries: List[Dict[str, Any]]):
    try:
        with open(INDEX_PATH, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass


# -------------------------------------------------------------------
# Build index.json by scanning workspace/jobs
# -------------------------------------------------------------------
def build_index_from_jobs() -> List[Dict[str, Any]]:
    jobs = list_job_dirs()
    index_entries = []

    for job_dir in jobs:
        meta = load_metadata(job_dir)

        entry = {
            "job_id": meta.get("job_id", job_dir.name),
            "job_path": str(job_dir.resolve()),
            "song_name": meta.get("song_name", job_dir.name),
            "job_label": meta.get("job_label", job_dir.name),
            "bpm": meta.get("bpm"),
            "key": meta.get("key"),
            "mode": meta.get("mode"),
            "created_at": meta.get("created_at"),

            # extra fields
            "key_confidence": meta.get("key_confidence"),
        }

        index_entries.append(entry)

    write_index(index_entries)
    return index_entries


# -------------------------------------------------------------------
# Search helpers for the Library tab
# -------------------------------------------------------------------
def search_index(
    text: str = "",
    bpm_min: float = 0,
    bpm_max: float = 400,
    key_choice: str = "Any",
) -> List[Dict[str, Any]]:
    
    index = load_index()
    text = text.lower().strip()

    results = []
    for e in index:

        # TEXT FILTER
        if text:
            hay = f"{e.get('song_name','')} {e.get('job_label','')}".lower()
            if text not in hay:
                continue

        # BPM FILTER
        bpm = e.get("bpm")
        if isinstance(bpm, (int, float)):
            if bpm < bpm_min or bpm > bpm_max:
                continue

        # KEY FILTER
        if key_choice != "Any":
            if e.get("key") != key_choice:
                continue

        results.append(e)

    # newest first
    results = sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)
    return results
