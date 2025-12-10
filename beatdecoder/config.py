# beatdecoder/config.py
from pathlib import Path

# ---------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------
WORKSPACE_DIR = Path("workspace")
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
JOBS_DIR = WORKSPACE_DIR / "jobs"
TMP_DIR = WORKSPACE_DIR / "tmp"

# Ensure directories exist
for d in [WORKSPACE_DIR, UPLOADS_DIR, JOBS_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------
DEFAULT_DEMUCS_MODEL = "htdemucs"
DEFAULT_STEMS_FOR_MIDI = ["bass", "other"]
DEFAULT_DEMUCS_DEVICE = "auto"

# Local development worker (FastAPI)
LOCAL_WORKER_URL = "http://localhost:8000"

# ---------------------------------------------------------
# UI defaults
# ---------------------------------------------------------
APP_TITLE = "🎧 BeatDecoder – AI Stem Splitter + MIDI Rebuilder"
APP_CAPTION = (
    "Demucs + BPM/Key + chords/sections + Serum-style analysis.\n"
    "Supports Local Processing or Remote GPU Worker."
)

# Chord segmentation parameters
MAX_ANALYSIS_DURATION = 240.0  # seconds for chord analysis

# Logging
MAX_LOG_LINES = 5000

# ---------------------------------------------------------
# History / Library
# ---------------------------------------------------------
INDEX_FILE = WORKSPACE_DIR / "index.json"

# ---------------------------------------------------------
# Stem player styling
# ---------------------------------------------------------
STEM_PLAYER_HEIGHT_BASE = 260
STEM_PLAYER_HEIGHT_PER_STEM = 24

