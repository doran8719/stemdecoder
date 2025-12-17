# TRIPWIRE_DEPLOY_9F3C2
import os
import io
import json
import time
import base64
import zipfile
import hashlib
import shutil
import subprocess
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
import os
import traceback
if "last_job" not in st.session_state:
    st.session_state.last_job = None


# =========================
# Streamlit config
# =========================
st.set_page_config(
    page_title="Decode Audio — Stem Decoder",
    page_icon="🎧",
    layout="wide",
)

HERE = Path(__file__).resolve().parent
FRONTEND_DIR = HERE / "frontend"
PLAYER_HTML = FRONTEND_DIR / "player.html"
PLAYER_CSS = FRONTEND_DIR / "player.css"
PLAYER_JS = FRONTEND_DIR / "player.js"

WORKSPACE = Path(os.environ.get("BEATDECODER_WORKSPACE_DIR", str(HERE))).expanduser()
JOBS_DIR = Path(os.environ.get("BEATDECODER_JOBS_DIR", str(WORKSPACE / "jobs"))).expanduser()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

INLINE_PREVIEW_MB = float(os.environ.get("BEATDECODER_INLINE_AUDIO_MB", "50"))
INLINE_PREVIEW_BYTES = int(INLINE_PREVIEW_MB * 1024 * 1024)


# =========================
# Decode UI shell (NO global span/label nuking)
# =========================
st.markdown(
    """
    <style>
      :root{ color-scheme: light; }
      html, body, .stApp { background:#f5f6f8 !important; }
      header[data-testid="stHeader"]{display:none;}
      footer{display:none;}
      #MainMenu{visibility:hidden;}

      section.main > div.block-container{
        max-width: 1120px !important;
        padding-top: 18px !important;
        padding-bottom: 56px !important;
      }

      /* Buttons */
      .stButton>button, .stDownloadButton>button{
        border-radius: 12px !important;
        font-weight: 900 !important;
        border: 1px solid rgba(0,0,0,.10) !important;
        background: rgba(255,255,255,.95) !important;
        color: #111111 !important;
        box-shadow: 0 1px 0 rgba(0,0,0,.04);
      }
      .stButton>button:hover, .stDownloadButton>button:hover{
        border-color: rgba(0,0,0,.16) !important;
      }
      .stButton>button:disabled{
        opacity: .55 !important;
        cursor: not-allowed !important;
      }

      /* Primary button (Start Processing) */
      .stButton>button[kind="primary"]{
        background: #ff4b4b !important;
        color: white !important;
        border: 1px solid rgba(0,0,0,.06) !important;
      }

      /* Panels */
      .decode-panel{
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 14px;
        backdrop-filter: blur(10px);
      }

      .decode-hr{
        height:1px;
        background: rgba(0,0,0,.08);
        margin: 14px 0 10px;
      }

      /* Text defaults (safe) */
      .stMarkdown, .stMarkdown p{
        color: #111111 !important;
      }
      .stCaption, .stCaption p{
        color: rgba(0,0,0,.55) !important;
        font-weight: 650;
      }

      /* File uploader container */
      div[data-testid="stFileUploader"]{
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 14px 14px 10px;
        backdrop-filter: blur(10px);
      }

      /* Fix uploader label + selected filename/size */
      div[data-testid="stFileUploader"] label{
        color:#111111 !important;
        font-weight: 850 !important;
      }
      div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
      div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"]{
        color:#111111 !important;
        opacity: 1 !important;
        font-weight: 750 !important;
      }

      /* IMPORTANT: the dark dropzone bar text must stay light */
      div[data-testid="stFileUploaderDropzone"]{
        border-radius: 14px !important;
        overflow: hidden;
      }
      div[data-testid="stFileUploaderDropzone"] *{
        color: rgba(255,255,255,.92) !important;
      }
      /* But keep the "Browse files" button readable */
      div[data-testid="stFileUploaderDropzone"] button,
      div[data-testid="stFileUploaderDropzone"] button *{
        color: #111111 !important;
      }

      /* Progress text */
      .decode-progress-text{
        margin-top: 10px;
        font-weight: 900;
        color:#111111;
      }

      /* Decoded Songs card */
      .decoded-card{
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 14px;
        backdrop-filter: blur(10px);
        box-shadow: 0 14px 40px rgba(0,0,0,.08);
      }
      .decoded-head{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
        margin-bottom: 10px;
      }
      .decoded-title{
        font-weight: 950;
        font-size: 18px;
        color:#111111;
      }
      .decoded-subtle{
        font-weight: 850;
        font-size: 12px;
        color: rgba(0,0,0,.55);
      }
      .decoded-table-head{
        padding: 10px 6px 10px;
        border-bottom: 1px solid rgba(0,0,0,.08);
        margin-bottom: 6px;
        color: rgba(0,0,0,.55);
        font-weight: 900;
        font-size: 12px;
      }
      .decoded-row{
        padding: 12px 6px;
        border-bottom: 1px solid rgba(0,0,0,.06);
      }
      .decoded-song{
        font-weight: 950;
        color:#111111;
      }
      .decoded-pill{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,.12);
        background: rgba(255,255,255,.70);
        font-weight: 900;
        font-size: 12px;
        color:#111111;
        min-width: 42px;
      }

      /* -----------------------------
         DECODED SONGS TABLE (override)
         ----------------------------- */
      .decoded-card{
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 14px;
        backdrop-filter: blur(10px);
        margin-top: 30px;
      }
      .decoded-head{
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap: 12px;
        padding: 4px 2px 10px;
      }
      .decoded-title{
        font-weight: 950;
        font-size: 16px;
        color: rgba(0,0,0,.92);
      }
      .decoded-subtle{
        font-weight: 850;
        font-size: 12px;
        color: rgba(0,0,0,.55);
      }

      table.decoded-table{
        width: 100%;
        border-collapse: collapse;
      }
      table.decoded-table thead th{
        text-align: left;
        font-size: 12px;
        color: rgba(0,0,0,.55);
        font-weight: 850;
        padding: 12px 8px;
        border-bottom: 1px solid rgba(0,0,0,.08);
      }
      table.decoded-table tbody td{
        padding: 18px 8px;
        border-bottom: 1px solid rgba(0,0,0,.08);
        vertical-align: middle;
      }
      table.decoded-table tbody tr:last-child td{
        border-bottom: none;
      }
      table.decoded-table th:nth-child(2),
      table.decoded-table td:nth-child(2),
      table.decoded-table th:nth-child(3),
      table.decoded-table td:nth-child(3){
        text-align: center;
        width: 120px;
      }
      table.decoded-table .decoded-action,
      table.decoded-table .decoded-action-h{
        text-align: right !important;
        width: 170px;
        white-space: nowrap;
      }

      .decoded-song{
        font-weight: 950;
        color: rgba(0,0,0,.92);
      }
      .decoded-pill{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,.12);
        background: rgba(255,255,255,.85);
        font-weight: 900;
        font-size: 12px;
        color: rgba(0,0,0,.85);
      }

      .decoded-btn{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,.10);
        background: rgba(255,255,255,.92);
        box-shadow: 0 1px 0 rgba(0,0,0,.04);
        font-weight: 900;
        cursor: pointer;
      }
      .decoded-btn:hover{
        transform: translateY(-1px);
      }
      .decoded-btn:disabled{
        opacity: .6;
        cursor: default;
        transform: none;
      }

      /* Remove stray horizontal rules / separators that may appear */
      hr, .stHorizontalBlock hr, .stDivider, div[data-testid="stDecoration"]{
        display:none !important;
      }

    
      /* Decoded Songs (Streamlit rows) */
      #decoded-songs { margin-top: 0px; }
      .decoded-th{
        font-size: 12px;
        font-weight: 850;
        color: rgba(0,0,0,.55);
        padding: 6px 0 10px;
      }
      .decoded-line{
        height:1px;
        background: rgba(0,0,0,.08);
        margin: 6px 0;
      }
      /* Make the action buttons match the player buttons */
      div[data-testid="column"] .stButton>button{
        border-radius: 12px !important;
        font-weight: 900 !important;
      }
      /* Narrow scope: only inside Decoded Songs area (buttons we create in row cols) */
      #decoded-songs ~ div .stButton>button{
        background: rgba(255,255,255,.95) !important;
        color: rgba(0,0,0,.88) !important;
        border: 1px solid rgba(0,0,0,.12) !important;
        box-shadow: 0 1px 0 rgba(0,0,0,.02);
        padding: 7px 12px !important;
      }
      #decoded-songs ~ div .stButton>button:hover{
        background: rgba(245,246,248,.95) !important;
        border-color: rgba(0,0,0,.16) !important;
      }
      #decoded-songs ~ div .stButton>button:disabled{
        opacity: .55 !important;
        cursor: default !important;
      }
</style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helpers: ports + servers
# =========================
def _pick_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_fileserver(root_dir: Path) -> int:
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def log_message(self, format, *args):
            return

    port = _pick_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return port


def _ffmpeg_ok() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def _safe_job_dir(job_name: str) -> Path:
    job = (job_name or "").replace("\\", "/").split("/")[-1]
    p = (JOBS_DIR / job).resolve()
    if not str(p).startswith(str(JOBS_DIR.resolve())):
        raise ValueError("Invalid job path.")
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError("Job not found.")
    return p


def _find_stems(job_dir: Path) -> List[Path]:
    stems_dir = job_dir / "stems"
    if stems_dir.exists() and stems_dir.is_dir():
        out = list(stems_dir.glob("*.wav"))
        out.sort()
        return out
    return []


def _build_zip_from_paths(zip_path: Path, paths: List[Path]) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=p.name)
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def _build_stems_zip(job_dir: Path, stems: List[Path]) -> Path:
    exports_dir = job_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    zip_path = exports_dir / "stems.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    return _build_zip_from_paths(zip_path, stems)


def _export_selection_zip(job_dir: Path, stems: List[Path], start_s: float, end_s: float) -> Path:
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found. Install it: brew install ffmpeg")

    start_s = float(max(0.0, start_s))
    end_s = float(max(0.0, end_s))
    if end_s <= start_s + 0.03:
        raise ValueError("Selection too small.")

    exports_dir = job_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    stamp = f"{int(start_s*1000)}_{int(end_s*1000)}"
    clips_dir = exports_dir / f"selection_{stamp}_clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir, ignore_errors=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        out_wav = clips_dir / stem.name
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-to", f"{end_s:.3f}",
            "-i", str(stem),
            "-acodec", "pcm_s16le",
            str(out_wav),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg trim failed for {stem.name}: {p.stderr[-300:]}")

    zip_path = exports_dir / f"selection_{stamp}.zip"
    return _build_zip_from_paths(zip_path, sorted(clips_dir.glob("*.wav")))


def _generate_midi_basic_pitch(job_dir: Path, stems: List[Path]) -> Path:
    try:
        from basic_pitch.inference import predict_and_save  # type: ignore
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
        except Exception:
            ICASSP_2022_MODEL_PATH = None  # type: ignore
    except Exception as e:
        raise RuntimeError("BasicPitch not installed. Install: pip install basic-pitch") from e

    exports_dir = job_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    src = None
    for s in stems:
        if "voc" in s.stem.lower():
            src = s
            break
    if src is None and stems:
        src = stems[0]
    if src is None:
        raise RuntimeError("No stems available for MIDI generation.")

    out_dir = exports_dir / "basicpitch_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        audio_path_list=[str(src)],
        output_directory=str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
    )

    try:
        if ICASSP_2022_MODEL_PATH:
            kwargs["model_or_model_path"] = ICASSP_2022_MODEL_PATH
        predict_and_save(**kwargs)
    except TypeError:
        kwargs.pop("model_or_model_path", None)
        predict_and_save(**kwargs)

    mids = list(out_dir.glob("*.mid")) + list(out_dir.glob("*.midi"))
    if not mids:
        raise RuntimeError("BasicPitch ran but no MIDI was created.")

    midi_out = exports_dir / "midi.mid"
    shutil.copyfile(mids[0], midi_out)
    return midi_out


def _custom_stems_zip(job_dir: Path, stems: List[Path], filenames: List[str], fmt: str) -> Path:
    exports_dir = job_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    fmt = (fmt or "wav").lower()
    if fmt not in ("wav", "mp3"):
        fmt = "wav"

    chosen = []
    stem_by_name = {p.name: p for p in stems}
    for fn in filenames:
        if fn in stem_by_name:
            chosen.append(stem_by_name[fn])

    if not chosen:
        raise ValueError("No stems selected.")

    stamp = hashlib.sha1((fmt + "|" + "|".join(sorted([p.name for p in chosen]))).encode("utf-8")).hexdigest()[:10]
    zip_path = exports_dir / f"stems_custom_{stamp}_{fmt}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path

    if fmt == "wav":
        return _build_zip_from_paths(zip_path, chosen)

    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg not found. Install it: brew install ffmpeg")

    tmp_dir = exports_dir / f"_tmp_mp3_{stamp}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mp3s: List[Path] = []
    for wav in chosen:
        out = tmp_dir / (wav.stem + ".mp3")
        cmd = ["ffmpeg", "-y", "-i", str(wav), "-b:a", "192k", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg mp3 failed for {wav.name}: {p.stderr[-300:]}")
        mp3s.append(out)

    return _build_zip_from_paths(zip_path, sorted(mp3s))


def _start_api_server(fileserver_base: str) -> int:
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse

    class API(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_json(self, code: int, obj: Dict[str, Any]):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                return self._send_json(200, {"ok": True})
            return self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                return self._send_json(400, {"ok": False, "error": "bad_json"})

            try:
                if path == "/export_selection":
                    job = payload.get("job") or ""
                    start_s = float(payload.get("start_s"))
                    end_s = float(payload.get("end_s"))
                    job_dir = _safe_job_dir(job)
                    stems = _find_stems(job_dir)
                    if not stems:
                        raise RuntimeError("No stems found.")
                    zip_path = _export_selection_zip(job_dir, stems, start_s, end_s)
                    url = f"{fileserver_base}/{job_dir.name}/exports/{zip_path.name}"
                    return self._send_json(200, {"ok": True, "zip_url": url})

                if path == "/build_midi":
                    job = payload.get("job") or ""
                    job_dir = _safe_job_dir(job)
                    stems = _find_stems(job_dir)
                    midi = _generate_midi_basic_pitch(job_dir, stems)
                    url = f"{fileserver_base}/{job_dir.name}/exports/{midi.name}"
                    return self._send_json(200, {"ok": True, "midi_url": url})

                if path == "/download_stems_custom":
                    job = payload.get("job") or ""
                    fmt = payload.get("format") or "wav"
                    files = payload.get("files") or []
                    if not isinstance(files, list):
                        raise ValueError("files must be a list")
                    files = [str(x) for x in files]
                    job_dir = _safe_job_dir(job)
                    stems = _find_stems(job_dir)
                    zip_path = _custom_stems_zip(job_dir, stems, files, fmt)
                    url = f"{fileserver_base}/{job_dir.name}/exports/{zip_path.name}"
                    return self._send_json(200, {"ok": True, "zip_url": url})

                return self._send_json(404, {"ok": False, "error": "not_found"})
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})

    port = _pick_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), API)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return port


# Start servers once
if "fileserver_port" not in st.session_state:
    st.session_state["fileserver_port"] = _start_fileserver(JOBS_DIR)
FILESERVER_PORT = int(st.session_state["fileserver_port"])
FILESERVER_BASE = f"http://127.0.0.1:{FILESERVER_PORT}"

if "api_port" not in st.session_state:
    st.session_state["api_port"] = _start_api_server(FILESERVER_BASE)
API_PORT = int(st.session_state["api_port"])
API_BASE = f"http://127.0.0.1:{API_PORT}"


# =========================
# Frontend loader
# =========================
def _nonce() -> str:
    parts = []
    for p in (PLAYER_HTML, PLAYER_CSS, PLAYER_JS):
        try:
            parts.append(str(p.stat().st_mtime_ns))
        except Exception:
            parts.append("0")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_frontend_html(data: Dict[str, Any]) -> str:
    if not PLAYER_HTML.exists():
        st.error("Missing frontend/player.html next to app.py")
        st.stop()

    html_text = _read_text(PLAYER_HTML)
    css_text = _read_text(PLAYER_CSS)
    js_text = _read_text(PLAYER_JS)

    html_text = html_text.replace(
        '<link rel="stylesheet" href="player.css"/>',
        "<style>\n" + css_text + "\n</style>",
    )
    html_text = html_text.replace(
        '<script src="player.js"></script>',
        "<script>\n" + js_text + "\n</script>",
    )

    blob = json.dumps(data, ensure_ascii=False)
    html_text = html_text.replace("__DATA_JSON__", blob)
    html_text = html_text.replace("</body>", f"<!-- nonce:{_nonce()} -->\n</body>")
    return html_text


# =========================
# Job UI helpers
# =========================
def _list_jobs() -> List[Path]:
    jobs = [p for p in JOBS_DIR.iterdir() if p.is_dir()]
    jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jobs


def _read_meta(job_dir: Path) -> Dict[str, Any]:
    p = job_dir / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_meta(job_dir: Path, meta: Dict[str, Any]) -> None:
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _label_from_filename(p: Path) -> str:
    n = p.stem.lower()
    if "voc" in n:
        return "Vocals"
    if "drum" in n:
        return "Drums"
    if "bass" in n:
        return "Bass"
    if "other" in n or "inst" in n:
        return "Instrumental"
    return p.stem


def _stem_order_key(label: str) -> int:
    # Player order: Vocals, Instrumental, Drums, Bass (as you requested earlier)
    order = {"Vocals": 0, "Instrumental": 1, "Drums": 2, "Bass": 3}
    return order.get(label, 99)


def _ensure_preview_mp3(stem_wav: Path, previews_dir: Path, max_bytes: int) -> Optional[Path]:
    if not _ffmpeg_ok():
        return None

    previews_dir.mkdir(parents=True, exist_ok=True)

    bitrate_candidates = [64, 48, 32, 24]
    for br in bitrate_candidates:
        out = previews_dir / f"{stem_wav.stem}.full_mono_{br}k.mp3"
        if out.exists() and out.stat().st_size > 0 and out.stat().st_size <= max_bytes:
            return out

        if out.exists():
            try:
                out.unlink()
            except Exception:
                pass

        cmd = ["ffmpeg", "-y", "-i", str(stem_wav), "-ac", "1", "-b:a", f"{br}k", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and out.exists() and out.stat().st_size > 0 and out.stat().st_size <= max_bytes:
            return out

    out = previews_dir / f"{stem_wav.stem}.clip120_mono_24k.mp3"
    if out.exists():
        try:
            out.unlink()
        except Exception:
            pass
    cmd = ["ffmpeg", "-y", "-i", str(stem_wav), "-t", "120", "-ac", "1", "-b:a", "24k", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0 and out.stat().st_size <= max_bytes:
        return out

    return None



def _waveform_cache_path(previews_dir: Path, stem_wav: Path, n_bars: int) -> Path:
    previews_dir.mkdir(parents=True, exist_ok=True)
    return previews_dir / f"{stem_wav.stem}.bars_{n_bars}.json"


def _compute_waveform_bars(stem_wav: Path, previews_dir: Path, n_bars: int = 240) -> List[float]:
    """
    Compute a compact waveform representation (RMS envelope) for rendering bar-style waveforms.

    Returns a list of floats in [0..1]. Cached to previews_dir for speed.
    """
    cache_path = _waveform_cache_path(previews_dir, stem_wav, n_bars)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [float(x) for x in data][:n_bars]
        except Exception:
            pass

    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception:
        # If deps aren't installed, gracefully fall back to placeholder waveforms in the frontend.
        return []

    try:
        y, sr = sf.read(str(stem_wav), dtype="float32", always_2d=True)
        if y.size == 0:
            return []
        y = y.mean(axis=1)  # mono
        y = np.abs(y)

        # Protect memory/time for very long files by downsampling.
        max_samples = 2_000_000  # ~250s at 8kHz equivalent
        if y.size > max_samples:
            step = int(np.ceil(y.size / max_samples))
            y = y[::step]

        # Chunk into bars and compute RMS per chunk.
        chunk = int(np.ceil(y.size / float(n_bars)))
        bars: List[float] = []
        for i in range(n_bars):
            s = i * chunk
            e = min((i + 1) * chunk, y.size)
            if s >= y.size:
                break
            seg = y[s:e]
            if seg.size == 0:
                bars.append(0.0)
            else:
                bars.append(float(np.sqrt(np.mean(seg * seg))))

        if not bars:
            return []

        mx = max(bars) or 1.0
        bars = [b / mx for b in bars]

        # Slight gamma curve so quieter parts are still visible
        bars = [float(b ** 0.7) for b in bars]

        try:
            cache_path.write_text(json.dumps(bars), encoding="utf-8")
        except Exception:
            pass

        return bars
    except Exception:
        return []


def _data_uri_audio_mp3(mp3_path: Path) -> str:
    b = mp3_path.read_bytes()
    return "data:audio/mpeg;base64," + base64.b64encode(b).decode("utf-8")


# =========================
# BPM/Key analysis (best effort)
# =========================
def _estimate_bpm_and_key(audio_path: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Best-effort BPM + key detection.
    Uses librosa if available. If librosa isn't installed, returns (None, None).
    """
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
    except Exception:
        return (None, None)

    try:
        y, sr = librosa.load(str(audio_path), mono=True, sr=None)
        if y is None or len(y) < 2048:
            return (None, None)

        # --- BPM ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_int = int(round(float(tempo))) if tempo and tempo > 0 else None

        # --- Key (simple Krumhansl-ish via chroma mean) ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        # major/minor profiles (Krumhansl)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        def _corr(a, b):
            a = (a - a.mean()) / (a.std() + 1e-9)
            b = (b - b.mean()) / (b.std() + 1e-9)
            return float((a * b).mean())

        notes = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        best = None
        best_score = -1e9
        best_mode = "major"

        for i in range(12):
            maj = np.roll(major_profile, i)
            mi = np.roll(minor_profile, i)
            smaj = _corr(chroma_mean, maj)
            smi = _corr(chroma_mean, mi)
            if smaj > best_score:
                best_score = smaj
                best = notes[i]
                best_mode = "major"
            if smi > best_score:
                best_score = smi
                best = notes[i]
                best_mode = "minor"

        key_str = f"{best} {best_mode}" if best else None
        return (bpm_int, key_str)
    except Exception:
        return (None, None)


# =========================
# Demucs runner (progress bar only)
# =========================
def _run_demucs_to_job(input_path: Path, job_dir: Path) -> None:
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", "-u", "-m", "demucs", "-o", str(stems_dir), str(input_path)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    prog = st.progress(0)
    prog_text = st.markdown('<div class="decode-progress-text">Decoding… 0%</div>', unsafe_allow_html=True)
    last_pct = -1

    if p.stdout:
        for line in p.stdout:
            m = re.search(r"(\d{1,3})%\|", line)
            if not m:
                m = re.search(r"^\s*(\d{1,3})%\s*", line)
            if m:
                pct = max(0, min(100, int(m.group(1))))
                if pct != last_pct:
                    prog.progress(pct / 100.0)
                    prog_text = st.markdown(
                        f'<div class="decode-progress-text">Decoding… {pct}%</div>',
                        unsafe_allow_html=True,
                    )
                    last_pct = pct

    rc = p.wait()

    try:
        prog.progress(1.0)
    except Exception:
        pass

    if rc != 0:
        raise RuntimeError(f"Demucs exited with code {rc}")

    produced = list(stems_dir.rglob("*.wav"))
    if not produced:
        raise RuntimeError("No stems were produced.")

    # Flatten if Demucs nested output
    top = list(stems_dir.glob("*.wav"))
    if not top:
        for f in produced:
            dest = stems_dir / f.name
            if not dest.exists():
                try:
                    f.replace(dest)
                except Exception:
                    pass


# =========================
# Upload / Process (NO extra empty container)
# =========================

upload = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav", "flac", "m4a"],
    accept_multiple_files=False,
    label_visibility="visible",
)

if upload is not None:
    st.caption("Ready to process with Demucs.")
    if st.button("Start Processing", type="primary"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = Path(upload.name).stem.replace(" ", "_")
        job_dir = JOBS_DIR / f"{ts}_{safe_name}"
        job_dir.mkdir(parents=True, exist_ok=True)

        in_path = job_dir / upload.name
        with open(in_path, "wb") as f:
            f.write(upload.getbuffer())

        meta = {"title": upload.name, "created": ts}
        _write_meta(job_dir, meta)

        try:
            _run_demucs_to_job(in_path, job_dir)
        st.write("DEBUG job_dir:", str(job_dir))
        st.write("DEBUG job_dir exists:", job_dir.exists())
        st.write("DEBUG files:", [p.name for p in job_dir.rglob("*")][:200])
        
            # Analyze BPM/Key after stems complete (best effort)
            bpm, key = _estimate_bpm_and_key(in_path)
            meta2 = _read_meta(job_dir)
            if bpm is not None:
                meta2["bpm"] = bpm
                meta2["tempo"] = bpm
            if key:
                meta2["key"] = key
            _write_meta(job_dir, meta2)

            st.success("Done! Loading…")
            st.query_params["job"] = job_dir.name
            st.rerun()

        except Exception:
            st.error("❌ Processing failed on the server. Full error:")
            st.code(traceback.format_exc())
            st.stop()

# remove div wrapper and separator


# =========================
# Active job + player
# =========================
jobs = _list_jobs()
if not jobs:
    st.info("No jobs yet. Upload a song and click Start Processing.")
    st.stop()

active_job_id = st.query_params.get("job", jobs[0].name)
active_job_dir = next((p for p in jobs if p.name == active_job_id), jobs[0])

meta = _read_meta(active_job_dir)
title = meta.get("title") or active_job_dir.name

stems = _find_stems(active_job_dir)
if not stems:
    st.warning("This job has no stems yet.")
    st.stop()

# Order stems in the player: Vocals, Instrumental, Drums, Bass
stems_sorted = sorted(stems, key=lambda p: _stem_order_key(_label_from_filename(p)))

zip_path = _build_stems_zip(active_job_dir, stems_sorted)
zip_url = f"{FILESERVER_BASE}/{active_job_dir.name}/exports/{zip_path.name}"

midi_path = active_job_dir / "exports" / "midi.mid"
midi_url = f"{FILESERVER_BASE}/{active_job_dir.name}/exports/midi.mid" if midi_path.exists() else ""

previews_dir = active_job_dir / "previews"
stems_payload = []
for wav in stems_sorted:
    label = _label_from_filename(wav)
    mp3 = _ensure_preview_mp3(wav, previews_dir, INLINE_PREVIEW_BYTES)
    audio_uri = _data_uri_audio_mp3(mp3) if mp3 else ""
    bars = _compute_waveform_bars(wav, previews_dir, n_bars=240)
    stems_payload.append(
        {
            "label": label,
            "file": wav.name,
            "audio_data_uri": audio_uri,
            "bars": bars,
        }
    )

data = {
    "brand": "Decode Audio",
    "active_job": active_job_dir.name,
    "api_base": API_BASE,
    "track": {
        "title": title,
        "bpm": meta.get("bpm") or meta.get("tempo") or "—",
        "key": meta.get("key") or "—",
    },
    "downloads": {
        "zip_url": zip_url,
        "midi_url": midi_url,
    },
    "stems": stems_payload,
}

components.html(_load_frontend_html(data), height=780, scrolling=False)

# Tight spacing between player and decoded songs
# =========================
# Decoded Songs list (below player)
# =========================
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

jobs_for_list = _list_jobs()
if jobs_for_list:
    # active_job_id is already set above for the player, but keep this safe:
    active_job_id = st.query_params.get("job", jobs_for_list[0].name)

    # Header + container
    st.markdown(
        f"""
        <div id="decoded-songs" class="decoded-card">
          <div class="decoded-head">
            <div class="decoded-title">Decoded Songs</div>
            <div class="decoded-subtle">{min(len(jobs_for_list), 50)} shown</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render header row aligned to our column layout
    header_cols = st.columns([0.58, 0.14, 0.14, 0.14], vertical_alignment="center")
    header_cols[0].markdown("<div class='decoded-th'>Song</div>", unsafe_allow_html=True)
    header_cols[1].markdown("<div class='decoded-th'>BPM</div>", unsafe_allow_html=True)
    header_cols[2].markdown("<div class='decoded-th'>Key</div>", unsafe_allow_html=True)
    header_cols[3].markdown("<div class='decoded-th' style='text-align:right;'>&nbsp;</div>", unsafe_allow_html=True)
    st.markdown("<div class='decoded-line'></div>", unsafe_allow_html=True)

    # Rows
    for j in jobs_for_list[:50]:
        m = _read_meta(j)
        song = (m.get("title") or j.name)
        bpm = (m.get("bpm") or m.get("tempo") or "—")
        key = (m.get("key") or "—")

        row_cols = st.columns([0.58, 0.14, 0.14, 0.14], vertical_alignment="center")
        row_cols[0].markdown(f"<div class='decoded-song'>{song}</div>", unsafe_allow_html=True)
        row_cols[1].markdown(f"<span class='decoded-pill'>{bpm}</span>", unsafe_allow_html=True)
        row_cols[2].markdown(f"<span class='decoded-pill'>{key}</span>", unsafe_allow_html=True)

        if j.name == active_job_id:
            row_cols[3].button("Loaded", key=f"load_{j.name}", disabled=True)
        else:
            if row_cols[3].button("Load Song", key=f"load_{j.name}"):
                st.query_params["job"] = j.name
                st.rerun()

        st.markdown("<div class='decoded-line'></div>", unsafe_allow_html=True)
