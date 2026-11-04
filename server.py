#!/usr/bin/env python3
"""
server.py - web front end for video_analyzer.py

video_analyzer.py is NOT modified. Each upload is analysed by running it exactly as
you would on the command line, so the HTML report is byte-for-byte what the script
writes. The server only:
  - accepts the video and options from the browser,
  - runs the script and turns its progress messages into a progress bar,
  - serves the finished report file as it is.

Run:
    pip install flask openai opencv-python
    export MODEL_API_KEY=your_meta_model_api_key     # or put MODEL_API_KEY=... in a .env file next to this script
    python server.py
Then open http://localhost:5000
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "video_analyzer.py"
JOBS_DIR = ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Optional: read MODEL_API_KEY from a .env file next to this script (never sent to the browser)
_env_file = ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _k, _, _v = _line.partition("=")
        if _k.strip() and not _k.strip().startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

JOBS = {}  # job_id -> dict
LOCK = threading.Lock()

# The steps the script prints to stderr, in order. The front end shows them as a checklist.
STEPS = [
    ("frames", "Reading the video"),
    ("describe", "Describing what happens"),
    ("analyze", "Finding movement problems"),
    ("verify", "Checking each finding up close"),
    ("equipment", "Looking at the gear"),
    ("report", "Writing the report"),
]


def _step_from_line(line):
    if line.startswith("Video length"):
        return "frames"
    if line.startswith("Pass 1/2"):
        return "describe"
    if line.startswith("Pass 2/2"):
        return "analyze"
    if line.startswith("Checking finding"):
        return "verify"
    if line.startswith("Equipment check"):
        return "equipment"
    if line.startswith("Report saved"):
        return "report"
    return None


def build_command(video_name, report_name):
    """Runs the analyzer with its default settings, exactly as `python video_analyzer.py clip.mp4`."""
    return [sys.executable, "-u", str(ANALYZER), video_name, "--out", report_name]


def run_job(job_id, cmd, env):
    job = JOBS[job_id]
    job_dir = JOBS_DIR / job_id
    try:
        proc = subprocess.Popen(
            cmd, cwd=job_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        job["proc"] = proc
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            with LOCK:
                job["log"].append(line)
                step = _step_from_line(line)
                if step:
                    job["step"] = step
                m = re.match(r"Checking finding (\d+)/(\d+)", line)
                if m:
                    job["detail"] = f"Finding {m.group(1)} of {m.group(2)}"
                elif step:
                    job["detail"] = ""
        code = proc.wait()
        report = job_dir / job["report_name"]
        with LOCK:
            if job["status"] == "cancelled":
                pass
            elif code == 0 and report.is_file():
                job["status"] = "done"
                job["step"] = "report"
            else:
                job["status"] = "error"
                job["error"] = next(
                    (l for l in reversed(job["log"]) if l.strip()), "The analysis stopped without a report."
                )
    except Exception as exc:  # noqa: BLE001
        with LOCK:
            job["status"] = "error"
            job["error"] = str(exc)
    finally:
        job["finished"] = time.time()
        job.pop("proc", None)
        meta = {k: job.get(k, "") for k in ("id", "video", "status", "created", "finished", "report_name", "error")}
        (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def public(job):
    return {
        "id": job["id"],
        "video": job["video"],
        "status": job["status"],
        "step": job["step"],
        "detail": job.get("detail", ""),
        "steps": [{"key": k, "label": v} for k, v in STEPS],
        "log": job["log"][-40:],
        "error": job.get("error", ""),
        "report_url": f"/api/jobs/{job['id']}/files/{job['report_name']}" if job["status"] == "done" else "",
        "created": job["created"],
    }


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/config")
def config():
    return jsonify({"ready": bool(os.environ.get("MODEL_API_KEY"))})


@app.post("/api/jobs")
def create_job():
    video = request.files.get("video")
    if not video or not video.filename:
        abort(400, "Choose a video file first.")
    api_key = os.environ.get("MODEL_API_KEY", "")
    if not api_key:
        abort(400, "The analysis service is not configured. Please contact the administrator.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    video_name = secure_filename(video.filename) or "video.mp4"
    video.save(job_dir / video_name)

    report_name = f"{Path(video_name).stem}_report.html"
    cmd = build_command(video_name, report_name)
    env = dict(os.environ, MODEL_API_KEY=api_key, PYTHONIOENCODING="utf-8")

    job = {
        "id": job_id, "video": video_name, "status": "running", "step": "frames", "detail": "",
        "log": [], "created": time.time(), "report_name": report_name,
    }
    with LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job_id, cmd, env), daemon=True).start()
    return jsonify(public(job)), 201


@app.get("/api/jobs")
def list_jobs():
    """Finished reports from earlier runs, newest first."""
    out = []
    for meta_file in JOBS_DIR.glob("*/meta.json"):
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("status") == "done":
            out.append({
                "id": m["id"], "video": m["video"], "created": m["created"],
                "report_url": f"/api/jobs/{m['id']}/files/{m['report_name']}",
            })
    out.sort(key=lambda m: m["created"], reverse=True)
    return jsonify(out[:20])


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    with LOCK:
        return jsonify(public(job))


@app.post("/api/jobs/<job_id>/cancel")
def cancel(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    with LOCK:
        proc = job.get("proc")
        if job["status"] == "running" and proc:
            job["status"] = "cancelled"
            proc.terminate()
        return jsonify(public(job))


@app.get("/api/jobs/<job_id>/files/<path:name>")
def job_file(job_id, name):
    """Serves the report exactly as written, plus any clips it links to (relative paths just work)."""
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(404)
    return send_from_directory(JOBS_DIR / job_id, name)


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": e.description}), 400


if __name__ == "__main__":
    if not ANALYZER.is_file():
        sys.exit("video_analyzer.py must sit next to server.py")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
