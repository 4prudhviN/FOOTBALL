#!/usr/bin/env python3
"""
video_analyzer.py - Analyze any video and produce a two-part report:

  Part 1: What is happening (scene, people, timeline, on-screen annotations)
  Part 2: Movement analysis (per-person movements, mistakes, better options,
          coaching points)

How it works: sample frames evenly with OpenCV, label each with its timestamp,
and send them to Meta's Muse Spark model (Meta Model API, OpenAI-compatible). Pass 1 describes the video; pass 2
analyzes it, using the pass 1 description plus the same frames as grounding.

Setup:
    pip install openai opencv-python
    export MODEL_API_KEY=your_meta_model_api_key   # from the Meta Model API dashboard

Usage:
    python video_analyzer.py clip.mp4
    python video_analyzer.py clip.mp4 --context "football skill tutorial"
    python video_analyzer.py clip.mp4 --start 8 --end 13 --max-frames 20   # zoom into a moment
    python video_analyzer.py clip.mp4 --pose yolo                            # measured angles for several people (pip install ultralytics)
    python video_analyzer.py clip.mp4 --clips                                # also save a short clip per finding
    python video_analyzer.py clip.mp4 --findings clip_findings.json         # re-render a saved report, no API call
    python video_analyzer.py clip.mp4 --track-only                            # tracking only: annotated video + per-player JSON timeline, no API key
    python video_analyzer.py clip.mp4 --identity                              # permanent player IDs across camera cuts (BoT-SORT stays the local tracker)
    python video_analyzer.py clip.mp4 --identity --reid-backend onnx:osnet.onnx --identity-truth truth.json
    python video_analyzer.py clip.mp4 --track-only --tracker both            # compare BoT-SORT and ByteTrack
    python video_analyzer.py clip.mp4 --dry-run                             # extract frames only, no API call

Limitations: frames are sampled (fine motion between them is missed) and
audio is not processed, so spoken commentary is not captured.
"""
import argparse
import base64
import datetime
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _k, _, _v = _line.partition("=")
        if _k.strip() and not _k.strip().startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


DEFAULT_MODEL = "muse-spark-1.1"  # also available: muse-spark-1.2
BASE_URL = "https://api.meta.ai/v1"

DESCRIBE_PROMPT = """You are given frames sampled in time order from one video, each preceded by its timestamp.
{context_line}
Write "Part 1: What is happening" with these sections:

1. Overview: what kind of video this is and its approximate length.
2. People/objects: who or what is visible (clothing, position, role). Do not identify real people by their faces.
3. Timeline: what happens, grouped into segments with timestamps (e.g. "0-12s").
4. On-screen annotations: arrows, text, overlays, and what they indicate.
5. Camera changes: angle shifts, slow motion, cutaways to other footage.

Rules:
- Describe only what is visible in the frames. If something is unclear or falls between sampled frames, say so.
- If later frames contradict an earlier reading (for example an arrow's direction), use the later evidence.
- Keep it concise and factual. State that audio was not analyzed.
- Some frames have a note such as [nothing moved since the previous frame]. If nothing moved, the picture is paused: do not describe any body or ball movement during that time, only what is shown (for example an arrow).
- At the end of the real play, say who has the ball and which way it went. Arrows may show a pretend move; follow the ball in the frames.
- Arrows, circles, lines and text drawn on the video are only drawings that explain an idea. They are not the movement. Say clearly when a move is only shown by a drawing and when it really happens, and give the time when it really starts.
- Give times in seconds (for example 14.5 s). Never say "frame 3"."""

ANALYZE_PROMPT = """Below is a description of a video, followed by the frames it was based on (each labelled with its time in seconds).

--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---
{context_line}
Analyze the whole-body movement of EVERY person, not only distances. For each person check each of these, and report only what the frames actually show: head and eyes, arms and hands, shoulders and upper body, hips, knees, the standing foot, the foot that touches the ball (which part of the foot), balance and weight shift, timing and speed, the first step after the touch, awareness of the other person.

Return ONLY valid JSON (no markdown fences, no extra text) in this shape:
{{
  "summary": "2 plain sentences",
  "people": [{{"id": "A", "label": "e.g. ball carrier (navy shirt)"}}],
  "findings": [
    {{
      "person": "A",
      "top_colour": "colour of this person's top, e.g. dark blue, black",
      "kind": "mistake | risk | good",
      "body_part": "e.g. standing foot, arms, hips, knees, timing, decision",
      "t": 14.5,
      "what_happened": "what you can see, in plain words",
      "evidence": "exactly what to look at in the frame at time t",
      "why_it_matters": "plain words",
      "fix": "the better way to move, in one plain sentence",
      "fix_steps": ["2 to 4 tiny steps, each one a single action the player can do"],
      "fix_visual": {{"type": "see the visual shapes below"}},
      "then_do": ["3 to 5 short steps: what the player must do AFTER he is in the right position, in order, until the move is finished"],
      "then_visual": {{"type": "path"}},
      "target": "a number or range if it helps (distance, angle, seconds), else empty"
    }}
  ],
  "coaching_points": [
    {{
      "title": "short plain name, e.g. Keep the ball close",
      "what_it_means": "one or two child-level sentences: what this means in real life",
      "show_the_player": ["2 to 4 short steps a coach can do to SHOW it, using cheap things like cones, shoes or a line on the ground, so the player sees right and wrong side by side"],
      "how_to_check": "one thing the coach can watch for to know the player did it right",
      "visual": {{"type": "see the visual shapes below"}}
    }}
  ],
  "better_options": [{{"name": "plain-English name", "when": "when to use it", "steps": ["step 1", "step 2"], "show_the_player": ["how a coach can demonstrate it"], "visual": {{"type": "path"}}}}],
  "caveats": ["..."]
}}

{visual_doc}

Rules:
- Every finding needs "t": one time in seconds, copied from a frame label, where the evidence is clearest. No finding without a time.
- Report mistakes for each person. If the main person made no real mistakes, include "good" findings and give better_options (a better way to do the same thing, or a better skill for the situation).
- Every finding of kind "mistake" or "risk" needs fix_steps and a fix_visual drawn from real numbers: use pose for body shape, distance for gaps, path for movement and timeline for timing. If you cannot draw the fix from real numbers, give fix_steps only.
- Every finding of kind "mistake" or "risk" also needs then_do: what the player must do once he is in the right position, step by step, until the move is finished. It must not repeat fix_steps. then_visual is optional (use path or timeline).
- Follow the ball itself in the frames, not the arrows. An arrow can show a pretend move, and the real move can go the other way. Say who really ends up with the ball and how you know.
- Only call something a mistake if the frames show it happening. A paused picture or a drawing is not a mistake.
- Never invent details. If you cannot tell from the frames, say so in caveats.
- Drawings on the video (arrows, lines, text) are not real movement. Base findings on how bodies and the ball really move, and do not report movement during a paused picture.
- Give 3 to 6 coaching_points. Each one must be something a coach can SHOW with their body or cheap equipment, not just say. Put the coaching points first in your thinking, then the better options.
- Use numbers from the frames for visuals. Do not draw a visual from a guess.
- Mark estimates with "about" and say what you measured against."""

PLAIN_STYLE = """

Audience and style (applies to EVERYTHING you write, in every part):
- Write so that a 7 to 8 year old child can understand it. If a child could not follow a sentence, rewrite it.
- Use very short sentences (about 12 words or fewer) and simple, everyday words. No sports slang, no big words.
- Say what a body part does in a simple picture-word way: "his knee is almost straight, like a stick", "bend your knees like you are sitting on a tiny chair".
- Explain every number with something a child knows. Examples: "about one big step away", "about as long as a school desk is wide", "about the width of a football". Give the number too.
- For angles, say what it looks like ("leaning a little", "bent over like picking up a coin") and give the number second.
- Describe every movement as small steps: which foot or hand, which way, and why it matters.
- Give each move a plain name first ("pulling the ball back with the bottom of the foot"). Give the official name only after it, if there is one.
- Always give times in seconds (for example 14.5 s), never frame numbers.
- Be kind. Say what went wrong gently and say how to fix it. Never mock anyone.
- Start each part with a 2-sentence summary in plain words, then give the details.
- Always say "about" with a small range for estimates, because they are guesses from video."""

DESCRIBE_PROMPT += PLAIN_STYLE
ANALYZE_PROMPT += PLAIN_STYLE


def _gray_small(img):
    return cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)


def motion_profile(path, step_s=0.25):
    """Change between neighbouring moments of the whole video (0 = identical picture)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_s = max(step_s, total / fps / 600)  # keep long videos quick
    stride = max(1, int(round(fps * step_s)))
    times, scores, prev = [], [], None
    for idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            break
        g = _gray_small(img)
        if prev is not None:
            times.append(idx / fps)
            scores.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    cap.release()
    return np.array(times), np.array(scores)


def adaptive_times(path, n, start, end):
    """Pick about n times: even coverage, plus extra where the picture changes, plus the moments a still picture starts to move."""
    times, scores = motion_profile(path)
    keep = (times >= start) & (times <= end)
    times, scores = times[keep], scores[keep]
    if len(times) < n or n < 4:
        return []
    base_n = max(1, int(n * 0.6))
    base = [start + (end - start) * (i + 0.5) / base_n for i in range(base_n)]  # even coverage
    extra_n = n - base_n
    w = np.sqrt(scores) + 1e-3  # compressed, so shaky close-ups cannot take every extra frame
    cum = np.cumsum(w) / w.sum()
    extra = [float(times[min(int(np.searchsorted(cum, (i + 0.5) / extra_n)), len(times) - 1)]) for i in range(extra_n)]
    onsets = []  # a still picture (1 s or more) that suddenly changes: show just before and just after
    for i in range(4, len(scores)):
        if scores[i] > MOVE_BIG * 0.6 and scores[i - 4:i].max() < MOVE_STILL * 1.5:
            onsets += [times[i] - 0.25, times[i] + 0.5]
    pts = sorted({round(t, 2) for t in base + extra + onsets if start <= t <= end})
    out = []
    for t in pts:  # drop points closer than 0.3 s
        if not out or t - out[-1] >= 0.3:
            out.append(t)
    return out[: n + 6]


def motion_note(g, prev):
    if prev is None:
        return ""
    d = float(np.mean(cv2.absdiff(g, prev)))
    if d < MOVE_STILL:
        return "[nothing moved since the previous frame: a paused picture]"
    if d < MOVE_BIG:
        return "[small change since the previous frame: for example an arrow appearing, or a little movement]"
    return "[big change since the previous frame: real movement, or a cut or zoom]"


MOVE_STILL, MOVE_BIG = 1.0, 4.0  # average grey-level change that counts as paused / big movement


def extract_frames(path, max_frames, max_width, start, end, adaptive=True):
    """Return ([(timestamp_seconds, base64_jpeg, motion_note), ...], duration_seconds)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if total else 0.0

    end = min(end, duration) if end else duration
    if start >= end:
        sys.exit("--start must be earlier than --end / the video duration.")

    n = max(1, min(max_frames, int((end - start) * fps)))
    times = [start + (end - start) * (i + 0.5) / n for i in range(n)]
    if adaptive:
        times = adaptive_times(path, n, start, end) or times

    frames, prev = [], None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        g = _gray_small(img)
        note = motion_note(g, prev)
        prev = g
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames.append((t, base64.b64encode(buf).decode("ascii"), note))
    cap.release()

    if not frames:
        sys.exit("No frames could be extracted.")
    return frames, duration


def build_content(frames, prompt):
    content = []
    for i, fr in enumerate(frames, 1):
        t, b64 = fr[0], fr[1]
        note = fr[2] if len(fr) > 2 else ""
        label = t if isinstance(t, str) else f"Frame {i} (t={t:.1f}s) {note}".strip()
        content.append({"type": "text", "text": f"{label}:"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    content.append({"type": "text", "text": prompt})
    return content


def ask(client, model, frames, prompt, max_tokens):
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,  # reasoning tokens count toward this limit
        messages=[{"role": "user", "content": build_content(frames, prompt)}],
    )
    return (response.choices[0].message.content or "").strip()


COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
MP_IDX = {"nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8, "l_shoulder": 11, "r_shoulder": 12,
          "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
          "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28}
LIMBS = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
BALL_M = 0.22  # a standard size-5 football is about 22 cm wide (used as the ruler)

EQUIP_PROMPT = """These are zoomed crops and frames from one video. Each is labelled with what it shows and its time.
List ONLY equipment you can actually see: ball, boots, shirts, shorts, socks, shin guards, gloves, goals, cones, bags, bottles, anything else.

Return ONLY valid JSON (no markdown fences) in this shape:
{
  "equipment": [
    {"item": "...", "used_by": "...", "visible_details": "colours, patterns, and any logo or text you can really read",
     "brand": "only if a logo or text is clearly readable in a crop, else null",
     "model": "only if readable text names it, else null",
     "confidence": "high | medium | low", "seen_at": [times in seconds], "could_not_tell": "..."}
  ],
  "not_visible": ["things a reader might expect but that cannot be seen"]
}

Rules:
- Never guess a brand or model from shape or colour alone. If it is not readable, use null and say "cannot be determined".
- If two crops disagree, set confidence to low and say so.
- Use confidence "high" only when the same detail is clearly visible in at least two different crops.
- Write so that a 7 to 8 year old child can understand it: short sentences and everyday words."""


def load_pose(backend):
    """Return a pose context (dict) or None. 'auto' tries YOLO (multi-person) then MediaPipe (one person)."""
    if backend == "none":
        return None
    tries = ["yolo", "mediapipe"] if backend == "auto" else [backend]
    for b in tries:
        try:
            if b == "yolo":
                from ultralytics import YOLO
                return {"backend": "yolo", "pose": YOLO("yolo11n-pose.pt"), "det": YOLO("yolo11n.pt")}
            import mediapipe as mp
            pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)
            print("Using MediaPipe: it follows ONE person per frame, which may be the wrong one when players overlap.", file=sys.stderr)
            return {"backend": "mediapipe", "pose": pose}
        except Exception as exc:
            print(f"Pose backend '{b}' unavailable ({exc}).", file=sys.stderr)
    print("Continuing without pose measurements.", file=sys.stderr)
    return None


def find_ball_hough(img, near, person_px, search=150):
    """Fallback ball finder: the circle nearest a foot, with a size limit from the person's height (ball ~ 22 cm, person ~ 1.75 m)."""
    d0 = BALL_M / 1.75 * person_px  # expected ball diameter in pixels
    h, w = img.shape[:2]
    x0, y0 = max(0, int(near[0] - search)), max(0, int(near[1] - search))
    x1, y1 = min(w, int(near[0] + search)), min(h, int(near[1] + search))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    gray = cv2.medianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=22, minRadius=max(8, int(0.4 * d0)), maxRadius=max(12, int(0.65 * d0)))
    if c is None:
        return None
    cx, cy, r = min(c[0], key=lambda q: (q[0] + x0 - near[0]) ** 2 + (q[1] + y0 - near[1]) ** 2)
    return (cx + x0 - r, cy + y0 - r, cx + x0 + r, cy + y0 + r)


def detect_poses(img, ctx):
    """Return (people, ball_box). people = [{"kp": {name: (x, y, confidence)}, "box": (x0, y0, x1, y1)}]."""
    people, ball = [], None
    if ctx["backend"] == "yolo":
        res = ctx["pose"](img, verbose=False)[0]
        if res.keypoints is not None:
            for kp, box in zip(res.keypoints.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                people.append({"kp": {n: tuple(map(float, kp[i])) for i, n in enumerate(COCO)}, "box": tuple(map(float, box))})
        det = ctx["det"](img, classes=[32], verbose=False)[0]  # COCO class 32 = sports ball
        if len(det.boxes):
            ball = tuple(map(float, det.boxes.xyxy[int(det.boxes.conf.argmax())].cpu().numpy()))
    else:
        h, w = img.shape[:2]
        r = ctx["pose"].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks:
            lm = r.pose_landmarks.landmark
            kp = {n: (lm[i].x * w, lm[i].y * h, lm[i].visibility) for n, i in MP_IDX.items()}
            xs, ys = [v[0] for v in kp.values()], [v[1] for v in kp.values()]
            people.append({"kp": kp, "box": (min(xs), min(ys), max(xs), max(ys))})
            for a in ("l_ankle", "r_ankle"):
                ball = ball or find_ball_hough(img, kp[a][:2], max(ys) - min(ys))
    return people, ball


def top_colour(img, k):
    """Rough colour name of the top (shirt) of one detected person."""
    pts = [k[n] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip") if k[n][2] > 0.4]
    if len(pts) < 3:
        return "unknown"
    cx, cy = int(np.mean([q[0] for q in pts])), int(np.mean([q[1] for q in pts]))
    patch = img[max(0, cy - 12):cy + 12, max(0, cx - 12):cx + 12]
    if patch.size == 0:
        return "unknown"
    h, sat, v = (int(x) for x in cv2.cvtColor(np.uint8([[patch.reshape(-1, 3).mean(0)]]), cv2.COLOR_BGR2HSV)[0][0])
    if v < 40:
        return "black"
    if sat < 50:
        return "white" if v > 170 else "grey"
    if h < 10 or h > 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "dark blue" if v < 150 else "blue"
    return "purple"


def colour_group(name):
    n = (name or "").lower()
    return "blue" if ("blue" in n or "navy" in n) else "black" if ("black" in n or "dark" in n) else n


def measure(img, ctx):
    """Measure body angles (degrees) and distances (metres) for every detected person, left to right."""
    people, ball = detect_poses(img, ctx)
    mpp = BALL_M / ((ball[2] - ball[0] + ball[3] - ball[1]) / 2) if ball else None  # metres per pixel
    out = []
    for i, p in enumerate(sorted(people, key=lambda q: q["box"][0]), 1):
        k = p["kp"]

        def pt(n):
            return np.array(k[n][:2]) if k[n][2] > 0.4 else None

        def ang(a, b, c):
            a, b, c = pt(a), pt(b), pt(c)
            if a is None or b is None or c is None:
                return None
            v1, v2 = a - b, c - b
            cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return round(math.degrees(math.acos(float(np.clip(cos, -1, 1)))))

        m = {}
        sh, hp = [pt("l_shoulder"), pt("r_shoulder")], [pt("l_hip"), pt("r_hip")]
        if all(x is not None for x in sh + hp):
            s, h = (sh[0] + sh[1]) / 2, (hp[0] + hp[1]) / 2
            m["Body lean in degrees (0 = standing straight up)"] = round(math.degrees(math.atan2(abs(s[0] - h[0]), h[1] - s[1])))
        for side, nm in (("l", "left"), ("r", "right")):
            m[f"{nm.capitalize()} knee in degrees (180 = straight like a stick, smaller = more bent)"] = ang(f"{side}_hip", f"{side}_knee", f"{side}_ankle")
            m[f"{nm.capitalize()} elbow in degrees (180 = straight like a stick, smaller = more bent)"] = ang(f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        a1, a2 = pt("l_ankle"), pt("r_ankle")
        if mpp and a1 is not None and a2 is not None:
            m["Feet apart (in metres)"] = round(float(np.linalg.norm(a1 - a2)) * mpp, 2)
        if mpp and ball and all(x is not None for x in hp):
            bc = np.array([(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2])
            m["Ball to hips, sideways (in metres)"] = round(abs(float(bc[0] - (hp[0] + hp[1])[0] / 2)) * mpp, 2)
        out.append({"id": i, "top": top_colour(img, k), "kp": k, "box": p["box"], "measures": {a: b for a, b in m.items() if b is not None}})
    return out, ball


def draw_overlay(img, people, ball):
    colors = [(0, 200, 255), (255, 120, 0), (0, 220, 0)]
    for p in people:
        c, k = colors[(p["id"] - 1) % 3], p["kp"]
        for a, b in LIMBS:
            if k[a][2] > 0.4 and k[b][2] > 0.4:
                cv2.line(img, (int(k[a][0]), int(k[a][1])), (int(k[b][0]), int(k[b][1])), c, 3)
        cv2.putText(img, f"P{p['id']} {p.get('top', '')}", (int(p["box"][0]), int(max(24, p["box"][1] - 6))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    if ball:
        cv2.rectangle(img, (int(ball[0]), int(ball[1])), (int(ball[2]), int(ball[3])), (255, 0, 255), 2)


def measure_window(cap, ctx, t, span=0.2, n=5):
    """Median and range of each measurement over n frames around t, so one noisy frame cannot mislead."""
    runs = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - span + i * 2 * span / (n - 1)) * 1000)
        ok, img = cap.read()
        if ok:
            runs.append(measure(img, ctx)[0])
    if not runs:
        return []
    sizes = [len(r) for r in runs]
    count = max(set(sizes), key=sizes.count)
    runs = [r for r in runs if len(r) == count]
    out = []
    for i in range(count):
        merged = {}
        for k in set().union(*[r[i]["measures"] for r in runs]):
            vals = [float(r[i]["measures"][k]) for r in runs if k in r[i]["measures"]]
            if len(vals) >= 3:
                merged[k] = (float(np.median(vals)), min(vals), max(vals))
        out.append({"id": i + 1, "top": runs[0][i].get("top"), "measures": merged})
    return out


def fmt_measures(measures):
    parts = []
    for k, (med, lo, hi) in measures.items():
        flag = " - the computer is not sure" if hi - lo > (0.15 if "metres" in k else 15) else ""
        parts.append(f"{k}: {med:g} (range {lo:g}-{hi:g}){flag}")
    return parts


def measured_text(cap, ctx, t, want=None):
    if ctx is None:
        return ""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        return ""
    return "; ".join(
        f"P{p['id']} ({p.get('top')} top): " + ", ".join(fmt_measures(p["measures"]))
        for p in measure_window(cap, ctx, t) if not want or colour_group(p.get("top")) == want
    )


def spaced_times(cap, n=6):
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    return [d * (i + 0.5) / n for i in range(n)]


def equipment_crops(cap, ctx, times, scale=3, max_crops=14):
    """Zoomed crops of ball, feet and torso at the given times -> [(label, base64_jpeg)]."""
    crops = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        people, ball = measure(img, ctx) if ctx else ([], None)
        boxes = [("ball", ball)] if ball else []
        for p in people:
            k = p["kp"]
            for side in ("l", "r"):
                a = k[f"{side}_ankle"]
                if a[2] > 0.4:
                    boxes.append((f"P{p['id']} {side} foot", (a[0] - 70, a[1] - 70, a[0] + 70, a[1] + 50)))
            ys = [k[n][1] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip") if k[n][2] > 0.4]
            if ys:
                boxes.append((f"P{p['id']} torso and shorts", (p["box"][0] - 10, min(ys) - 25, p["box"][2] + 10, max(ys) + 60)))
        if not boxes:  # no detector: send the whole frame at native resolution instead
            boxes = [("full frame", (0, 0, w, h))]
        for name, (x0, y0, x1, y1) in boxes:
            x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            s = 1 if name == "full frame" else scale
            c = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                crops.append((f"{name} at {t:.1f}s", base64.b64encode(buf).decode("ascii")))
    return crops[:max_crops]


def clean_analysis(data):
    """Make model output safe to use: times become numbers; findings without a usable time are dropped."""
    if not isinstance(data, dict):
        raise ValueError("The model's JSON was not an object.")
    good, dropped = [], 0
    for f in data.get("findings") or []:
        m = re.search(r"-?\d+(?:\.\d+)?", str(f.get("t", ""))) if isinstance(f, dict) else None
        if m:
            f["t"] = float(m.group())
            good.append(f)
        else:
            dropped += 1
    if dropped:
        print(f"Dropped {dropped} finding(s) with no usable time.", file=sys.stderr)
    data["findings"] = good
    for key in ("coaching_points", "better_options", "equipment", "caveats"):
        if not isinstance(data.get(key, []), list):
            data[key] = []
    return data


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The model did not return JSON.")
    chunk = text[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        pass
    cleaned = re.sub(r'\"to\":\s*\[\s*\d+(?:\.\d+)?\s*,\s*\"[a-zA-Z_]+\":\s*\d+(?:\.\d+)?\s*\],\s*', '', chunk)
    cleaned = re.sub(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*"[a-zA-Z_]+":\s*(\d+(?:\.\d+)?)\s*\]', r'[\1, \2]', cleaned)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        import jiter
        return jiter.from_json(chunk.encode("utf-8"))
    except Exception:
        pass
    return json.loads(cleaned)


def frame_at(cap, t, width=640, ctx=None, info=None, want=None):
    """JPEG bytes of the frame at time t (seconds), time stamped; with a skeleton overlay if ctx is given."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, img = cap.read()
    if not ok:
        return None
    if ctx is not None:
        people, ball = measure(img, ctx)
        people = [p for p in people if not want or colour_group(p.get("top")) == want]
        draw_overlay(img, people, ball)
        if info is not None:
            info["people"], info["ball"] = people, ball
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)))
    label = f"{t:.2f}s"
    cv2.rectangle(img, (0, 0), (14 + 12 * len(label), 28), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def verify_finding(client, model, cap, f, span=1.0, measured=""):
    """Re-check one finding on 8 close-together frames; returns {confirmed, best_t, note}."""
    t = float(f["t"])
    frames = []
    for i in range(8):
        tt = max(0.0, t - span + i * (2 * span) / 7)
        data = frame_at(cap, tt, width=768)
        if data:
            frames.append((tt, base64.b64encode(data).decode("ascii")))
    prompt = (
        "Check one claim against these close-together frames (each labelled with its time).\n"
        f"Claim about person {f.get('person')}: {f.get('what_happened')}\n"
        f"Where to look: {f.get('evidence')}\n"
        'Reply ONLY with JSON: {"confirmed": true or false, "best_t": the time from the labels where the claim '
        'is clearest, "note": "one plain sentence on what the frames really show"}. '
        "Say false if the frames do not clearly show it."
    )
    if measured:
        prompt = (f"Pose-model measurements at {t:.1f}s (they can be wrong if the model followed the wrong person; "
                  f"check the claim's numbers against them): {measured}\n") + prompt
    return parse_json(ask(client, model, frames, prompt, max_tokens=3000))


def make_clip(video_path, t, out_path, half=1.5):
    if not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t - half):.2f}", "-t", f"{2 * half:.2f}",
         "-i", video_path, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
         "-pix_fmt", "yuv420p", str(out_path)]
    )
    return r.returncode == 0


KINDS = {"mistake": ("Oops!", "#c0392b"), "risk": ("Be careful", "#b9770e"), "good": ("Great job!", "#1e8449")}
CSS = (
    "body{font-family:system-ui,sans-serif;font-size:1.1rem;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.6;color:#1b1b1b}"
    "h1{font-size:1.6rem}h2{margin-top:2rem;border-bottom:2px solid #ddd;padding-bottom:4px}"
    ".card{border:1px solid #ddd;border-left:6px solid var(--c);border-radius:8px;padding:12px 14px;margin:14px 0}"
    ".tag{display:inline-block;color:#fff;background:var(--c);border-radius:4px;padding:1px 8px;font-size:.8rem;margin-right:6px}"
    ".strip{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.strip img{width:32%;min-width:220px;border-radius:4px}"
    ".note{color:#555;font-size:.9rem}pre{white-space:pre-wrap;background:#f6f6f6;padding:10px;border-radius:6px}"
    "dt{font-weight:600;margin-top:6px}dd{margin:0}.story{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:6px 0}.step{flex:1 1 150px;background:#fff;border:2px solid #1e8449;border-radius:10px;padding:8px 10px}.step span{display:inline-block;background:#1e8449;color:#fff;border-radius:50%;width:24px;height:24px;text-align:center;line-height:24px;margin-right:6px;font-weight:700}.arrow{align-self:center;color:#1e8449;font-size:1.4rem}h5{margin:12px 0 4px;font-size:1.05rem}.fix{background:#eafaf1;border-radius:8px;padding:4px 14px 10px;margin-top:12px}.viz{margin:8px 0;overflow-x:auto}.viz svg{max-width:100%;height:auto}h4{margin:10px 0 2px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:3px 7px;font-size:.85rem;text-align:left;vertical-align:top}"
    "@media (prefers-color-scheme:dark){.step{background:#222}.fix{background:#1c2b22}body{background:#151515;color:#eee}.card{border-color:#444}pre{background:#222}.note{color:#aaa}}"
)


def esc(x):
    return html.escape(str(x or ""))


def fmt_time(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


VISUAL_DOC = """Visuals: add a "visual" to every coaching point and every better option when you have real numbers or a real path. Leave it out if you do not. Pick ONE of these shapes:
- {"type": "distance", "label": "ball to your hips", "from": "your hips", "unit": "m", "now": 0.6, "target_min": 0.3, "target_max": 0.5}   (unit is "m" or "cm"; "now" is optional)
- {"type": "angle", "label": "knee bend", "joint": "knee", "now": 172, "target_min": 150, "target_max": 160}   (joint is "knee", "elbow" or "body lean"; degrees; for knee and elbow 180 = straight; for body lean 0 = standing straight up; "now" is optional)
- {"type": "path", "label": "fake one way, go the other", "actors": [{"id": "you", "role": "player", "x": 5, "y": 5}, {"id": "def", "role": "defender", "x": 6.5, "y": 3.5}, {"id": "ball", "role": "ball", "x": 6, "y": 5}], "steps": [{"actor": "you", "to": [5.3, 5], "label": "pretend to pass", "style": "fake"}, {"actor": "ball", "to": [4.5, 5.6], "label": "pull the ball back"}]}   (top view of the pitch; x goes right 0 to 10, y goes down 0 to 10; keep people and the ball at least 1.5 units apart and moves about 1.5 to 4 units long so the picture is clear; roles: player, defender, ball, teammate; "style": "fake" is a pretend move; steps are drawn in order and numbered)
- {"type": "gap", "label": "Space between you and the attacker", "between": ["you", "attacker"], "unit": "cm", "now": 0, "target_min": 70, "target_max": 100, "hint": "about one arm's length"}   (two stick people with the space between them, left person first; use gap when the number is the space between two people, and distance only for a ball or a foot)
- {"type": "pose", "label": "Stand lower", "joints": [{"joint": "knee", "now": 172, "target_min": 150, "target_max": 160}, {"joint": "body lean", "now": 21, "target_min": 10, "target_max": 20}]}   (a whole-body stick figure of how the body should look; joints are "knee", "body lean" and "elbow"; use it when the fix is about body shape)
- {"type": "timeline", "label": "look up after the touch", "segments": [{"label": "touch the ball", "seconds": 0.3}, {"label": "eyes up", "seconds": 0.5}]}"""


def _n(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w:.0f} {h:.0f}' width='{w:.0f}' role='img'>"
            f"<rect width='{w:.0f}' height='{h:.0f}' rx='10' fill='#fff' stroke='#ccc'/>{body}</svg>")


def _text(x, y, s, size=14, anchor="middle", colour="#222", bold=False, outline=""):
    weight = " font-weight='700'" if bold else ""
    if outline:
        weight += f" stroke='{outline}' stroke-width='3' paint-order='stroke'"
    return (f"<text x='{x:.0f}' y='{y:.0f}' font-size='{size}' text-anchor='{anchor}' fill='{colour}' "
            f"font-family='system-ui,sans-serif'{weight}>{esc(s)}</text>")


def _ball(cx, cy, colour, r=14):
    return (f"<circle cx='{cx:.0f}' cy='{cy:.0f}' r='{r}' fill='#fff' stroke='{colour}' stroke-width='3'/>"
            f"<circle cx='{cx:.0f}' cy='{cy:.0f}' r='{r * 0.35:.0f}' fill='#222'/>")


def svg_distance(v):
    unit = "cm" if v.get("unit") == "cm" else "m"
    lo, hi, now = _n(v.get("target_min")), _n(v.get("target_max")), _n(v.get("now"))
    if lo is None or hi is None or hi <= 0 or lo > hi:
        return ""
    top = max(hi, now or 0) * 1.3
    x0, wid = 70, 450

    def px(val):
        return x0 + wid * min(max(val, 0), top) / top

    tick = next((s for s in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100) if top / s <= 8), 100)
    b = [_text(280, 26, v.get("label", "How far"), 16, bold=True),
         f"<line x1='{x0}' y1='130' x2='{x0 + wid}' y2='130' stroke='#888' stroke-width='3'/>",
         f"<rect x='{x0 - 26}' y='116' width='26' height='14' rx='6' fill='#34495e'/>",
         _text(10, 176, "start: " + str(v.get("from", "your foot")), 12, "start")]
    k = 0
    while k * tick <= top + 1e-9:
        b.append(f"<line x1='{px(k * tick):.0f}' y1='126' x2='{px(k * tick):.0f}' y2='136' stroke='#888'/>")
        b.append(_text(px(k * tick), 152, f"{k * tick:g}", 12))
        k += 1
    b.append(_text(x0 + wid + 6, 152, unit, 12, "start"))
    xa, xb = px(lo), px(hi)
    b.append(f"<rect x='{xa:.0f}' y='72' width='{max(xb - xa, 6):.0f}' height='58' fill='#1e8449' fill-opacity='.18' "
             f"stroke='#1e8449' stroke-dasharray='4 3'/>")
    b.append(_text((xa + xb) / 2, 66, f"just right: {lo:g}-{hi:g} {unit}", 13, colour="#1e8449", bold=True))
    b.append(_ball((xa + xb) / 2, 114, "#1e8449"))
    if now is not None:
        xn = px(now)
        b.append(f"<line x1='{x0}' y1='96' x2='{xn:.0f}' y2='96' stroke='#c0392b' stroke-width='3'/>")
        b.append(f"<polygon points='{xn:.0f},96 {xn - 9:.0f},90 {xn - 9:.0f},102' fill='#c0392b'/>")
        b.append(_ball(xn, 114, "#c0392b"))
        verdict = "too far" if now > hi else "too close" if now < lo else "just right"
        b.append(_text(min(xn, x0 + wid - 40), 188, f"now: {now:g} {unit} ({verdict})", 13, colour="#c0392b", bold=True))
    return _svg(560, 196, "".join(b))


def svg_angle(v):
    joint = v.get("joint")
    lo, hi, now = _n(v.get("target_min")), _n(v.get("target_max")), _n(v.get("now"))
    if joint not in ("knee", "elbow", "body lean") or lo is None or hi is None:
        return ""
    panels = ([("Now", now, "#c0392b")] if now is not None else []) + [("Try this", (lo + hi) / 2, "#1e8449")]
    names = {"knee": ("hip", "knee", "ankle"), "elbow": ("shoulder", "elbow", "wrist")}
    b = [_text(140 * len(panels), 24, v.get("label", joint), 16, bold=True)]
    for i, (title, deg, col) in enumerate(panels):
        cx = 140 + 280 * i
        deg = max(0.0, min(80.0 if joint == "body lean" else 180.0, deg))
        s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
        b.append(_text(cx, 48, f"Now: {deg:.0f} deg" if title == "Now" else f"Goal: {lo:g}-{hi:g} deg", 14, colour=col, bold=True))
        if joint == "body lean":
            bx, by = cx, 196
            b.append(f"<line x1='{bx - 70}' y1='{by}' x2='{bx + 70}' y2='{by}' stroke='#888' stroke-width='3'/>")
            b.append(f"<line x1='{bx}' y1='{by}' x2='{bx}' y2='{by - 115}' stroke='#aaa' stroke-dasharray='5 4'/>")
            b.append(f"<line x1='{bx}' y1='{by}' x2='{bx + 100 * s:.0f}' y2='{by - 100 * c:.0f}' stroke='{col}' stroke-width='10' stroke-linecap='round'/>")
            b.append(f"<circle cx='{bx + 116 * s:.0f}' cy='{by - 116 * c:.0f}' r='13' fill='#fff' stroke='{col}' stroke-width='4'/>")
            b.append(f"<path d='M {bx} {by - 45} A 45 45 0 0 1 {bx + 45 * s:.0f} {by - 45 * c:.0f}' fill='none' stroke='{col}' stroke-width='2'/>")
            b.append(_text(bx + 70 * s + 22, by - 70 * c, f"{deg:.0f}", 13, colour=col, bold=True))
            b.append(_text(bx, by + 16, "0 = standing straight up", 11, colour="#666"))
        else:
            jx, jy, ln = cx, 142, 62
            p2 = (jx + ln * s, jy - ln * c)
            b.append(f"<polyline points='{jx},{jy - ln} {jx},{jy} {p2[0]:.0f},{p2[1]:.0f}' fill='none' stroke='{col}' stroke-width='9' stroke-linecap='round' stroke-linejoin='round'/>")
            for px_, py_ in ((jx, jy - ln), (jx, jy), (p2[0], p2[1])):
                b.append(f"<circle cx='{px_:.0f}' cy='{py_:.0f}' r='7' fill='#fff' stroke='{col}' stroke-width='3'/>")
            b.append(f"<path d='M {jx} {jy - 26} A 26 26 0 0 1 {jx + 26 * s:.0f} {jy - 26 * c:.0f}' fill='none' stroke='{col}' stroke-width='2'/>")
            half = math.radians(deg / 2)
            b.append(_text(jx + 48 * math.sin(half) + 10, jy - 48 * math.cos(half) + 4, f"{deg:.0f}", 13, colour=col, bold=True))
            n1, n2, n3 = names[joint]
            b.append(_text(jx + 14, jy - ln - 10, n1, 11, "start", "#666"))
            b.append(_text(jx - 12, jy + 4, n2, 11, "end", "#666"))
            b.append(_text(p2[0] + 12, p2[1] + 16, n3, 11, "start", "#666"))
    return _svg(280 * len(panels), 232, "".join(b))


def svg_path(v):
    actors = {a["id"]: dict(a) for a in v.get("actors", []) if isinstance(a, dict) and a.get("id") is not None}
    steps = [s for s in v.get("steps", []) if isinstance(s, dict) and s.get("actor") in actors
             and isinstance(s.get("to"), (list, tuple)) and len(s["to"]) == 2]
    if not actors or not steps:
        return ""
    cols = {"player": "#2563eb", "defender": "#222222", "ball": "#f59e0b", "teammate": "#16a34a"}
    names = {"player": "You", "defender": "Defender", "ball": "Ball", "teammate": "Teammate"}

    def px(x):
        return 40 + 48 * max(0.0, min(10.0, _n(x, 0.0)))

    def py(y):
        return 66 + 25 * max(0.0, min(10.0, _n(y, 0.0)))

    w, h = 580, 350 + 24 * len(steps)
    b = [_text(w / 2, 26, v.get("label", "How it moves"), 16, bold=True),
         f"<rect x='10' y='40' width='{w - 20}' height='290' rx='8' fill='#2e7d32'/>",
         _text(w / 2, 56, "(looking down from above)", 11, colour="#e8f5e9")]
    defs = "".join(
        f"<marker id='ar_{r}' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='6' markerHeight='6' orient='auto'>"
        f"<path d='M0,0 L10,5 L0,10 z' fill='{c}'/></marker>" for r, c in cols.items())
    b.insert(0, f"<defs>{defs}</defs>")
    pos = {k: (_n(a.get("x"), 5.0), _n(a.get("y"), 5.0)) for k, a in actors.items()}
    for k, a in actors.items():
        role = a.get("role") if a.get("role") in cols else "player"
        x, y = px(pos[k][0]), py(pos[k][1])
        if role == "ball":
            b.append(_ball(x, y, cols[role], 11))
        else:
            b.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='15' fill='{cols[role]}' stroke='#fff' stroke-width='3'/>")
        b.append(_text(x, y + 32, a.get("label") or names[role], 12, colour="#fff", bold=True, outline="#1b5e20"))
    legend = []
    for i, st in enumerate(steps, 1):
        k = st["actor"]
        role = actors[k].get("role") if actors[k].get("role") in cols else "player"
        x1, y1 = px(pos[k][0]), py(pos[k][1])
        pos[k] = (_n(st["to"][0], pos[k][0]), _n(st["to"][1], pos[k][1]))
        x2, y2 = px(pos[k][0]), py(pos[k][1])
        dash = "7 6" if (role == "ball" or st.get("style") == "fake") else "none"
        opacity = "0.55" if st.get("style") == "fake" else "1"
        b.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='{cols[role]}' stroke-width='4' "
                 f"stroke-dasharray='{dash}' opacity='{opacity}' marker-end='url(#ar_{role})'/>")
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        b.append(f"<circle cx='{mx:.0f}' cy='{my:.0f}' r='11' fill='#fff' stroke='{cols[role]}' stroke-width='3'/>")
        b.append(_text(mx, my + 5, str(i), 13, bold=True))
        tag = " [a pretend move]" if st.get("style") == "fake" else ""
        legend.append(f"{i}. {names.get(role, 'You') if actors[k].get('label') is None else actors[k]['label']}: {st.get('label', '')}{tag}")
    for j, line in enumerate(legend):
        b.append(_text(20, 356 + 24 * j, line, 14, "start"))
    return _svg(w, h, "".join(b))


def svg_timeline(v):
    segs = [s for s in v.get("segments", []) if isinstance(s, dict) and (_n(s.get("seconds")) or 0) > 0]
    if not segs:
        return ""
    total = sum(_n(s["seconds"]) for s in segs)
    cols = ["#2563eb", "#16a34a", "#f59e0b", "#7d3c98", "#c0392b"]
    x, wid = 20.0, 520.0
    b = [_text(280, 26, v.get("label", "Timing"), 16, bold=True)]
    for i, s in enumerate(segs):
        w = wid * _n(s["seconds"]) / total
        b.append(f"<rect x='{x:.0f}' y='60' width='{w:.0f}' height='46' fill='{cols[i % 5]}' stroke='#fff' stroke-width='2'/>")
        b.append(_text(x + w / 2, 88, f"{_n(s['seconds']):g} s", 14, colour="#fff", bold=True))
        b.append(_text(x + w / 2, 128 + 20 * (i % 2), str(s.get("label", ""))[:28], 13))
        x += w
    b.append(_text(280, 176, f"all together: about {total:g} seconds", 13, colour="#555"))
    return _svg(560, 190, "".join(b))


def _figure(cx, ground, lean, knee, elbow, col, faded=False):
    """A side-view stick figure in a ready stance. Angles in degrees; knee and elbow: 180 = straight."""
    op = "0.55" if faded else "1"
    bend = max(0.0, 180.0 - knee)
    s = lambda a: math.sin(math.radians(a))
    c = lambda a: math.cos(math.radians(a))
    legs = []
    for t0, w in ((bend / 2 + 9, 9), (bend / 2 - 9, 7)):  # front leg, back leg
        k = (58 * s(t0), 58 * c(t0))
        a = (k[0] + 58 * s(t0 - bend), k[1] + 58 * c(t0 - bend))
        legs.append((k, a, w))
    hy = ground - max(l[1][1] for l in legs)
    hx = cx
    out = []
    for (k, a, w) in legs:
        pts = f"{hx + k[0] * 0:.0f},{hy:.0f} {hx + k[0]:.0f},{hy + k[1]:.0f} {hx + a[0]:.0f},{hy + a[1]:.0f} {hx + a[0] + 16:.0f},{hy + a[1]:.0f}"
        out.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='{w}' stroke-linecap='round' stroke-linejoin='round' opacity='{op}'/>")
    sx, sy = hx + 74 * s(lean), hy - 74 * c(lean)
    out.append(f"<line x1='{hx:.0f}' y1='{hy:.0f}' x2='{sx:.0f}' y2='{sy:.0f}' stroke='{col}' stroke-width='11' stroke-linecap='round' opacity='{op}'/>")
    out.append(f"<circle cx='{sx + 22 * s(lean):.0f}' cy='{sy - 22 * c(lean):.0f}' r='13' fill='#fff' stroke='{col}' stroke-width='4' opacity='{op}'/>")
    al = 25.0
    ex, ey = sx + 40 * s(al), sy + 40 * c(al)
    fa = al + (180.0 - elbow)
    wx, wy = ex + 38 * s(fa), ey + 38 * c(fa)
    out.append(f"<polyline points='{sx:.0f},{sy:.0f} {ex:.0f},{ey:.0f} {wx:.0f},{wy:.0f}' fill='none' stroke='{col}' stroke-width='7' stroke-linecap='round' stroke-linejoin='round' opacity='{op}'/>")
    return "".join(out)


def svg_pose(v):
    """A whole-body picture of how the body should look, with the current numbers shown next to the goal."""
    base = {"knee": 172.0, "body lean": 5.0, "elbow": 150.0}
    now, goal, lines = dict(base), dict(base), []
    for j in v.get("joints", []):
        if not isinstance(j, dict) or j.get("joint") not in base:
            continue
        lo, hi, n = _n(j.get("target_min")), _n(j.get("target_max")), _n(j.get("now"))
        if lo is None or hi is None:
            continue
        name = j["joint"]
        goal[name] = (lo + hi) / 2
        now[name] = n if n is not None else goal[name]
        label = {"knee": "Knees", "body lean": "Body lean", "elbow": "Elbow"}[name]
        lines.append((label, n, lo, hi))
    if not lines:
        return ""
    show_now = any(n is not None for _, n, _, _ in lines)
    w, h = 560, 270 + 24 * len(lines)
    b = [_text(w / 2, 26, v.get("label", "How the body should look"), 16, bold=True)]
    panels = ([("Now", now, "#c0392b", 150)] if show_now else []) + [("Try this", goal, "#1e8449", 410 if show_now else 280)]
    for title, vals, col, cx in panels:
        b.append(_text(cx, 54, title, 15, colour=col, bold=True))
        b.append(f"<line x1='{cx - 80}' y1='222' x2='{cx + 80}' y2='222' stroke='#888' stroke-width='3'/>")
        fig = _figure(cx, 222, min(max(vals["body lean"], 0.0), 70.0), min(max(vals["knee"], 60.0), 180.0),
                      min(max(vals["elbow"], 40.0), 180.0), col, faded=False)
        b.append(f"<g transform='translate({cx},222) scale(0.66) translate({-cx},-222)'>{fig}</g>")
    for i, (label, n, lo, hi) in enumerate(lines):
        y = 258 + 24 * i
        unit = "degrees"
        txt = f"{label}: " + (f"now {n:g} -> " if n is not None else "") + f"try {lo:g} to {hi:g} {unit}" + (" (180 = straight like a stick)" if label != "Body lean" else " (0 = standing straight up)")
        b.append(_text(20, y, txt, 14, "start", "#1e8449", True))
    return _svg(w, h, "".join(b))


def svg_gap(v):
    """Two stick people with the space between them, 'Now' next to 'Try this'. Left person first."""
    k = 100.0 if v.get("unit") == "m" else 1.0  # work in cm
    lo, hi, now = _n(v.get("target_min")), _n(v.get("target_max")), _n(v.get("now"))
    if lo is None or hi is None:
        return ""
    lo, hi = lo * k, hi * k
    now = None if now is None else now * k
    names = v.get("between") if isinstance(v.get("between"), list) and len(v["between"]) == 2 else ["you", "the other player"]
    panels = ([("Now", now, "#c0392b")] if now is not None else []) + [("Try this", (lo + hi) / 2, "#1e8449")]
    w, h = 290 * len(panels), 270
    b = [_text(w / 2, 26, v.get("label", "The gap between two people"), 16, bold=True)]
    for i, (title, dist, col) in enumerate(panels):
        xl = 290 * i + 60
        d = max(0.0, min(dist, 140.0))
        xr = xl + 30 + d
        head = f"Now: {dist:g} cm" if title == "Now" else f"Try this: {lo:g} to {hi:g} cm"
        b.append(_text(xl + 100, 54, head, 14, colour=col, bold=True))
        b.append(f"<line x1='{xl - 30}' y1='214' x2='{xr + 70:.0f}' y2='214' stroke='#888' stroke-width='3'/>")
        for cx, fill in ((xl, "#222222"), (xr, "#2563eb")):
            fig = _figure(cx, 214, 6, 165, 150, fill)
            b.append(f"<g transform='translate({cx:.0f},214) scale(0.5) translate({-cx:.0f},-214)'>{fig}</g>")
        ya, xa, xb = 150, xl + 22, xr - 8
        if d < 4:
            b.append(f"<circle cx='{(xa + xb) / 2:.0f}' cy='{ya}' r='9' fill='none' stroke='{col}' stroke-width='3'/>")
            b.append(_text((xa + xb) / 2, ya - 18, "touching", 13, colour=col, bold=True))
        else:
            b.append(f"<line x1='{xa:.0f}' y1='{ya}' x2='{xb:.0f}' y2='{ya}' stroke='{col}' stroke-width='3'/>")
            b.append(f"<polygon points='{xa:.0f},{ya} {xa + 9:.0f},{ya - 6} {xa + 9:.0f},{ya + 6}' fill='{col}'/>")
            b.append(f"<polygon points='{xb:.0f},{ya} {xb - 9:.0f},{ya - 6} {xb - 9:.0f},{ya + 6}' fill='{col}'/>")
            b.append(_text((xa + xb) / 2, ya - 14, f"{dist:g} cm", 13, colour=col, bold=True))
        b.append(_text(xl + 6, 236, names[0], 12, "end", "#222"))
        b.append(_text(xr - 6, 236, names[1], 12, "start", "#2563eb"))
    if v.get("hint"):
        b.append(_text(w / 2, 260, str(v["hint"]), 13, colour="#1e8449", bold=True))
    return _svg(w, h, "".join(b))


def render_visual(v):
    fn = {"distance": svg_distance, "angle": svg_angle, "path": svg_path, "timeline": svg_timeline, "pose": svg_pose, "gap": svg_gap}.get(
        v.get("type") if isinstance(v, dict) else None)
    if not fn:
        return ""
    try:
        svg = fn(v)
    except Exception as exc:  # a bad drawing must never break the report
        print(f"Could not draw a visual: {exc}", file=sys.stderr)
        svg = ""
    return f"<div class='viz'>{svg}</div>" if svg else ""


def render_html(title, description, data, cap, window, video_path, clips_dir=None, ctx=None, crops=None):
    all_findings = [f for f in data.get("findings", []) if f.get("t") is not None]
    findings = [f for f in all_findings if f.get("verified") is not False]
    unconfirmed = [f for f in all_findings if f.get("verified") is False]
    people = {p.get("id"): p.get("label", p.get("id")) for p in data.get("people", [])}
    order = list(people) + [i for i in dict.fromkeys(f.get("person") for f in findings) if i not in people]
    out = [
        f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style>",
        f"<h1>{esc(title)}</h1><p>{esc(data.get('summary'))}</p>",
    ]
    if description:
        out.append(f"<details><summary>What happens in the video</summary><pre>{esc(description)}</pre></details>")
    n = 0
    for pid in order:
        out.append(f"<h2>{esc(people.get(pid, pid))}</h2>")
        for f in sorted((x for x in findings if x.get("person") == pid), key=lambda x: float(x["t"])):
            n += 1
            t = float(f["t"])
            name, color = KINDS.get(f.get("kind"), ("Note", "#555"))
            imgs, info = "", {}
            for tt in (max(0.0, t - window), t, t + window):
                want = colour_group(f.get("top_colour"))
                b = frame_at(cap, tt, ctx=ctx, info=info if tt == t else None, want=want)
                if b:
                    imgs += f"<img alt='frame at {tt:.2f}s' src='data:image/jpeg;base64,{base64.b64encode(b).decode()}'>"
            meas = ""
            shown = [p for p in measure_window(cap, ctx, t) if not want or colour_group(p.get("top")) == want] if ctx else []
            if ctx and not shown:
                meas = "<p class='note'>No body numbers here: the computer lost track of this person at this moment.</p>"
            if shown:
                rows = ""
                for p in shown:
                    vals = "; ".join(fmt_measures(p["measures"])) or "no joints clearly visible"
                    rows += f"<tr><td>P{p['id']} ({esc(p.get('top'))} top)</td><td>{esc(vals)}</td></tr>"
                meas = (
                    "<details open><summary>What the computer measured on the body (the middle number is the typical one over 5 pictures; P1 = first person from the left)</summary>"
                    f"<table>{rows}</table><p class='note'>These numbers come from a flat picture, so they are best when "
                    "the person runs across the screen. \"The computer is not sure\" means the number jumped around. Look at the stick-figure lines: "
                    "if they are on the wrong person or the wrong body part, ignore the numbers.</p></details>"
                )
            badge = {True: "I looked again up close and it is true", False: "I looked again and I am NOT sure about this one"}.get(f.get("verified"), "")
            clip = ""
            if clips_dir and make_clip(video_path, t, clips_dir / f"finding_{n:02d}.mp4"):
                clip = f"<p><a href='{clips_dir.name}/finding_{n:02d}.mp4'>Play a 3-second clip around this moment</a></p>"
            rows = [("What I saw", f.get("what_happened")), ("Look here in the pictures", f.get("evidence")),
                    ("Why it matters", f.get("why_it_matters")), ("Try this instead", f.get("fix")),
                    ("Goal", f.get("target")), ("I looked again and saw", f.get("verify_note"))]
            dl = "".join(f"<dt>{k}</dt><dd>{esc(v)}</dd>" for k, v in rows if v)
            steps = f.get("fix_steps") if isinstance(f.get("fix_steps"), list) else []
            fix_list = "".join(f"<li>{esc(x)}</li>" for x in steps)
            fix_pic = render_visual(f.get("fix_visual"))
            then = f.get("then_do") if isinstance(f.get("then_do"), list) else []
            story = ""
            if then:
                cards = "<div class='arrow'>&#10140;</div>".join(
                    f"<div class='step'><span>{n_}</span>{esc(x)}</div>" for n_, x in enumerate(then, 1))
                story = f"<h5>Step 2: Now do this, one after the other</h5><div class='story'>{cards}</div>{render_visual(f.get('then_visual'))}"
            fix = ""
            if f.get("kind") != "good" and (fix_list or fix_pic or story):
                fix = ("<div class='fix'><h4>How to fix it</h4><h5>Step 1: Get into the right position</h5>"
                       f"{fix_pic}<ol>{fix_list}</ol>{story}</div>")
            out.append(
                f"<div class='card' style='--c:{color}'><span class='tag'>{name}</span><b>{esc(f.get('body_part'))}</b> "
                f"at {fmt_time(t)} ({t:.1f}s) <span class='note'>{badge}</span>"
                f"<div class='strip'>{imgs}</div>{meas}{clip}<dl>{dl}</dl>{fix}</div>"
            )
    if unconfirmed:
        out.append("<h2>Things I looked at again and could not confirm</h2><p class='note'>I am not showing these as mistakes, because the close-up pictures did not clearly show them.</p><ul>")
        for f in unconfirmed:
            out.append(f"<li><b>{esc(f.get('body_part'))}</b> at {float(f['t']):.1f}s: {esc(f.get('what_happened'))} <span class='note'>({esc(f.get('verify_note'))})</span></li>")
        out.append("</ul>")
    if data.get("equipment"):
        out.append("<h2>Gear I can see</h2><table><tr><th>Thing</th><th>Who uses it</th><th>What I can see</th><th>Brand name</th><th>How sure I am</th><th>What I cannot tell</th></tr>")
        for e in data["equipment"]:
            bm = " / ".join(x for x in (e.get("brand"), e.get("model")) if x) or "I cannot tell"
            out.append(
                f"<tr><td>{esc(e.get('item'))}</td><td>{esc(e.get('used_by'))}</td><td>{esc(e.get('visible_details'))}</td>"
                f"<td>{esc(bm)}</td><td>{esc(e.get('confidence'))}</td><td>{esc(e.get('could_not_tell'))}</td></tr>"
            )
        out.append("</table>")
        if data.get("equipment_not_visible"):
            out.append("<p class='note'>I cannot see these: " + esc("; ".join(data["equipment_not_visible"])) + "</p>")
        if crops:
            out.append("<p class='note'>Zoomed-in pictures I used to look at the gear:</p><div class='strip'>" + "".join(
                f"<figure style='margin:0'><img style='width:170px;min-width:0' alt='{esc(lb)}' src='data:image/jpeg;base64,{b}'>"
                f"<figcaption class='note'>{esc(lb)}</figcaption></figure>" for lb, b in crops) + "</div>")
    if data.get("coaching_points"):
        out.append("<h2>Coaching points (how to show the player)</h2>")
        for c in data["coaching_points"]:
            show = "".join(f"<li>{esc(x)}</li>" for x in c.get("show_the_player", []))
            check = f"<p><b>How to check:</b> {esc(c.get('how_to_check'))}</p>" if c.get("how_to_check") else ""
            demo = f"<h4>How to show the player</h4><ol>{show}</ol>" if show else ""
            out.append(f"<div class='card' style='--c:#7d3c98'><b>{esc(c.get('title'))}</b><p>{esc(c.get('what_it_means'))}</p>"
                       f"{render_visual(c.get('visual'))}{demo}{check}</div>")
    if data.get("better_options"):
        out.append("<h2>Cool ways to do it better</h2>")
        for o in data["better_options"]:
            steps = "".join(f"<li>{esc(s)}</li>" for s in o.get("steps", []))
            show = "".join(f"<li>{esc(x)}</li>" for x in o.get("show_the_player", []))
            demo = f"<h4>How to show the player</h4><ol>{show}</ol>" if show else ""
            out.append(f"<div class='card' style='--c:#2874a6'><b>{esc(o.get('name'))}</b><p>{esc(o.get('when'))}</p>"
                       f"{render_visual(o.get('visual'))}<h4>The moves</h4><ol>{steps}</ol>{demo}</div>")
    if data.get("caveats"):
        out.append("<h2>Things I am not sure about</h2><ul>" + "".join(f"<li>{esc(c)}</li>" for c in data["caveats"]) + "</ul>")
    return "\n".join(out)


def write_report(args, stem, description, analysis, cap, ctx=None, crops=None):
    clips_dir = None
    if args.clips:
        clips_dir = Path(f"{stem}_clips")
        clips_dir.mkdir(exist_ok=True)
    if crops is None and analysis.get("equipment"):
        crops = equipment_crops(cap, ctx, spaced_times(cap))
    doc = render_html(f"Video analysis: {Path(args.video).name}", description, analysis, cap, args.window,
                      args.video, clips_dir, ctx, crops)
    out_path = args.out or f"{stem}_report.html"
    Path(out_path).write_text(doc, encoding="utf-8")
    print(f"Report saved to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1: persistent multi-player tracking and a per-player movement timeline.
# Runs on EVERY frame in order (not on the sampled frames used for the model prompts).
# Needs Ultralytics (pip install ultralytics). The timeline code itself needs no model.
# ---------------------------------------------------------------------------
TRACK_KP_MIN_CONF = 0.5  # a keypoint below this confidence is treated as "not reliable" (stored as null)
TRACK_MAX_GAP_S = 0.5  # speed is only computed if the same track was seen this recently
TRACKERS = {"botsort": "botsort.yaml", "bytetrack": "bytetrack.yaml"}
TRACK_UNITS = {
    "time": "seconds",
    "box": "pixels [x0, y0, x1, y1] in the original frame",
    "keypoints": "pixels in the original frame; x and y are null when confidence is below the reliability threshold",
    "angles": "degrees measured in the flat 2D picture (not 3D body angles)",
    "speed": "pixels per second of the box centre",
    "calibrated": False,
    "camera_motion_compensated": False,
}


def _to_np(x):
    """Torch tensor or array-like -> numpy array (None stays None)."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def json_safe(o):
    """Make a structure safe for strict JSON: numpy types become Python types, NaN and infinity become null."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def joint_angle_deg(a, b, c):
    """Angle at point b between the lines to a and c, in degrees (180 = straight). None if a point is missing or too close."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = np.asarray(a, float) - np.asarray(b, float), np.asarray(c, float) - np.asarray(b, float)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 5 or n2 < 5:  # segments shorter than 5 px are too noisy to give an angle
        return None
    return round(math.degrees(math.acos(float(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0)))), 1)


def trunk_lean_deg(l_sh, r_sh, l_hip, r_hip):
    """How far the shoulders-to-hips line leans from vertical, in degrees (0 = upright). None if any point is missing."""
    if any(p is None for p in (l_sh, r_sh, l_hip, r_hip)):
        return None
    s, h = (np.asarray(l_sh, float) + np.asarray(r_sh, float)) / 2, (np.asarray(l_hip, float) + np.asarray(r_hip, float)) / 2
    dy = float(h[1] - s[1])
    if dy < 10:  # torso shorter than 10 px, or upside down: not meaningful
        return None
    return round(math.degrees(math.atan2(abs(float(h[0] - s[0])), dy)), 1)


def _box_centre(box):
    return None if box is None else ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class FrameClock:
    """Turns frames into seconds. Uses the video's own timestamp when it is valid (screen recordings can have a variable
    frame rate), otherwise frame_number / fps. Time never goes backwards."""

    def __init__(self, fps):
        self.fps = fps if fps and fps > 0 else 30.0
        self.last = None
        self.counts = {"video_timestamp": 0, "frame_number_over_fps": 0, "clamped_to_keep_order": 0}

    def time_of(self, frame_idx, pos_msec):
        t = None
        if pos_msec is not None and math.isfinite(pos_msec) and pos_msec > 0:
            t = pos_msec / 1000.0
            self.counts["video_timestamp"] += 1
        if t is None:
            t = frame_idx / self.fps
            self.counts["frame_number_over_fps"] += 1
        if self.last is not None and t < self.last:
            t = self.last
            self.counts["clamped_to_keep_order"] += 1
        self.last = t
        return t


class TrackingTimeline:
    """Collects tracking results frame by frame into one separate timeline per track ID.
    Plain Python and numpy only: no model is needed, so it can be unit tested with mock detections."""

    MEASURE_KEYS = ("knee_angle_left_deg", "knee_angle_right_deg", "body_lean_deg", "speed_px_s")

    def __init__(self, fps, frame_size, kp_min_conf=TRACK_KP_MIN_CONF, max_gap_s=TRACK_MAX_GAP_S):
        self.fps = fps
        self.frame_size = [int(frame_size[0]), int(frame_size[1])]
        self.kp_min_conf = kp_min_conf
        self.max_gap_s = max_gap_s
        self.tracks = {}  # track id -> list of samples (oldest first)
        self.processed_frames = []
        self.frames_without_detections = 0
        self.untracked_detections = 0  # detections that came back without an ID: kept out of every timeline
        self.duplicate_id_detections = 0
        self._frames_with_tracks = set()
        self._recent_speeds = {}

    def add_frame(self, frame_idx, t, detections):
        """detections: list of {"track_id": int | None, "box": [x0, y0, x1, y1], "box_conf": float | None, "kp": (17, 3) array}."""
        self.processed_frames.append(int(frame_idx))
        if not detections:
            self.frames_without_detections += 1
            return []
        by_id = {}
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                self.untracked_detections += 1
                continue
            tid = int(tid)
            if tid in by_id:  # two boxes with the same ID in one frame should not happen: keep the more confident one
                self.duplicate_id_detections += 1
                if (d.get("box_conf") or 0.0) <= (by_id[tid].get("box_conf") or 0.0):
                    continue
            by_id[tid] = d
        made = []
        for tid, d in by_id.items():
            rec = self._make_sample(int(frame_idx), float(t), tid, d)
            self.tracks.setdefault(tid, []).append(rec)
            self._frames_with_tracks.add(int(frame_idx))
            made.append(rec)
        return made

    def _make_sample(self, frame_idx, t, tid, d):
        kp = np.asarray(d["kp"], dtype=float).reshape(-1, 3)
        keypoints, pts = {}, {}
        for i, name in enumerate(COCO):
            x, y, c = (float(v) for v in kp[i])
            ok = math.isfinite(x) and math.isfinite(y) and math.isfinite(c) and c >= self.kp_min_conf
            keypoints[name] = {"x": round(x, 1) if ok else None, "y": round(y, 1) if ok else None,
                               "conf": round(c, 3) if math.isfinite(c) else None}
            pts[name] = np.array([x, y]) if ok else None
        box = [float(v) for v in d["box"]]
        box = [round(v, 1) for v in box] if all(math.isfinite(v) for v in box) else None
        centre = _box_centre(box)
        measures = {
            "knee_angle_left_deg": joint_angle_deg(pts["l_hip"], pts["l_knee"], pts["l_ankle"]),
            "knee_angle_right_deg": joint_angle_deg(pts["r_hip"], pts["r_knee"], pts["r_ankle"]),
            "body_lean_deg": trunk_lean_deg(pts["l_shoulder"], pts["r_shoulder"], pts["l_hip"], pts["r_hip"]),
            "centre_px": None if centre is None else [round(centre[0], 1), round(centre[1], 1)],
            "velocity_px_s": None, "speed_px_s": None, "speed_px_s_smoothed": None,
        }
        prev = self.tracks[tid][-1] if tid in self.tracks else None
        if prev is not None and centre is not None and prev["measures"]["centre_px"] is not None:
            dt = t - prev["t"]
            if 0 < dt <= self.max_gap_s:
                vx = (centre[0] - prev["measures"]["centre_px"][0]) / dt
                vy = (centre[1] - prev["measures"]["centre_px"][1]) / dt
                speed = math.hypot(vx, vy)
                measures["velocity_px_s"] = [round(vx, 1), round(vy, 1)]
                measures["speed_px_s"] = round(speed, 1)
                recent = [(tt, s) for tt, s in self._recent_speeds.get(tid, []) if t - tt <= 1.0] + [(t, speed)]
                self._recent_speeds[tid] = recent[-5:]
                if len(self._recent_speeds[tid]) >= 3:  # a median of fewer than 3 values is not reported
                    measures["speed_px_s_smoothed"] = round(float(np.median([s for _, s in self._recent_speeds[tid]])), 1)
        return {"t": round(t, 3), "frame": frame_idx, "track_id": tid, "box": box,
                "box_conf": None if d.get("box_conf") is None or not math.isfinite(float(d["box_conf"])) else round(float(d["box_conf"]), 3),
                "keypoints": keypoints, "measures": measures}

    def track_stats(self):
        """Per track: first and last time, frames seen, and frames MISSING between its first and last sighting."""
        out = {}
        for tid, recs in self.tracks.items():
            seen = {r["frame"] for r in recs}
            first, last = recs[0]["frame"], recs[-1]["frame"]
            life = [f for f in self.processed_frames if first <= f <= last]
            missing, run, longest = 0, 0, 0
            for f in life:
                if f in seen:
                    run = 0
                else:
                    missing += 1
                    run += 1
                    longest = max(longest, run)
            out[tid] = {
                "first_frame": first, "last_frame": last, "first_t": recs[0]["t"], "last_t": recs[-1]["t"],
                "frames_observed": len(recs), "frames_in_life": len(life),
                "missing_frames_inside_life": missing, "longest_gap_frames": longest,
                "non_null_counts": {k: sum(1 for r in recs if r["measures"][k] is not None) for k in self.MEASURE_KEYS},
            }
        return out

    def suspected_restarts(self, max_gap_s=1.0, dist_factor=1.5):
        """HEURISTIC, not a measured ID switch: one track ends and a different track starts close by shortly after.
        It may be an ID switch, or two different people. Use hand labels (evaluate_tracking) to count real switches."""
        found = []
        for a, recs_a in self.tracks.items():
            end = recs_a[-1]
            for b, recs_b in self.tracks.items():
                if a == b:
                    continue
                start = recs_b[0]
                gap = start["t"] - end["t"]
                if not (0 < gap <= max_gap_s) or end["box"] is None or start["box"] is None:
                    continue
                ca, cb = _box_centre(end["box"]), _box_centre(start["box"])
                diag = math.hypot(end["box"][2] - end["box"][0], end["box"][3] - end["box"][1])
                dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                if dist <= dist_factor * diag:
                    found.append({"ended_track": a, "started_track": b, "gap_s": round(gap, 3), "distance_px": round(dist, 1)})
        return found

    def to_dict(self, metadata=None):
        return json_safe({
            "schema_version": 1,
            "metadata": dict(metadata or {}),
            "units": TRACK_UNITS,
            "frame_stats": {
                "frames_processed": len(self.processed_frames),
                "frames_without_any_detection": self.frames_without_detections,
                "frames_with_at_least_one_track": len(self._frames_with_tracks),
                "untracked_detections": self.untracked_detections,
                "duplicate_id_detections": self.duplicate_id_detections,
                "fps": self.fps, "frame_size": self.frame_size,
            },
            "track_stats": self.track_stats(),
            "suspected_id_restarts_heuristic": self.suspected_restarts(),
            "tracks": {str(tid): {"samples": recs} for tid, recs in sorted(self.tracks.items())},
        })


def evaluate_tracking(timeline_dict, truth):
    """Count real ID switches and missing frames against HAND LABELS. truth = {"labels": [{"frame": 120,
    "players": {"A": [x, y], "B": [x, y]}}, ...]}, where [x, y] is any pixel on that player's body in that frame."""
    boxes_at = {}
    for tid, tr in timeline_dict.get("tracks", {}).items():
        for s in tr["samples"]:
            if s["box"] is not None:
                boxes_at.setdefault(s["frame"], []).append((int(tid), s["box"]))
    processed = set(timeline_dict.get("_processed_frames", []))
    result = {}
    last_id, per_player = {}, {}
    shared = 0
    for lab in sorted(truth.get("labels", []), key=lambda r: r["frame"]):
        f, matched_now = lab["frame"], {}
        for name, (x, y) in lab["players"].items():
            p = per_player.setdefault(name, {"labelled": 0, "missing": 0, "id_switches": 0, "ids_used": [], "not_processed": 0})
            if processed and f not in processed:
                p["not_processed"] += 1
                continue
            p["labelled"] += 1
            hits = [(tid, b) for tid, b in boxes_at.get(f, []) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
            if not hits:
                p["missing"] += 1
                continue
            tid = min(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]))[0]  # tightest box wins
            matched_now[name] = tid
            if name in last_id and last_id[name] != tid:
                p["id_switches"] += 1
            last_id[name] = tid
            if tid not in p["ids_used"]:
                p["ids_used"].append(tid)
        if len(set(matched_now.values())) < len(matched_now):
            shared += 1
    result["players"] = per_player
    result["total_id_switches"] = sum(p["id_switches"] for p in per_player.values())
    result["total_missing_labels"] = sum(p["missing"] for p in per_player.values())
    result["labels_where_two_players_share_one_track"] = shared
    return result


def parse_tracking_result(res, kp_count=17):
    """Ultralytics Results (pose model, tracked) -> list of detections. Boxes, IDs and keypoints keep the same row order."""
    boxes, kps = getattr(res, "boxes", None), getattr(res, "keypoints", None)
    if boxes is None or len(boxes) == 0:
        return []
    if kps is None:
        raise RuntimeError("The model returned no keypoints. Use a pose model such as yolo11n-pose.pt.")
    xyxy = _to_np(boxes.xyxy)
    n = len(xyxy)
    conf = _to_np(getattr(boxes, "conf", None))
    ids = _to_np(getattr(boxes, "id", None))
    data = _to_np(kps.data)
    if data.ndim != 3 or data.shape[0] != n or data.shape[1] != kp_count:
        raise RuntimeError(f"Unexpected keypoint shape {tuple(data.shape)}; expected ({n}, {kp_count}, 2 or 3).")
    if data.shape[2] == 2:  # no confidence available: mark it unknown so every keypoint counts as not reliable
        data = np.concatenate([data, np.full((n, kp_count, 1), np.nan)], axis=2)
    dets = []
    for i in range(n):
        tid = None
        if ids is not None and i < len(ids) and math.isfinite(float(ids[i])):
            tid = int(ids[i])
        dets.append({"track_id": tid, "box": [float(v) for v in xyxy[i]],
                     "box_conf": None if conf is None else float(conf[i]), "kp": data[i]})
    return dets


def _id_colour(tid):
    palette = [(60, 76, 231), (113, 204, 46), (219, 152, 52), (15, 196, 241), (182, 89, 155), (34, 126, 230), (156, 188, 26), (80, 62, 44)]
    return palette[int(tid) % len(palette)]


def draw_tracking_frame(frame, detections, kp_min_conf=TRACK_KP_MIN_CONF, t=None, id_prefix="trk", banner=None):
    """Return a copy of the frame with box, label and skeleton drawn on it. Local tracker IDs are labelled 'trk N' (they are
    not player identities). A detection may carry its own "label" (e.g. a permanent player label) and a small "sublabel"."""
    out = frame.copy()
    for d in detections:
        tid = d.get("track_id")
        col = (160, 160, 160) if tid is None else _id_colour(tid)
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        label = d.get("label") or ("no ID" if tid is None else f"{id_prefix} {tid}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x0, max(0, y0 - th - 8)), (x0 + tw + 8, max(th + 8, y0)), col, -1)
        cv2.putText(out, label, (x0 + 4, max(th + 2, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if d.get("sublabel"):
            cv2.putText(out, str(d["sublabel"]), (x0, min(out.shape[0] - 4, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
        kp = np.asarray(d["kp"], float)
        ok = {name: bool(np.isfinite(kp[i]).all() and kp[i][2] >= kp_min_conf) for i, name in enumerate(COCO)}
        pos = {name: (int(kp[i][0]), int(kp[i][1])) for i, name in enumerate(COCO) if ok[name]}
        for a, b in LIMBS:
            if ok[a] and ok[b]:
                cv2.line(out, pos[a], pos[b], col, 2)
        for p in pos.values():
            cv2.circle(out, p, 3, (255, 255, 255), -1)
    if t is not None:
        cv2.putText(out, f"{t:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if banner:
        cv2.putText(out, banner, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return out


def _version_tuple(v):
    parts = []
    for piece in str(v).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def load_tracking_model(model_name, tracker_yaml):
    """Load a FRESH pose model (a fresh model means fresh tracker state) and check this Ultralytics install can track with it."""
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Tracking needs Ultralytics: pip install ultralytics") from exc
    version = getattr(ultralytics, "__version__", "unknown")
    print(f"Ultralytics {version}", file=sys.stderr)
    if "yolo11" in model_name and _version_tuple(version) < (8, 3, 0):
        print("Warning: YOLO11 weights need Ultralytics 8.3.0 or newer. Run: pip install -U ultralytics", file=sys.stderr)
    if not hasattr(YOLO, "track"):
        raise RuntimeError("This Ultralytics version has no YOLO.track(). Run: pip install -U ultralytics")
    try:
        from ultralytics.utils.checks import check_yaml
        check_yaml(tracker_yaml)
    except Exception as exc:
        raise RuntimeError(f"Tracker config '{tracker_yaml}' was not found in this Ultralytics install: {exc}") from exc
    model = YOLO(model_name)
    if getattr(model, "task", None) != "pose":
        raise RuntimeError(f"'{model_name}' is a '{getattr(model, 'task', None)}' model, not a pose model. Use e.g. yolo11n-pose.pt")
    return model


def _try_h264(path):
    """Re-encode to H.264 so browsers can play it; keeps the original if ffmpeg is missing or fails."""
    if not shutil.which("ffmpeg"):
        return path
    tmp = str(Path(path).with_suffix(".h264.mp4"))
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)
    return path


def run_tracking(video_path, out_prefix, tracker="botsort", model_name="yolo11n-pose.pt", start=0.0, end=0.0, stride=1,
                 conf=None, imgsz=None, device="", save_video=True, model=None):
    """Track players through the video frame by frame, in order, with persistent tracker state.
    Returns (timeline, metadata, annotated_video_path or None). Pass `model` to inject a fake model in tests."""
    if stride < 1:
        raise ValueError("stride must be 1 or more")
    tracker_yaml = TRACKERS.get(tracker, tracker)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if model is None:
        model = load_tracking_model(model_name, tracker_yaml)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    track_kwargs = {"persist": True, "tracker": tracker_yaml, "verbose": False}
    if conf is not None:
        track_kwargs["conf"] = conf  # otherwise Ultralytics' tracking default (0.1) is used, which the trackers expect
    if imgsz:
        track_kwargs["imgsz"] = imgsz
    if device:
        track_kwargs["device"] = device
    timeline, clock = TrackingTimeline(fps, size), FrameClock(fps)
    writer, video_out, n_read = None, f"{out_prefix}_tracked.mp4" if save_video else None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = start_frame + n_read
        t = clock.time_of(idx, cap.get(cv2.CAP_PROP_POS_MSEC))
        if end and t > end:
            break
        n_read += 1
        if (idx - start_frame) % stride:
            continue
        results = model.track(frame, **track_kwargs)
        dets = parse_tracking_result(results[0]) if results else []
        timeline.add_frame(idx, t, dets)
        if save_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
            writer.write(draw_tracking_frame(frame, dets, t=t))
        if len(timeline.processed_frames) % 100 == 0:
            print(f"  tracked {len(timeline.processed_frames)} frames...", file=sys.stderr)
    cap.release()
    if writer is not None:
        writer.release()
        _try_h264(video_out)
    version = None
    try:
        import ultralytics
        version = getattr(ultralytics, "__version__", None)
    except ImportError:
        pass
    meta = {"video": str(video_path), "model": model_name, "tracker": tracker_yaml, "ultralytics_version": version,
            "frame_stride": stride, "start_frame": start_frame, "time_source_counts": clock.counts,
            "detection_conf": "ultralytics tracking default" if conf is None else conf,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    return timeline, meta, (video_out if writer is not None else None)


def write_tracking_json(timeline, meta, path):
    d = timeline.to_dict(meta)
    d["_processed_frames"] = list(timeline.processed_frames)  # lets evaluate_tracking ignore frames that were never processed
    Path(path).write_text(json.dumps(d, indent=1, allow_nan=False), encoding="utf-8")
    return d


def format_tracking_summary(d, evaluation=None):
    fs = d["frame_stats"]
    lines = [f"Tracker: {d['metadata'].get('tracker')}  |  frames processed: {fs['frames_processed']}  |  "
             f"frames with no detection at all: {fs['frames_without_any_detection']}  |  "
             f"detections without an ID (left out of timelines): {fs['untracked_detections']}",
             f"Tracks (IDs) created: {len(d['tracks'])}"]
    for tid, s in d["track_stats"].items():
        lines.append(f"  ID {tid}: {s['first_t']:.2f}s to {s['last_t']:.2f}s, seen in {s['frames_observed']} frames, "
                     f"MISSING in {s['missing_frames_inside_life']} frames inside its life (longest gap {s['longest_gap_frames']} frames), "
                     f"non-null knee/lean/speed: {s['non_null_counts']['knee_angle_left_deg']}/"
                     f"{s['non_null_counts']['body_lean_deg']}/{s['non_null_counts']['speed_px_s']}")
    sr = d["suspected_id_restarts_heuristic"]
    lines.append(f"Suspected ID restarts (heuristic, NOT verified ID switches): {len(sr)}")
    if evaluation:
        lines.append(f"Against your hand labels: ID switches = {evaluation['total_id_switches']}, "
                     f"labels with no matching box = {evaluation['total_missing_labels']}, "
                     f"labels where two players share one track = {evaluation['labels_where_two_players_share_one_track']}")
        for name, p in evaluation["players"].items():
            lines.append(f"  player {name}: labelled {p['labelled']}, missing {p['missing']}, switches {p['id_switches']}, track IDs used {p['ids_used']}")
    else:
        lines.append("ID switches cannot be counted without hand labels (use --track-truth).")
    return lines


def run_tracking_cli(args):
    names = ["botsort", "bytetrack"] if args.tracker == "both" else [args.tracker]
    truth = json.loads(Path(args.track_truth).read_text(encoding="utf-8")) if args.track_truth else None
    stem = Path(args.video).stem
    for name in names:
        print(f"Tracking with {name} on every frame (this can take a while)...", file=sys.stderr)
        try:
            timeline, meta, video_out = run_tracking(
                args.video, f"{stem}_{name}", name, args.track_model, args.start, args.end, args.track_stride,
                args.track_conf, args.track_imgsz, args.track_device, not args.track_no_video)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            sys.exit(f"Tracking failed: {exc}")
        json_path = f"{stem}_{name}_tracking.json"
        d = write_tracking_json(timeline, meta, json_path)
        print("\n".join(format_tracking_summary(d, evaluate_tracking(d, truth) if truth else None)))
        print(f"Saved: {json_path}" + (f" and {video_out}" if video_out else ""), file=sys.stderr)


# ---------------------------------------------------------------------------
# GLOBAL PLAYER IDENTITY: a permanent player_id that survives camera cuts, angle/zoom changes and occlusion.
# BoT-SORT stays the LOCAL tracker (local_track_id, restarted at every camera cut = new shot_id).
# The identity logic lives in global_identity.py; this section runs the frame loop, builds the global timeline
# and the annotated video, and wires the command line. Local IDs are never shown as player identities.
# ---------------------------------------------------------------------------
def _kp_array_from_sample(kps):
    """Rebuild a (17, 3) array from a timeline sample (null x/y become NaN so they are drawn/used as unreliable)."""
    arr = np.full((17, 3), np.nan)
    for i, name in enumerate(COCO):
        k = (kps or {}).get(name) or {}
        arr[i] = (np.nan if k.get("x") is None else k["x"], np.nan if k.get("y") is None else k["y"],
                  np.nan if k.get("conf") is None else k["conf"])
    return arr


def build_global_timeline(result, timelines):
    """Merge per-shot local timelines into one timeline per permanent player_id.
    Only tracklets the identity manager assigned (confident match, or a newly created player) are combined; everything else
    stays in `unassigned_tracklets`. Every sample keeps local_track_id and shot_id as SEPARATE fields from player_id."""
    players, unassigned = {}, {}
    for shot_id, tl in timelines.items():
        for local_id, recs in tl.tracks.items():
            for s in recs:
                seg = result.segment_for(shot_id, local_id, s["frame"])
                pid = None if seg is None else seg.player_id
                rec = {k: v for k, v in s.items() if k != "track_id"}
                rec.update({"player_id": pid, "shot_id": shot_id, "local_track_id": local_id,
                            "tracklet": f"s{shot_id}_t{local_id}" if seg is None else seg.key,
                            "identity_status": "unassigned" if seg is None else seg.status,
                            "identity_confidence": None if seg is None else seg.confidence})
                if pid is None:
                    unassigned.setdefault(rec["tracklet"], []).append(rec)
                else:
                    players.setdefault(pid, []).append(rec)
    out = {}
    for pid, recs in sorted(players.items()):
        recs.sort(key=lambda r: (r["t"], r["frame"]))
        p = result.players[pid]
        out[str(pid)] = {"player_id": pid, "label": f"P{pid}" + ("?" if p.origin == "uncertain_new" else ""),
                         "origin": p.origin, "first_t": recs[0]["t"], "last_t": recs[-1]["t"],
                         "shots_seen": sorted({r["shot_id"] for r in recs}),
                         "tracklets": sorted({r["tracklet"] for r in recs}), "n_samples": len(recs), "samples": recs}
    for recs in unassigned.values():
        recs.sort(key=lambda r: (r["t"], r["frame"]))
    return {"players": out, "unassigned_tracklets": unassigned}


def render_identity_video(video_path, out_path, timelines, result, start_frame, shot_starts, show_local_ids=False):
    """Second pass over the video: draw the PERMANENT player label, box and skeleton. A tracklet with no confident identity is
    drawn as 'P? (unmatched)'. A local tracker ID is only drawn (small, grey, prefixed 'trk') with show_local_ids."""
    by_frame = {}
    for shot_id, tl in timelines.items():
        for local_id, recs in tl.tracks.items():
            for s in recs:
                if s["box"] is None:
                    continue
                pid, _ = result.player_for(shot_id, local_id, s["frame"])
                by_frame.setdefault(s["frame"], []).append({
                    "track_id": pid, "label": result.label_for(shot_id, local_id, s["frame"]), "box": s["box"],
                    "kp": _kp_array_from_sample(s["keypoints"]),
                    "sublabel": f"trk {local_id} / shot {shot_id}" if show_local_ids else None})
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cut_frames = {f: sid for sid, f in shot_starts.items() if sid > 0}
    shot, banner_left, banner, writer, idx = 0, 0, None, None, start_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in cut_frames:
            shot, banner_left, banner = cut_frames[idx], int(fps), f"CAMERA CUT -> shot {cut_frames[idx]}"
        img = draw_tracking_frame(frame, by_frame.get(idx, []), t=None, banner=banner if banner_left > 0 else None)
        banner_left -= 1
        cv2.putText(img, f"shot {shot}  frame {idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        _try_h264(out_path)
        return out_path
    return None


def run_identity_tracking(video_path, out_prefix, cfg=None, tracker="botsort", model_name="yolo11n-pose.pt", start=0.0, end=0.0,
                          conf=None, imgsz=None, device="", save_video=True, show_local_ids=False, label_sheets=True,
                          model=None, extractor=None, number_reader=None):
    """Local tracking on every frame + camera-cut detection + global player identity. `model`, `extractor` and `number_reader`
    can be injected (tests use fakes). Returns a dict with the result, per-shot timelines, global timeline and file paths."""
    import global_identity as gi
    cfg = cfg or gi.IdentityConfig().apply_profile()
    tracker_yaml = TRACKERS.get(tracker, tracker)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if model is None:
        model = load_tracking_model(model_name, tracker_yaml)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    kwargs = {"persist": True, "tracker": tracker_yaml, "verbose": False}
    if conf is not None:
        kwargs["conf"] = conf
    if imgsz:
        kwargs["imgsz"] = imgsz
    if device:
        kwargs["device"] = device
    mgr = gi.GlobalPlayerIdentityManager(cfg, extractor=extractor or gi.make_extractor(cfg.reid_backend, device or "cpu"),
                                         number_reader=number_reader)
    shot_detector, clock = gi.ShotDetector(cfg), FrameClock(fps)
    shot_id, timelines, shot_starts, n_read, n_frames = 0, {0: TrackingTimeline(fps, size)}, {0: start_frame}, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = start_frame + n_read
        t = clock.time_of(idx, cap.get(cv2.CAP_PROP_POS_MSEC))
        if end and t > end:
            break
        n_read += 1
        is_cut, info = shot_detector.update(frame)
        if is_cut:  # new shot: restart the LOCAL tracker; the global gallery is untouched
            shot_id += 1
            action = gi.reset_local_tracker(model)
            mgr.start_shot(shot_id, idx, t, info, action)
            timelines[shot_id] = TrackingTimeline(fps, size)
            shot_starts[shot_id] = idx
        results = model.track(frame, **kwargs)
        dets = parse_tracking_result(results[0]) if results else []
        timelines[shot_id].add_frame(idx, t, dets)
        mgr.observe_frame(idx, t, shot_id, frame, dets)
        n_frames += 1
        if n_frames % 100 == 0:
            print(f"  processed {n_frames} frames (shot {shot_id})...", file=sys.stderr)
    cap.release()
    result = mgr.finalize()
    gtl = build_global_timeline(result, timelines)
    video_out = None
    if save_video:
        video_out = render_identity_video(video_path, f"{out_prefix}_identity.mp4", timelines, result, start_frame, shot_starts, show_local_ids)
    sheets = []
    if label_sheets:
        sheets = result.write_label_sheets(f"{out_prefix}_identity_labels")
    meta = {"video": str(video_path), "tracker": tracker_yaml, "model": model_name, "frames_processed": n_frames, "fps": fps,
            "frame_size": list(size), "start_frame": start_frame, "shot_start_frames": shot_starts,
            "time_source_counts": clock.counts, "created": datetime.datetime.now().isoformat(timespec="seconds")}
    return {"result": result, "timelines": timelines, "global_timeline": gtl, "meta": meta, "video_out": video_out, "label_sheets": sheets}


def write_identity_json(run, path):
    r = run["result"]
    doc = {"schema_version": 1, "meta": run["meta"],
           "ids_explained": {"player_id": "permanent global identity (the only thing to show as 'who')",
                             "local_track_id": "BoT-SORT's label inside ONE shot: restarts at every cut, NOT an identity",
                             "shot_id": "increases at every detected camera cut"},
           "identity": r.to_dict(),
           "global_timeline": run["global_timeline"],
           "local_tracking_per_shot": {str(sid): {"frame_stats": tl.to_dict()["frame_stats"], "track_stats": tl.track_stats()}
                                       for sid, tl in run["timelines"].items()}}
    Path(path).write_text(json.dumps(json_safe(doc), indent=1, allow_nan=False), encoding="utf-8")
    return doc


def format_identity_summary(doc):
    ident, gt = doc["identity"], doc["global_timeline"]
    lines = [f"Shots (camera cuts + 1): {len(ident['shots'])}"]
    for s in ident["shots"]:
        cut = s.get("cut") or {}
        lines.append(f"  shot {s['shot_id']}: starts frame {s['start_frame']}" + (f" (cut evidence: diff {cut.get('diff')}, histogram distance {cut.get('hist_distance')})" if cut else ""))
    lines.append(f"Tracklets (one local track inside one shot, split if a local-ID swap was suspected): {len(ident['tracklets'])}; "
                 f"status counts: {ident['segment_status_counts']}")
    lines.append(f"Permanent players created: {len(gt['players'])}")
    for pid, p in gt["players"].items():
        lines.append(f"  {p['label']}: shots {p['shots_seen']}, {len(p['tracklets'])} tracklet(s), {p['first_t']:.2f}s to {p['last_t']:.2f}s, {p['n_samples']} samples")
    unmatched = len(gt["unassigned_tracklets"])
    lines.append(f"Tracklets left UNMATCHED (not merged, shown as 'P? (unmatched)'): {unmatched}")
    lines.append(f"Merge suggestions to confirm by eye (never merged automatically): {len(ident['merge_suggestions'])}")
    lines.append("Identity accuracy is UNKNOWN until you inspect the annotated video and score it with identity_eval.py.")
    return lines


def run_identity_cli(args):
    import global_identity as gi
    d = json.loads(Path(args.identity_config).read_text(encoding="utf-8")) if args.identity_config else {}
    if args.reid_backend:
        d["reid_backend"] = args.reid_backend
    cfg = gi.IdentityConfig.from_dict(d)
    tracker = "botsort" if args.tracker == "both" else args.tracker
    stem = Path(args.video).stem
    print(f"Global player identity: local tracker = {tracker}, appearance = {cfg.reid_backend}", file=sys.stderr)
    try:
        run = run_identity_tracking(args.video, f"{stem}_{tracker}", cfg, tracker, args.track_model, args.start, args.end,
                                    args.track_conf, args.track_imgsz, args.track_device, not args.track_no_video,
                                    args.show_local_ids, not args.identity_no_sheets)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        sys.exit(f"Identity tracking failed: {exc}")
    json_path = f"{stem}_{tracker}_identity.json"
    doc = write_identity_json(run, json_path)
    print("\n".join(format_identity_summary(doc)))
    if args.identity_truth:
        import identity_eval
        truth = json.loads(Path(args.identity_truth).read_text(encoding="utf-8"))
        print("\n".join(identity_eval.format_report(identity_eval.evaluate_identity(doc, truth))))
    print(f"Saved: {json_path}" + (f", {run['video_out']}" if run["video_out"] else "") +
          (f", label sheets in {stem}_{tracker}_identity_labels/" if run["label_sheets"] else ""), file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Describe and analyze any video with Meta Muse Spark.")
    p.add_argument("video", help="Path to the video file")
    p.add_argument("--context", default="", help='Optional hint, e.g. "football skill tutorial"')
    p.add_argument("--max-frames", type=int, default=24, help="Frames to sample (default 24)")
    p.add_argument("--max-width", type=int, default=1024, help="Resize frames to this width (default 1024)")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--end", type=float, default=0.0, help="End time in seconds (default: end of video)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default {DEFAULT_MODEL})")
    p.add_argument("--out", default="", help="Report path (default: <video name>_report.html in the current directory)")
    p.add_argument("--dry-run", action="store_true", help="Extract and save frames only; no API call")
    p.add_argument("--window", type=float, default=0.5, help="Seconds before/after each finding to show as evidence (default 0.5)")
    p.add_argument("--no-verify", action="store_true", help="Skip the close-up re-check of each finding (faster, less reliable)")
    p.add_argument("--clips", action="store_true", help="Also save a 3-second clip per finding (needs ffmpeg)")
    p.add_argument("--findings", default="", help="Re-render a saved <video>_findings.json without calling the API")
    p.add_argument("--uniform", action="store_true", help="Sample frames evenly instead of spending more of them where the picture moves")
    p.add_argument("--track", action="store_true", help="Also run persistent multi-player tracking on EVERY frame (needs ultralytics); saves <video>_<tracker>_tracked.mp4 and _tracking.json")
    p.add_argument("--track-only", action="store_true", help="Run tracking and stop: no Muse Spark calls and no API key needed")
    p.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack", "both"], help="Tracker to test (default botsort)")
    p.add_argument("--track-model", default="yolo11n-pose.pt", help="Ultralytics pose model (default yolo11n-pose.pt)")
    p.add_argument("--track-conf", type=float, default=None, help="Detection confidence for tracking. Default: Ultralytics' tracking default (0.1), which trackers expect")
    p.add_argument("--track-imgsz", type=int, default=None, help="Inference size, e.g. 1280 for small far-away players (slower)")
    p.add_argument("--track-stride", type=int, default=1, help="Track every Nth frame (default 1 = every frame; larger values hurt tracking)")
    p.add_argument("--track-device", default="", help="Ultralytics device, e.g. cpu, 0, mps")
    p.add_argument("--track-truth", default="", help="Hand-labelled JSON to count ID switches and missing frames (format in test_tracking.py)")
    p.add_argument("--track-no-video", action="store_true", help="Do not write the annotated video")
    p.add_argument("--identity", action="store_true", help="GLOBAL PLAYER IDENTITY: BoT-SORT local tracking + camera-cut detection + permanent player_id across shots (needs ultralytics; stops after identity output, no API calls)")
    p.add_argument("--identity-config", default="", help="JSON of IdentityConfig overrides: ReID thresholds, ambiguity margin, track memory, cut detection (see global_identity.py)")
    p.add_argument("--reid-backend", default="", help="handcrafted (default, NOT a trained ReID model) | onnx:<model.onnx> | torchreid:<name>:<weights.pth>")
    p.add_argument("--identity-truth", default="", help="hand-labelled identity JSON to score the result (format in identity_eval.py)")
    p.add_argument("--identity-no-sheets", action="store_true", help="Do not write tracklet label sheets")
    p.add_argument("--show-local-ids", action="store_true", help="Also draw each box's local tracker ID in small grey text ('trk N / shot S')")
    p.add_argument("--pose", default="auto", choices=["auto", "yolo", "mediapipe", "none"], help="Pose backend for measured angles (default auto: YOLO, else MediaPipe)")
    args = p.parse_args()

    if not Path(args.video).is_file():
        sys.exit(f"File not found: {args.video}")

    if args.identity:
        run_identity_cli(args)
        return

    if args.track or args.track_only:
        run_tracking_cli(args)
        if args.track_only:
            return

    if args.findings:  # re-render a saved findings file; no API call
        saved = json.loads(Path(args.findings).read_text(encoding="utf-8"))
        write_report(args, Path(args.video).stem, saved.get("description", ""), clean_analysis(saved["analysis"]), cv2.VideoCapture(args.video), load_pose(args.pose))
        return

    frames, duration = extract_frames(
        args.video, args.max_frames, args.max_width, args.start, args.end, not args.uniform
    )
    print(f"Video length: {duration:.1f}s | frames sampled: {len(frames)}", file=sys.stderr)

    if args.dry_run:
        out_dir = Path(f"{Path(args.video).stem}_frames")
        out_dir.mkdir(exist_ok=True)
        for i, (t, b64, *_) in enumerate(frames, 1):
            (out_dir / f"frame_{i:02d}_{t:.1f}s.jpg").write_bytes(base64.b64decode(b64))
        print(f"Saved frames to {out_dir}", file=sys.stderr)
        return

    if not os.environ.get("MODEL_API_KEY"):
        sys.exit("Set the MODEL_API_KEY environment variable first (Meta Model API key).")

    from openai import OpenAI  # imported here so --dry-run works without the SDK

    client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
    context_line = f"Context from the user: {args.context}\n" if args.context else ""

    stem = Path(args.video).stem
    desc_file = Path(f"{stem}_description.txt")
    if desc_file.is_file():
        print("Using cached video description...", file=sys.stderr)
        description = desc_file.read_text(encoding="utf-8")
    else:
        print("Pass 1/2: describing the video...", file=sys.stderr)
        description = ask(
            client, args.model, frames,
            DESCRIBE_PROMPT.format(context_line=context_line), max_tokens=4000,
        )
        desc_file.write_text(description, encoding="utf-8")

    raw_file = Path(f"{stem}_raw.txt")
    if raw_file.is_file():
        try:
            analysis = clean_analysis(parse_json(raw_file.read_text(encoding="utf-8")))
            print("Loaded existing analysis from raw output...", file=sys.stderr)
            raw = raw_file.read_text(encoding="utf-8")
        except Exception:
            analysis = None
    else:
        analysis = None

    if analysis is None:
        print("Pass 2/2: finding movement problems...", file=sys.stderr)
        raw = ask(
            client, args.model, frames,
            ANALYZE_PROMPT.format(description=description, context_line=context_line, visual_doc=VISUAL_DOC),
            max_tokens=8000,
        )
        raw_file.write_text(raw, encoding="utf-8")
        try:
            analysis = clean_analysis(parse_json(raw))
        except ValueError:
            sys.exit(f"Model output was not valid JSON; saved to {stem}_raw.txt")

    cap = cv2.VideoCapture(args.video)
    ctx = load_pose(args.pose)
    findings = analysis.get("findings", [])
    if not args.no_verify:
        for i, f in enumerate(findings, 1):
            print(f"Checking finding {i}/{len(findings)} on close-up frames...", file=sys.stderr)
            try:
                v = verify_finding(client, args.model, cap, f, measured=measured_text(cap, ctx, float(f["t"]), colour_group(f.get("top_colour"))))
                f["verified"] = bool(v.get("confirmed"))
                f["verify_note"] = v.get("note", "")
                if f["verified"] and v.get("best_t") is not None:
                    f["t"] = float(v["best_t"])
            except Exception as exc:  # keep the finding, but mark it as unchecked
                print(f"  could not check: {exc}", file=sys.stderr)
    print("Equipment check on zoomed crops...", file=sys.stderr)
    crops = equipment_crops(cap, ctx, spaced_times(cap))
    try:
        eq = parse_json(ask(client, args.model, crops, EQUIP_PROMPT, max_tokens=4000))
        analysis["equipment"] = eq.get("equipment", [])
        analysis["equipment_not_visible"] = eq.get("not_visible", [])
    except ValueError:
        print("Equipment answer was not valid JSON; skipping that section.", file=sys.stderr)
    Path(f"{stem}_findings.json").write_text(
        json.dumps({"description": description, "analysis": analysis}, indent=2), encoding="utf-8"
    )
    write_report(args, stem, description, analysis, cap, ctx, crops)


if __name__ == "__main__":
    main()
