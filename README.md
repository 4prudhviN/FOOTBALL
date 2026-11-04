# Football Move & Movement Analyzer ⚽📊

An AI-powered video analysis platform for football coaching and movement breakdown. It combines **Meta's Muse Spark** vision-language models with **YOLOv11 pose estimation** and computer vision tracking to evaluate player biomechanics, decision making, tactical positioning, and drill performance.

---

## 🎥 Demo Video

Watch the project in action:

[![Watch Demo Video](https://img.shields.io/badge/Demo%20Video-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1y3_ySBlHGCCj4zjrQcfTwkq5BLGQcC0k/view?usp=sharing)

👉 **[Click here to watch the full demo video](https://drive.google.com/file/d/1y3_ySBlHGCCj4zjrQcfTwkq5BLGQcC0k/view?usp=sharing)**

---

## 🚀 Key Features

- **Multi-Pass AI Video Understanding:**
  - **Pass 1:** Scene overview, participant identification, chronological timeline segmentation, and annotation detection.
  - **Pass 2:** Deep biomechanical and tactical evaluation across every player (posture, head position, knee angles, balance, weight transfer, and ball contact).
- **YOLOv11 Biomechanical Grounding:**
  - Automatic multi-person pose estimation (`yolo11n-pose`) and object detection (`yolo11n`).
  - Quantitative angle, distance, and joint position measurements.
- **Verification & Evidence Close-ups:**
  - Automated zoom and re-inspection of flagged moments to eliminate false positives.
  - Equipment and gear detection (boots, bibs, balls, cones).
- **Interactive Visual Reports:**
  - Outputs self-contained interactive HTML reports with side-by-side timestamps, keyframe overlays, corrective drill instructions, and coaching visuals.
- **Web UI & REST API:**
  - Modern Flask web application with drag-and-drop upload, live progress tracking, and report history.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask
- **Vision & LLM:** Meta Model API (`muse-spark-1.1` / `muse-spark-1.2`), OpenAI Python SDK
- **Computer Vision & Tracking:** OpenCV, Ultralytics YOLOv11 (Pose + Detection)
- **Frontend:** HTML5, CSS3, Vanilla JS

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/4prudhviN/FOOTBALL.git
cd FOOTBALL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install ultralytics   # for YOLO pose & tracking support
```

### 3. Configure API Key
Copy the example environment file and add your Meta Model API key:
```bash
cp env.example .env
```
Edit `.env`:
```env
MODEL_API_KEY=your_meta_model_api_key
```

---

## 💻 Usage

### Option 1: Web Interface
Start the local server:
```bash
python server.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to upload videos, track analysis progress in real time, and review reports.

### Option 2: Command Line (CLI)
Run analysis directly on any video file:
```bash
python video_analyzer.py path/to/clip.mp4
```

#### Additional CLI Options:
- **Set video context:**
  ```bash
  python video_analyzer.py clip.mp4 --context "1v1 attacking drill"
  ```
- **Analyze a specific time window:**
  ```bash
  python video_analyzer.py clip.mp4 --start 8 --end 20 --max-frames 24
  ```
- **Dry run (extract frames only without API calls):**
  ```bash
  python video_analyzer.py clip.mp4 --dry-run
  ```
- **Re-render report from existing findings JSON:**
  ```bash
  python video_analyzer.py clip.mp4 --findings clip_findings.json
  ```

---

## 📋 Generated Outputs

- `<video_name>_report.html` - Self-contained interactive report with visual cards and diagrams.
- `<video_name>_findings.json` - Structured JSON containing all findings, timestamps, and coaching cues.
- `<video_name>_description.txt` - Text breakdown of the video timeline and scene details.

---

## 📄 License

This project is licensed under the MIT License.
