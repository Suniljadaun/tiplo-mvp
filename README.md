# Tiplo — AI Physiotherapy Adherence System
### MVP Sprint · 3 Weeks · Private Repo

> Tiplo uses computer vision to watch patients perform prescribed physiotherapy exercises at home, counts repetitions automatically, and reports adherence data to clinicians — without ever recording or storing raw video.

---

## Team

| Person | Role | Area |
|--------|------|-------|
| **Adam** | CEO | Product, patient interviews, exercise selection |
| **Sunil** | Computer Vision + Frontend | Camera, pose detection, rep counting, desktop UI |
| **Sameer** | Backend + AI | Cloud infrastructure, dashboard, voice guidance, data storage |
| **Ali** | UI/UX | Visual design, mobile UI (Week 2+) |

---

## Project Structure

```
tiplo-mvp/
│
├── vision/
│   ├── rep_counter.py          # Main app — Flask server, camera loop, rep state machine
│   ├── exercise_classifier.py  # Detects 5 exercises from MediaPipe landmarks
│   └── pose_detector.py        # Early prototype (reference only)
│
├── utils/
│   └── angles.py               # Joint angle calculator (dot product maths)
│
├── ui/
│   └── desktop.html            # 4-screen desktop UI (Select → Guidance → Session → Summary)
│
├── backend/                    # Sameer's folder — cloud, dashboard, voice guidance
│
├── shared/
│   └── data_models.py          # Agreed data structures — Patient, Session, Reps (WIP)
│
├── data/
│   └── session_log.json        # Local session output (not committed)
│
├── pose_landmarker_lite.task   # MediaPipe model file — download separately (see below)
├── requirements.txt
└── README.md
```

---

## Quick Start (Sunil's CV System)

### 1. Prerequisites

- Python 3.12
- Webcam connected
- Ubuntu / Linux (tested on Ubuntu 24)

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/tiplo-mvp.git
cd tiplo-mvp

# Create virtual environment
python3.12 -m venv tiplo
source tiplo/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the pose model

The MediaPipe model file is too large for git. Download it manually:

```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Place it in the root of the project (`tiplo-mvp/pose_landmarker_lite.task`).

### 4. Run

```bash
# Auto-detect mode (camera recognises exercise automatically)
python vision/rep_counter.py

# Or lock to a specific exercise
python vision/rep_counter.py squat
python vision/rep_counter.py arm_raise
python vision/rep_counter.py bird_dog
python vision/rep_counter.py deep_lunge
python vision/rep_counter.py back_extension
```

### 5. Open in browser

```
http://localhost:5000
```

---

## How the CV System Works

### The Pipeline

Every camera frame (30fps) goes through this sequence:

```
Camera frame
    → MediaPipe (33 body landmarks + segmentation mask)
    → exercise_classifier.py (orientation → angles → exercise + phase)
    → rep_counter.py (state machine → rep counted)
    → Privacy frame (silhouette + skeleton, NO raw video)
    → Browser via MJPEG stream
```

### Exercises Detected

| Exercise | Detection Method | Key Threshold |
|----------|-----------------|---------------|
| Squat | Symmetric knee bend, hip drop | Knee angle < 130° |
| Arm Raise | Wrist above shoulder | Wrist y < shoulder y − 6% |
| Bird Dog | All-fours, opposite arm+leg extend | Diagonal extension |
| Deep Lunge | Asymmetric knees (one bent, one straight) | Front knee < 115°, asymmetry > 30° |
| Back Extension | Prone, chest lift | Shoulder lift > 10% frame height |

### Rep Counting States

```
CALIBRATING → READY → ACTIVE → COOLDOWN → READY (next rep)
```

A rep is counted when:
1. System has seen a clean rest position (calibrated)
2. Active phase confirmed for N consecutive frames
3. Rest phase confirmed after active
4. Cooldown timer has elapsed (prevents double-counting)

### Privacy Architecture

- Raw camera pixels are **never sent to the browser**
- MediaPipe runs entirely **on-device**
- Browser receives only: silhouette outline + skeleton lines drawn on a black canvas
- No video is stored anywhere

---

## Live API Endpoints

The Flask server exposes these endpoints. **Sameer — these are what your dashboard connects to.**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `ui/desktop.html` |
| `/video` | GET | MJPEG privacy stream. Param: `?exercise=squat` |
| `/state` | GET | Live JSON state (poll every 350ms) |
| `/reset` | GET | Reset all counters for new session |

### `/state` Response Shape

```json
{
  "exercise":         "Squat",
  "phase":            "active",
  "reps":             12,
  "exercise_reps":    { "Squat": 8, "Deep Lunge": 4 },
  "status":           "Good — hold the position",
  "calibrated":       true,
  "progress":         0.75,
  "orientation":      "upright",
  "candidate":        null,
  "candidate_frames": 0
}
```

**This is the integration contract between Sunil's CV and Sameer's backend.**  
At session end, `exercise_reps` maps directly to the `Session.reps` data model.

---

## Data Models (Shared Contract)

> These live in `shared/data_models.py`. Everyone imports from here.

```python
Patient:
  - patient_id, name
  - prescribed_exercises: { "Squat": 10, "Arm Raise": 8 }   # reps per session
  - sessions: List[Session]

Session:
  - session_id, patient_id, date
  - reps: List[RepData]          # Sunil produces this
  - pain_score: int (1–10)       # patient inputs this
  - adherence_score: float       # Sameer calculates this
  - duration_seconds: int

RepData:
  - exercise: str
  - count: int
  - timestamp: datetime
```

Adherence score formula (Sameer to confirm):
```
adherence = actual_reps / prescribed_reps   # per exercise, averaged across session
```

---

## Week-by-Week Plan

### Week 1 ✅ — Foundation (Complete)
- [x] Camera working, MediaPipe pose detection
- [x] Privacy mode — silhouette + skeleton, no raw video
- [x] 5 exercises detected with hysteresis and smoothing
- [x] 4-state rep counting machine
- [x] Exercise locking (no flickering labels)
- [x] Desktop UI — Select, Guidance, Session, Summary screens
- [x] Live `/state` API endpoint

### Week 2 — Core Product
**Sunil**
- [ ] Stable silhouette tracking improvements
- [ ] Gamification — score, level, streak display
- [ ] Performance metric output (form quality signal)
- [ ] Output `Session` object at end of session

**Sameer**
- [ ] GCP/AWS backend structure
- [ ] REST endpoint to receive `Session` data from CV
- [ ] Clinician dashboard — adherence %, pain score trend, rep counts
- [ ] AI voice guidance — predefined scripts + TTS output
- [ ] Secure patient data storage

**Ali**
- [ ] Mobile UI (portrait, touch-friendly)
- [ ] Design system — colours, typography, components

### Week 3 — Integration + Polish
- [ ] CV → Backend data flow live end-to-end
- [ ] Clinician dashboard showing real patient data
- [ ] Voice guidance playing during exercise
- [ ] Mobile camera overlay working
- [ ] Demo-ready build

---

## Branching Convention

```bash
main              # always working, demo-ready
sunil/feature     # e.g. sunil/gamification
sameer/feature    # e.g. sameer/dashboard
ali/feature       # e.g. ali/mobile-ui
```

**Never push directly to `main`.** Open a pull request, get one review, then merge.

---

## Dependencies

```
mediapipe
opencv-python
tensorflow
flask
pillow
numpy
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Notes for Sameer

1. The `/state` endpoint is your data source. Poll it at ~350ms during a session.
2. At session end, `exercise_reps` dict is ready to populate a `Session` object.
3. For the dashboard, `adherence_score = actual / prescribed` per exercise.
4. Voice guidance trigger: when `phase` transitions to `"active"`, play the cue for `exercise`.
5. To test without the CV system running, mock `/state` with the JSON shape above.

## Notes for Ali

1. The UI lives in `ui/desktop.html` — single self-contained file, no build step.
2. The three live data sources are `/video` (image), `/state` (JSON), `/reset` (action).
3. Mobile UI designs go in `ui/mobile.html` — same API, different layout.
4. Color palette and component patterns are documented in the HTML CSS variables.

---

## Contact

Raise issues in GitHub Issues. Tag the relevant person (`@sunil`, `@sameer`, `@ali`).  
For urgent blockers — WhatsApp group.