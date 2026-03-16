"""
rep_counter.py - Tiplo Unified Rep Counter  (Week 2)
All 5 exercises + Gamification + Performance metrics output

Usage:
    python vision/rep_counter.py                   # auto-detect
    python vision/rep_counter.py squat
    python vision/rep_counter.py arm_raise
    python vision/rep_counter.py bird_dog
    python vision/rep_counter.py deep_lunge
    python vision/rep_counter.py back_extension
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request, os, sys, time, uuid, json
from datetime import datetime
from flask import Flask, Response, request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exercise_classifier import classify_exercise
from gamification import GamificationEngine, aggregate_performance

# ── Model ─────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

TARGET        = sys.argv[1] if len(sys.argv) > 1 else None
ACTIVE_TARGET = TARGET

# ── Skeleton ──────────────────────────────────────────────────────────────────
CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

# ── Per-exercise config ───────────────────────────────────────────────────────
EXERCISE_CONFIG = {
    "Squat":          {"active_needed": 6,  "rest_needed": 5,  "lost_grace": 8,  "cooldown_s": 1.2},
    "Arm Raise":      {"active_needed": 5,  "rest_needed": 4,  "lost_grace": 6,  "cooldown_s": 0.8},
    "Bird Dog":       {"active_needed": 8,  "rest_needed": 6,  "lost_grace": 10, "cooldown_s": 1.5},
    "Deep Lunge":     {"active_needed": 6,  "rest_needed": 5,  "lost_grace": 8,  "cooldown_s": 1.2},
    "Back Extension": {"active_needed": 7,  "rest_needed": 6,  "lost_grace": 10, "cooldown_s": 1.5},
    "Detecting...":   {"active_needed": 10, "rest_needed": 8,  "lost_grace": 5,  "cooldown_s": 2.0},
}

# ── Colours per exercise ──────────────────────────────────────────────────────
COLORS = {
    "Squat":          (  0, 200, 255),
    "Arm Raise":      (180,  80, 255),
    "Bird Dog":       (  0, 180, 120),
    "Deep Lunge":     (  0, 140, 220),
    "Back Extension": (140, 220,   0),
    "Detecting...":   ( 80,  80,  80),
}
SKELETON_COLOR = (255, 255, 255)
JOINT_COLOR    = (  0, 255, 180)

# ── State ──────────────────────────────────────────────────────────────────────
state = {
    # Rep counting
    "reps":               0,
    "exercise_reps":      {},
    "phase":              "rest",
    "exercise":           "Detecting...",
    "orientation":        "unknown",
    "candidate_ex":       None,
    "candidate_frames":   0,
    "status":             "Get into position...",
    "active_frames":      0,
    "rest_frames":        0,
    "lost_frames":        0,
    "reached_active":     False,
    "seen_initial_rest":  False,
    "last_rep_time":      0.0,
    "last_landmarks":     None,
    # Form tracking (updated each frame during active phase)
    "active_peak":        {},
    # Session metadata
    "session_id":         str(uuid.uuid4()),
    "session_start":      time.time(),
}

EXERCISE_LOCK_FRAMES = 20

def cfg(key):
    return EXERCISE_CONFIG.get(state["exercise"], EXERCISE_CONFIG["Detecting..."])[key]


# ── Gamification engine (one per app lifetime) ────────────────────────────────
gamification = GamificationEngine()


# ── Privacy frame ─────────────────────────────────────────────────────────────
def build_privacy_frame(frame, seg_mask, landmarks, h, w):
    color  = COLORS.get(state["exercise"], (80, 80, 80))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if seg_mask is not None:
        mask_data    = seg_mask.numpy_view()
        mask_resized = cv2.resize(mask_data, (w, h))
        # ── Stable silhouette: tighter threshold + morphological cleanup ──────
        body_mask = (mask_resized > 0.55).astype(np.uint8)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN,  kernel)

        blurred = cv2.GaussianBlur(body_mask * 255, (31, 31), 0)
        glow    = np.zeros_like(canvas)
        glow[:] = [c // 3 for c in color]
        canvas  = (canvas + glow * (blurred[..., None] / 255.0)).astype(np.uint8)

        silhouette = np.zeros_like(canvas)
        silhouette[body_mask == 1] = color
        canvas = cv2.addWeighted(canvas, 0.35, silhouette, 0.95, 0)

    if landmarks:
        points = {}
        for idx, lm in enumerate(landmarks):
            cx, cy      = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)

        for a, b in CONNECTIONS:
            if a in points and b in points:
                cv2.line(canvas, points[a], points[b], SKELETON_COLOR, 3, cv2.LINE_AA)

        for idx, (cx, cy) in points.items():
            dot = JOINT_COLOR if landmarks[idx].visibility > 0.5 else (70, 70, 70)
            cv2.circle(canvas, (cx, cy), 7, dot,           -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 7, SKELETON_COLOR,  1, cv2.LINE_AA)

    return canvas


# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_hud(canvas, reps, phase, status, exercise, debug_text=""):
    h, w    = canvas.shape[:2]
    color   = COLORS.get(exercise, (80, 80, 80))
    ex_reps = state["exercise_reps"].get(exercise, 0)
    gami    = gamification.snapshot

    # Top bar background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (0, 0, 0), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)

    # Exercise name + rep count
    cv2.putText(canvas, exercise, (20, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{exercise[:3].upper()}: {ex_reps}", (20, 78),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"TOTAL: {reps}", (w - 240, 78),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

    phase_color = {"active": (0, 255, 120), "rest": (80, 80, 80),
                   "partial": (0, 180, 255)}.get(phase, (80, 80, 80))
    cv2.putText(canvas, (phase or "---").upper(), (w - 200, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 2, cv2.LINE_AA)

    # Progress bar
    if state["phase"] == "rest" and state["active_frames"] > 0:
        needed  = cfg("active_needed")
        frac    = min(state["active_frames"] / needed, 1.0)
        bar_px  = int(frac * (w - 40))
        cv2.rectangle(canvas, (20, 96), (w - 20, 106), (30, 30, 30), -1)
        cv2.rectangle(canvas, (20, 96), (20 + bar_px, 106), color, -1)

    # Per-exercise summary
    if state["exercise_reps"]:
        summary = "  |  ".join(f"{k[:3]}:{v}" for k, v in state["exercise_reps"].items())
        cv2.putText(canvas, summary, (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 100), 1, cv2.LINE_AA)

    # Exercise switching bar
    if state["candidate_ex"] and state["candidate_ex"] != state["exercise"]:
        pct        = state["candidate_frames"] / EXERCISE_LOCK_FRAMES
        bar_px     = int(pct * (w - 40))
        cand_color = COLORS.get(state["candidate_ex"], (80, 80, 80))
        cv2.rectangle(canvas, (20, 122), (w - 20, 130), (30, 30, 30), -1)
        cv2.rectangle(canvas, (20, 122), (20 + bar_px, 130), cand_color, -1)
        cv2.putText(canvas, f"Detecting: {state['candidate_ex']}...", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cand_color, 1, cv2.LINE_AA)

    # ── Gamification strip (right side) ──────────────────────────────────────
    score_txt  = f"SCORE {gami['session_score']:4d}"
    level_txt  = f"LVL {gami['level']}  {gami['level_name'].upper()}"
    streak_txt = f"STREAK {gami['streak_days']}d"

    cv2.putText(canvas, score_txt,  (w - 280, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, level_txt,  (w - 280, 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, streak_txt, (w - 280, 154),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 255, 140), 1, cv2.LINE_AA)

    # Level-up flash
    if gami.get("level_up"):
        cv2.putText(canvas, f"LEVEL UP!  Now Level {gami['level']}",
                    (w // 2 - 180, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 220, 60), 3, cv2.LINE_AA)
        gamification.clear_level_up()

    # Bottom bar
    overlay2 = canvas.copy()
    cv2.rectangle(overlay2, (0, h - 52), (w, h), (0, 0, 0), -1)
    canvas[:] = cv2.addWeighted(overlay2, 0.68, canvas, 0.32, 0)

    cv2.putText(canvas, status, (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRIVACY MODE", (w - 225, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 160, 50), 1, cv2.LINE_AA)

    if debug_text:
        cv2.putText(canvas, debug_text[:90], (10, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1, cv2.LINE_AA)


# ── Exercise lock ─────────────────────────────────────────────────────────────
def update_exercise_lock(detected_ex):
    if detected_ex in (None, "Detecting..."):
        return
    if state["phase"] == "active":
        return
    if detected_ex == state["exercise"]:
        state["candidate_ex"]     = None
        state["candidate_frames"] = 0
        return
    if detected_ex != state["candidate_ex"]:
        state["candidate_ex"]     = detected_ex
        state["candidate_frames"] = 1
    else:
        state["candidate_frames"] += 1
    if state["candidate_frames"] >= EXERCISE_LOCK_FRAMES:
        print(f"\n🔄  Switched: {state['exercise']} -> {detected_ex}")
        state["exercise"]         = detected_ex
        state["candidate_ex"]     = None
        state["candidate_frames"] = 0
        state["phase"]            = "rest"
        state["active_frames"]    = 0
        state["rest_frames"]      = 0
        state["reached_active"]   = False
        state["seen_initial_rest"]= False
        state["active_peak"]      = {}


# ── Rep state machine ─────────────────────────────────────────────────────────
def update_state(raw_phase):
    now = time.time()

    if raw_phase == "active":
        state["active_frames"] += 1
        state["rest_frames"]    = 0
        state["lost_frames"]    = 0
    elif raw_phase == "rest":
        state["rest_frames"]   += 1
        state["active_frames"]  = 0
        state["lost_frames"]    = 0
    elif raw_phase in ("partial", "wrong_position"):
        state["lost_frames"] = 0
    else:
        state["lost_frames"] += 1
        if state["lost_frames"] > cfg("lost_grace"):
            state["active_frames"]     = 0
            state["rest_frames"]       = 0
            state["lost_frames"]       = 0
            state["reached_active"]    = False
            state["seen_initial_rest"] = False
            state["phase"]             = "rest"
            state["active_peak"]       = {}
            state["status"]            = "Step into frame — full body needed"
        return

    # CALIBRATING
    if not state["seen_initial_rest"]:
        needed = cfg("rest_needed")
        state["status"] = f"Hold still to calibrate... ({state['rest_frames']}/{needed})"
        if state["rest_frames"] >= needed:
            state["seen_initial_rest"] = True
            state["rest_frames"]       = 0
            state["status"]            = "Ready — begin exercise"
        return

    # COOLDOWN
    if (now - state["last_rep_time"]) < cfg("cooldown_s"):
        state["status"] = "Rest... next rep ready soon"
        return

    # READY → ACTIVE
    if state["phase"] == "rest":
        needed = cfg("active_needed")
        if state["active_frames"] >= needed:
            state["phase"]          = "active"
            state["active_frames"]  = 0
            state["reached_active"] = True
            state["active_peak"]    = {}   # fresh form capture
            state["status"]         = "Exercise detected! Hold it..."
        elif state["active_frames"] > 0:
            state["status"] = f"Keep going... ({state['active_frames']}/{needed})"
        else:
            state["status"] = "Ready — begin exercise"

    # ACTIVE → REST = count rep
    elif state["phase"] == "active":
        needed = cfg("rest_needed")
        if state["rest_frames"] >= needed and state["reached_active"]:
            state["phase"]          = "rest"
            state["rest_frames"]    = 0
            state["reached_active"] = False
            state["reps"]          += 1
            ex = state["exercise"]
            state["exercise_reps"][ex] = state["exercise_reps"].get(ex, 0) + 1
            state["last_rep_time"]  = now
            ex_count = state["exercise_reps"][ex]

            # ── Gamification ─────────────────────────────────────────────────
            gami_result = gamification.record_rep(ex, dict(state["active_peak"]))
            state["active_peak"] = {}

            level_msg = f"  LEVEL UP -> {gami_result['level']}!" if gami_result["level_up"] else ""
            state["status"] = (
                f"Rep {ex_count} — +{gami_result['rep_score']}pts  "
                f"Score:{gami_result['session_score']}{level_msg}"
            )
            print(f"\n✅ {ex} rep {ex_count} | +{gami_result['rep_score']}pts | "
                  f"form:{gami_result['form_score']:.2f} | "
                  f"session:{gami_result['session_score']} | "
                  f"total:{gami_result['total_score']} | "
                  f"lvl:{gami_result['level']} | streak:{gami_result['streak_days']}d")

        elif state["rest_frames"] > 0:
            state["status"] = f"Return to rest... ({state['rest_frames']}/{needed})"
        else:
            state["status"] = "Hold position!"


# ── Form metric tracker (called every frame from generate_frames) ─────────────
def track_form_metrics(cl: dict, landmarks):
    """
    During an active phase, capture the best (peak) form values seen.
    These are passed to gamification.record_rep() when the rep is counted.
    """
    if state["phase"] != "active":
        return

    ex   = cl.get("exercise", "")
    peak = state["active_peak"]

    if ex == "Squat":
        angle = cl.get("angle")
        if angle is not None:
            # Track lowest angle reached (deepest squat)
            peak["min_angle"] = min(peak.get("min_angle", 999), angle)
        # Symmetry: we need L and R angles — read from landmarks if available
        if landmarks:
            try:
                from exercise_classifier import _knee_angle
                la = _knee_angle(landmarks, "L")
                ra = _knee_angle(landmarks, "R")
                if la and ra:
                    diff = abs(la - ra)
                    # Track BEST symmetry (minimum difference) seen
                    peak["symmetry_diff"] = min(peak.get("symmetry_diff", 999), diff)
            except Exception:
                pass

    elif ex == "Arm Raise":
        if landmarks:
            try:
                sh_l = landmarks[11].y
                sh_r = landmarks[12].y
                sh_y = (sh_l + sh_r) / 2
                wl   = landmarks[15].y if landmarks[15].visibility > 0.45 else sh_y
                wr   = landmarks[16].y if landmarks[16].visibility > 0.45 else sh_y
                best_wrist = min(wl, wr)
                above = sh_y - best_wrist   # positive = wrist above shoulder
                peak["max_height_above_shoulder"] = max(
                    peak.get("max_height_above_shoulder", 0), above
                )
            except Exception:
                pass

    elif ex == "Bird Dog":
        conf = cl.get("confidence", 0)
        peak["confidence"] = max(peak.get("confidence", 0), conf)

    elif ex == "Deep Lunge":
        angle = cl.get("angle")
        if angle is not None:
            peak["min_front_angle"] = min(peak.get("min_front_angle", 999), angle)

    elif ex == "Back Extension":
        lift = cl.get("lift", 0)
        if lift is not None:
            peak["max_lift"] = max(peak.get("max_lift", 0), lift)


# ── MediaPipe setup ───────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=1,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
app = Flask(__name__)


# ── Frame generator ───────────────────────────────────────────────────────────
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        seg_mask   = None
        landmarks  = None
        raw_phase  = None
        debug_text = ""
        cl         = {}

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            state["last_landmarks"] = landmarks
            if result.segmentation_masks:
                seg_mask = result.segmentation_masks[0]

            cl          = classify_exercise(landmarks, target_exercise=ACTIVE_TARGET)
            detected_ex = cl.get("exercise", "Detecting...")
            raw_phase   = cl.get("phase")
            debug_text  = cl.get("debug", "")

            # Store orientation for /state endpoint
            state["orientation"] = cl.get("orientation", "unknown")

            if ACTIVE_TARGET:
                state["exercise"] = detected_ex
            else:
                update_exercise_lock(detected_ex)

            if state["candidate_ex"] and state["candidate_ex"] != state["exercise"]:
                pct = int(state["candidate_frames"] / EXERCISE_LOCK_FRAMES * 100)
                debug_text = f"Switching to {state['candidate_ex']}... {pct}%"

            # Track form metrics during active phase
            track_form_metrics(cl, landmarks)

            print(f"{state['exercise']:18} | raw:{str(raw_phase):14} | "
                  f"confirmed:{state['phase']:7} | reps:{state['reps']} | "
                  f"a:{state['active_frames']} r:{state['rest_frames']}",
                  end="\r")
        else:
            landmarks  = state["last_landmarks"]
            seg_mask   = None
            raw_phase  = None
            state["status"] = "Step into frame — full body needed"

        update_state(raw_phase)

        canvas = build_privacy_frame(frame, seg_mask, landmarks, h, w)
        draw_hud(canvas, state["reps"], state["phase"], state["status"],
                 state["exercise"], debug_text)

        canvas = cv2.resize(canvas, (960, 720))
        _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "desktop.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "desktop.html"),
        os.path.join(os.getcwd(), "ui", "desktop.html"),
        os.path.join(os.getcwd(), "desktop.html"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    paths_list = "\n".join(candidates)
    msg = (
        "<pre style=\"font:14px monospace;padding:40px;background:#07090E;color:#FF4466\">"
        "desktop.html not found.\n\nLooked in:\n" + paths_list +
        "\n\nFix:\n  cp ~/Downloads/desktop.html ui/desktop.html\n\nThen refresh.</pre>"
    )
    return msg, 404


@app.route("/video")
def video():
    ex_param = request.args.get("exercise", "").strip()
    if ex_param and ex_param != TARGET:
        key_map = {
            "squat": "squat", "arm_raise": "arm_raise",
            "bird_dog": "bird_dog", "deep_lunge": "deep_lunge",
            "back_extension": "back_extension", "": None,
        }
        global ACTIVE_TARGET
        ACTIVE_TARGET = key_map.get(ex_param, None)
        _reset_state()
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    """Polled by UI every 350ms. Includes rep data + gamification."""
    ex     = state["exercise"]
    needed = EXERCISE_CONFIG.get(ex, EXERCISE_CONFIG["Detecting..."])["active_needed"]
    progress = min(state["active_frames"] / max(needed, 1), 1.0) if state["phase"] == "rest" else 0.0
    gami   = gamification.snapshot

    return jsonify({
        # Rep data
        "exercise":         ex,
        "phase":            state["phase"],
        "orientation":      state.get("orientation", "unknown"),
        "reps":             state["reps"],
        "exercise_reps":    state["exercise_reps"],
        "status":           state["status"],
        "calibrated":       state["seen_initial_rest"],
        "progress":         round(progress, 2),
        "candidate":        state.get("candidate_ex"),
        "candidate_frames": state.get("candidate_frames", 0),
        # Gamification
        "session_score":    gami["session_score"],
        "total_score":      gami["total_score"],
        "level":            gami["level"],
        "level_name":       gami["level_name"],
        "streak_days":      gami["streak_days"],
        "level_up":         gami["level_up"],
    })


@app.route("/session_data")
def session_data():
    """
    Full structured session payload.
    Called by Sameer's backend at session end to store in the database.

    Also writes to data/latest_session.json locally.
    """
    gami_summary = gamification.end_session(state["exercise_reps"])
    performance  = aggregate_performance(gamification.rep_log, state["exercise_reps"])
    duration     = int(time.time() - state["session_start"])

    payload = {
        # Identity
        "session_id":   state["session_id"],
        "patient_id":   "local",          # Sameer replaces with real patient ID
        "timestamp":    datetime.now().isoformat(),
        "duration_seconds": duration,
        # Rep counts
        "total_reps":      state["reps"],
        "exercise_reps":   state["exercise_reps"],
        # Per-exercise form quality
        "performance":     performance,
        # Gamification summary
        "gamification":    gami_summary,
        # Adherence (Sameer fills prescribed_reps from patient profile)
        "prescribed_reps": {},   # populated by backend
        "adherence":       {},   # calculated by backend
    }

    # Save locally for inspection / backup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "latest_session.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n📄 Session data saved: {out_path}")

    return jsonify(payload)


@app.route("/reset")
def reset():
    """Reset session counters. Does NOT reset level/streak (those persist)."""
    global gamification
    gamification     = GamificationEngine()
    state["session_id"]    = str(uuid.uuid4())
    state["session_start"] = time.time()
    _reset_state()
    return jsonify({"ok": True})


def _reset_state():
    state["reps"]             = 0
    state["exercise_reps"]    = {}
    state["phase"]            = "rest"
    state["exercise"]         = "Detecting..."
    state["orientation"]      = "unknown"
    state["active_frames"]    = 0
    state["rest_frames"]      = 0
    state["lost_frames"]      = 0
    state["reached_active"]   = False
    state["seen_initial_rest"]= False
    state["candidate_ex"]     = None
    state["candidate_frames"] = 0
    state["last_rep_time"]    = 0.0
    state["active_peak"]      = {}
    state["status"]           = "Get into position..."


if __name__ == "__main__":
    print(f"\nTiplo — Week 2  |  Mode: {TARGET or 'auto-detect'}")
    print("Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)