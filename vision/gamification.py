"""
gamification.py — Tiplo Gamification Engine
============================================
Handles:
  - Per-rep scoring (base points + form quality bonus)
  - Cumulative level system (persists across sessions)
  - Daily streak tracking (persists across sessions)
  - Per-rep form metric capture (used by Sameer's performance output)

Persistence file: data/tiplo_progress.json
"""

import json
import os
from datetime import datetime, date


# ── File paths ────────────────────────────────────────────────────────────────
_DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PROGRESS_FILE = os.path.join(_DATA_DIR, "tiplo_progress.json")

os.makedirs(_DATA_DIR, exist_ok=True)


# ── Points per exercise ────────────────────────────────────────────────────────
# Harder / less common exercises get more points to encourage variety
BASE_POINTS = {
    "Squat":          10,
    "Arm Raise":       8,
    "Bird Dog":       15,   # hardest — requires coordination
    "Deep Lunge":     12,
    "Back Extension": 10,
}

# ── Level thresholds (total cumulative score) ─────────────────────────────────
LEVELS = [
    (0,    1, "Beginner"),
    (100,  2, "Starter"),
    (300,  3, "Active"),
    (600,  4, "Consistent"),
    (1000, 5, "Dedicated"),
    (1500, 6, "Advanced"),
    (2500, 7, "Expert"),
    (4000, 8, "Elite"),
]


# ── Load / save progress ──────────────────────────────────────────────────────
def _load() -> dict:
    if os.path.exists(_PROGRESS_FILE):
        try:
            with open(_PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total_score":   0,
        "level":         1,
        "level_name":    "Beginner",
        "streak_days":   0,
        "last_session_date": None,
        "sessions_total": 0,
    }


def _save(data: dict):
    with open(_PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────
class GamificationEngine:
    """
    One instance per app session. Loaded once at startup.
    Call record_rep() on every rep. Call end_session() to finalise.
    """

    def __init__(self):
        self._progress     = _load()
        self._session_score = 0
        self._session_start = datetime.now()
        self._level_up_flag = False  # true for one rep after levelling up
        self._rep_log: list[dict] = []  # full rep-level detail for Sameer

        # Track streak at session open
        self._progress["streak_days"] = _calc_streak(
            self._progress.get("last_session_date")
        )

    # ── Called once per counted rep ───────────────────────────────────────────
    def record_rep(self, exercise: str, form_metrics: dict) -> dict:
        """
        exercise     : e.g. "Squat"
        form_metrics : dict of raw values captured during active phase
                       e.g. {"min_angle": 118.5, "symmetry": 0.91, ...}

        Returns a dict with all gamification data for this rep — goes into /state
        """
        form_score   = _compute_form_score(exercise, form_metrics)
        base_pts     = BASE_POINTS.get(exercise, 10)
        form_bonus   = round(form_score * 5)          # 0–5 bonus points
        rep_score    = base_pts + form_bonus

        self._session_score            += rep_score
        self._progress["total_score"]  += rep_score

        old_level = self._progress["level"]
        new_level, name = _calc_level(self._progress["total_score"])
        self._progress["level"]      = new_level
        self._progress["level_name"] = name
        self._level_up_flag          = (new_level > old_level)

        # Log for session_data output
        self._rep_log.append({
            "exercise":    exercise,
            "timestamp":   datetime.now().isoformat(),
            "rep_score":   rep_score,
            "form_score":  round(form_score, 3),
            "form_metrics": form_metrics,
        })

        _save(self._progress)

        return {
            "rep_score":     rep_score,
            "session_score": self._session_score,
            "total_score":   self._progress["total_score"],
            "level":         self._progress["level"],
            "level_name":    self._progress["level_name"],
            "streak_days":   self._progress["streak_days"],
            "form_score":    round(form_score, 2),
            "level_up":      self._level_up_flag,
        }

    # ── Called when user presses Done ─────────────────────────────────────────
    def end_session(self, exercise_reps: dict) -> dict:
        """
        Returns the full gamification summary for this session.
        Also increments session count + saves last session date for streak.
        """
        self._progress["sessions_total"]    = self._progress.get("sessions_total", 0) + 1
        self._progress["last_session_date"] = date.today().isoformat()
        _save(self._progress)

        return {
            "session_score": self._session_score,
            "total_score":   self._progress["total_score"],
            "level":         self._progress["level"],
            "level_name":    self._progress["level_name"],
            "streak_days":   self._progress["streak_days"],
            "sessions_total": self._progress["sessions_total"],
        }

    # ── Current snapshot (for /state polling) ────────────────────────────────
    @property
    def snapshot(self) -> dict:
        return {
            "session_score": self._session_score,
            "total_score":   self._progress["total_score"],
            "level":         self._progress["level"],
            "level_name":    self._progress["level_name"],
            "streak_days":   self._progress["streak_days"],
            "level_up":      self._level_up_flag,
        }

    # ── Rep log (for /session_data endpoint) ─────────────────────────────────
    @property
    def rep_log(self) -> list:
        return self._rep_log

    def clear_level_up(self):
        self._level_up_flag = False


# ── Form scoring formulas ─────────────────────────────────────────────────────
def _compute_form_score(exercise: str, m: dict) -> float:
    """
    Converts raw form metrics into a 0.0–1.0 score.
    Each exercise uses different signals.
    """
    try:
        if exercise == "Squat":
            # depth: 130° = threshold (0.0 score), 90° = perfect (1.0)
            min_angle   = m.get("min_angle", 130)
            depth_score = max(0.0, min(1.0, (130 - min_angle) / 40))
            # symmetry: 0 = perfect symmetry, 25 = threshold
            symmetry_diff = m.get("symmetry_diff", 25)
            sym_score     = max(0.0, 1.0 - symmetry_diff / 25)
            return round((depth_score * 0.6 + sym_score * 0.4), 3)

        elif exercise == "Arm Raise":
            # height above shoulder (normalised screen units)
            # 0.06 = just above (threshold), 0.20 = excellent
            height = m.get("max_height_above_shoulder", 0.06)
            return round(max(0.0, min(1.0, (height - 0.06) / 0.14)), 3)

        elif exercise == "Bird Dog":
            # confidence directly from classifier (0–1)
            return round(max(0.0, min(1.0, m.get("confidence", 0.5))), 3)

        elif exercise == "Deep Lunge":
            # front knee depth: 115° = threshold (0.0), 90° = perfect (1.0)
            min_angle   = m.get("min_front_angle", 115)
            depth_score = max(0.0, min(1.0, (115 - min_angle) / 25))
            return round(depth_score, 3)

        elif exercise == "Back Extension":
            # lift amount: 0.10 = threshold (0.0), 0.22 = excellent (1.0)
            max_lift = m.get("max_lift", 0.10)
            return round(max(0.0, min(1.0, (max_lift - 0.10) / 0.12)), 3)

    except Exception:
        pass

    return 0.5   # fallback


# ── Level calculation ─────────────────────────────────────────────────────────
def _calc_level(total_score: int) -> tuple[int, str]:
    level, name = 1, "Beginner"
    for threshold, lvl, lname in LEVELS:
        if total_score >= threshold:
            level, name = lvl, lname
    return level, name


# ── Streak calculation ────────────────────────────────────────────────────────
def _calc_streak(last_date_str: str | None) -> int:
    if last_date_str is None:
        return 1  # first ever session

    try:
        last = date.fromisoformat(last_date_str)
        today = date.today()
        delta = (today - last).days

        if delta == 0:
            # Already exercised today — load existing streak from file
            progress = _load()
            return progress.get("streak_days", 1)
        elif delta == 1:
            # Exercised yesterday — increment
            progress = _load()
            return progress.get("streak_days", 1) + 1
        else:
            # Missed day(s) — reset
            return 1
    except Exception:
        return 1


# ── Performance aggregator ────────────────────────────────────────────────────
def aggregate_performance(rep_log: list, exercise_reps: dict) -> dict:
    """
    Takes the per-rep log and returns aggregated performance per exercise.
    This is what goes into the /session_data output for Sameer's backend.

    Returns:
    {
      "Squat": {
        "reps": 8,
        "avg_form_score": 0.78,
        "avg_depth_angle": 118.5,
        "avg_symmetry_diff": 8.2
      }, ...
    }
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)

    for entry in rep_log:
        buckets[entry["exercise"]].append(entry)

    result = {}
    for ex, entries in buckets.items():
        agg = {
            "reps":           len(entries),
            "avg_form_score": round(sum(e["form_score"] for e in entries) / len(entries), 3),
        }
        # Add exercise-specific averages
        if ex == "Squat":
            angles = [e["form_metrics"].get("min_angle") for e in entries if e["form_metrics"].get("min_angle")]
            syms   = [e["form_metrics"].get("symmetry_diff") for e in entries if e["form_metrics"].get("symmetry_diff")]
            if angles: agg["avg_depth_angle"]    = round(sum(angles) / len(angles), 1)
            if syms:   agg["avg_symmetry_diff"]  = round(sum(syms)   / len(syms),   1)

        elif ex == "Arm Raise":
            heights = [e["form_metrics"].get("max_height_above_shoulder") for e in entries if e["form_metrics"].get("max_height_above_shoulder")]
            if heights: agg["avg_height_above_shoulder"] = round(sum(heights) / len(heights), 3)

        elif ex == "Bird Dog":
            confs = [e["form_metrics"].get("confidence") for e in entries if e["form_metrics"].get("confidence")]
            if confs: agg["avg_confidence"] = round(sum(confs) / len(confs), 3)

        elif ex == "Deep Lunge":
            angles = [e["form_metrics"].get("min_front_angle") for e in entries if e["form_metrics"].get("min_front_angle")]
            if angles: agg["avg_front_knee_angle"] = round(sum(angles) / len(angles), 1)

        elif ex == "Back Extension":
            lifts = [e["form_metrics"].get("max_lift") for e in entries if e["form_metrics"].get("max_lift")]
            if lifts: agg["avg_max_lift"] = round(sum(lifts) / len(lifts), 3)

        result[ex] = agg

    return result