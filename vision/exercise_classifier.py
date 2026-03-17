"""
exercise_classifier.py - Tiplo Exercise Classifier
Detects: Squat | Arm Raise | Bird Dog | Deep Lunge | Back Extension

Stability features:
- Angle smoothing over rolling window
- Hysteresis thresholds (different enter/exit values)
- Confidence scoring per landmark visibility
- Orientation detection with margin
"""

import numpy as np
from collections import deque
from utils.angles import calculate_angle

# ── Landmark indices ─────────────────────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

# ── Per-joint rolling smoothers ──────────────────────────────────────────────
_smoothers = {}

def smooth(key, value, window=6):
    if value is None:
        return None
    if key not in _smoothers:
        _smoothers[key] = deque(maxlen=window)
    _smoothers[key].append(value)
    return float(np.mean(_smoothers[key]))

# ── Helpers ──────────────────────────────────────────────────────────────────
def _vis(lm, idx, threshold=0.45):
    return lm[idx].visibility > threshold

def _y(lm, idx):  return lm[idx].y
def _x(lm, idx):  return lm[idx].x
def _pt(lm, idx): return (lm[idx].x, lm[idx].y)

def _avg_y(lm, *indices):
    vals = [_y(lm, i) for i in indices if _vis(lm, i)]
    return float(np.mean(vals)) if vals else None

def _confidence(lm, *indices):
    return float(np.mean([lm[i].visibility for i in indices]))

def _knee_angle(lm, side):
    """Returns smoothed knee angle for given side, or None."""
    if side == 'L':
        hp, kn, ank = L_HIP, L_KNEE, L_ANKLE
    else:
        hp, kn, ank = R_HIP, R_KNEE, R_ANKLE

    if _vis(lm, hp) and _vis(lm, kn) and _vis(lm, ank):
        a = calculate_angle(_pt(lm, hp), _pt(lm, kn), _pt(lm, ank))
        return smooth(f'knee_{side}', a)

    # Ankle not visible — fallback to hip angle (shoulder→hip→knee)
    sh = L_SHOULDER if side == 'L' else R_SHOULDER
    if _vis(lm, sh) and _vis(lm, hp) and _vis(lm, kn):
        a = calculate_angle(_pt(lm, sh), _pt(lm, hp), _pt(lm, kn))
        return smooth(f'hip_{side}', a)

    return None


# ── Body orientation ─────────────────────────────────────────────────────────
def get_body_orientation(lm):
    """Returns: 'upright' | 'all_fours' | 'prone' | 'unknown'"""
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    if sh_y is None or hp_y is None:
        return 'unknown'

    diff = hp_y - sh_y  # positive = hips below shoulders

    if diff > 0.22:
        return 'upright'
    if abs(diff) <= 0.22:
        return 'all_fours' if hp_y < 0.60 else 'prone'
    return 'unknown'


# ── EXERCISE 1: Squat ────────────────────────────────────────────────────────
# Hysteresis tuned for stable calibration + reliable rep entry/exit.
SQUAT_ENTER    = 140
SQUAT_EXIT     = 158
SQUAT_SYMMETRY = 40   # allow mild asymmetry from camera perspective

_squat_active = False

def detect_squat(lm):
    """
    Squat: upright, both knees bend symmetrically.
    Distinguished from lunge by symmetry of both knee angles.
    Returns: (phase, avg_angle, debug)
    """
    global _squat_active

    l_angle = _knee_angle(lm, 'L')
    r_angle = _knee_angle(lm, 'R')

    if l_angle is None and r_angle is None:
        return 'partial', None, "stand back — need full body visible"

    # Use whichever sides we have
    available = [a for a in [l_angle, r_angle] if a is not None]
    avg_angle = float(np.mean(available))

    # Symmetry check — if only one side visible, skip symmetry
    is_symmetric = True
    if l_angle and r_angle:
        is_symmetric = abs(l_angle - r_angle) < SQUAT_SYMMETRY

    debug = (f"L:{f'{l_angle:.0f}' if l_angle else '--'} "
             f"R:{f'{r_angle:.0f}' if r_angle else '--'} "
             f"avg:{avg_angle:.0f} sym:{is_symmetric}")

    # Hysteresis
    if not _squat_active and avg_angle < SQUAT_ENTER and is_symmetric:
        _squat_active = True
    elif _squat_active and avg_angle > SQUAT_EXIT:
        _squat_active = False

    if _squat_active:
        return 'active', avg_angle, debug
    # Keep a narrow partial band to avoid getting stuck in partial at rest.
    elif avg_angle < (SQUAT_ENTER + 8):
        return 'partial', avg_angle, debug
    else:
        return 'rest', avg_angle, debug


# ── EXERCISE 2: Arm Raise ────────────────────────────────────────────────────
# Active: wrist(s) rise above shoulder level
# Hysteresis: enter when wrist_y < shoulder_y - 0.06, exit when wrist_y > shoulder_y + 0.02
ARM_ENTER_MARGIN = 0.06   # wrist must be this much above shoulder
ARM_EXIT_MARGIN  = 0.02   # wrist falls back to shoulder level + margin

_arm_active = False

def detect_arm_raise(lm):
    """
    Arm Raise: standing, one or both wrists rise above shoulder height.
    Returns: (phase, side, debug)
    """
    global _arm_active

    orientation = get_body_orientation(lm)
    if orientation == 'all_fours' or orientation == 'prone':
        return 'wrong_position', None, f"orient:{orientation} — stand up for arm raise"
    # 'unknown' is OK — arm raise only needs shoulders and wrists

    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    if sh_y is None:
        return None, None, "shoulders not visible"

    enter_thresh = sh_y - ARM_ENTER_MARGIN
    exit_thresh  = sh_y + ARM_EXIT_MARGIN

    l_raised = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < enter_thresh
    r_raised = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < enter_thresh
    l_down   = not _vis(lm, L_WRIST) or _y(lm, L_WRIST) > exit_thresh
    r_down   = not _vis(lm, R_WRIST) or _y(lm, R_WRIST) > exit_thresh

    debug = (f"sh_y:{sh_y:.2f} thresh:{enter_thresh:.2f} "
             f"L_wrist:{_y(lm,L_WRIST):.2f} R_wrist:{_y(lm,R_WRIST):.2f} "
             f"L_raised:{l_raised} R_raised:{r_raised}")

    # Determine side
    if l_raised and r_raised:
        side = 'both'
    elif l_raised:
        side = 'left'
    elif r_raised:
        side = 'right'
    else:
        side = None

    # Hysteresis
    if not _arm_active and (l_raised or r_raised):
        _arm_active = True
    elif _arm_active and l_down and r_down:
        _arm_active = False

    if _arm_active:
        return 'active', side, debug
    else:
        return 'rest', None, debug


# ── EXERCISE 3: Bird Dog ─────────────────────────────────────────────────────
_bird_dog_active = False

def detect_bird_dog(lm):
    """
    On all fours, extend opposite arm + leg simultaneously.
    Returns: (phase, side, confidence, debug)
    """
    global _bird_dog_active

    orientation = get_body_orientation(lm)
    if orientation not in ('all_fours', 'prone'):
        return 'wrong_position', None, 0.0, f"orient:{orientation} — get on hands+knees"

    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    if sh_y is None or hp_y is None:
        return None, None, 0.0, "landmarks missing"

    arm_thresh = sh_y - 0.07
    leg_thresh = 0.14

    l_arm_ext = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < arm_thresh
    r_arm_ext = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < arm_thresh
    l_leg_ext = _vis(lm, L_ANKLE) and abs(_y(lm, L_ANKLE) - hp_y) < leg_thresh
    r_leg_ext = _vis(lm, R_ANKLE) and abs(_y(lm, R_ANKLE) - hp_y) < leg_thresh

    debug = (f"L_arm:{l_arm_ext} R_arm:{r_arm_ext} "
             f"L_leg:{l_leg_ext} R_leg:{r_leg_ext}")

    right_arm_left_leg = r_arm_ext and l_leg_ext
    left_arm_right_leg = l_arm_ext and r_leg_ext
    any_extension      = l_arm_ext or r_arm_ext or l_leg_ext or r_leg_ext

    conf = _confidence(lm, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

    if not _bird_dog_active and (right_arm_left_leg or left_arm_right_leg):
        _bird_dog_active = True
    elif _bird_dog_active and not any_extension:
        _bird_dog_active = False

    if _bird_dog_active:
        side = 'R_arm+L_leg' if right_arm_left_leg else 'L_arm+R_leg'
        return 'active', side, min(conf + 0.2, 1.0), debug
    elif any_extension:
        return 'partial', 'one_limb', conf * 0.6, debug
    else:
        return 'rest', None, conf, debug


# ── EXERCISE 4: Deep Lunge ───────────────────────────────────────────────────
LUNGE_ENTER    = 115
LUNGE_EXIT     = 145
LUNGE_SYMMETRY = 30   # lunges are ASYMMETRIC — knees must differ by this much

# Auto-detect should be stricter than locked mode to avoid flickering between
# Squat and Deep Lunge on small left-right knee differences.
AUTO_LUNGE_ASYMMETRY = 38
AUTO_LUNGE_FRONT_MAX = 138

_lunge_active = False

def detect_deep_lunge(lm):
    """
    One knee bends deeply while the other stays relatively straight.
    Key distinction from squat: asymmetric knee angles.
    Returns: (phase, side, angle, debug)
    """
    global _lunge_active

    orientation = get_body_orientation(lm)
    # Only block prone — deep lunge also converges hips/shoulders like all_fours
    if orientation == 'prone':
        return 'wrong_position', None, None, "lie flat detected — stand up for lunge"
    # 'unknown', 'upright', 'all_fours' → fall through to angle detection

    l_angle = _knee_angle(lm, 'L')
    r_angle = _knee_angle(lm, 'R')

    if l_angle is None and r_angle is None:
        return None, None, None, "no knee landmarks"

    # Need both angles to confirm asymmetry
    if l_angle and r_angle:
        asymmetry = abs(l_angle - r_angle)
        front_side  = 'L' if l_angle < r_angle else 'R'
        front_angle = min(l_angle, r_angle)
    else:
        # Only one side visible — can still detect
        asymmetry   = 999
        front_side  = 'L' if l_angle else 'R'
        front_angle = l_angle or r_angle

    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    hip_drop = (hp_y - sh_y) if (sh_y and hp_y) else 0

    debug = (f"L:{f'{l_angle:.0f}' if l_angle else '--'} "
             f"R:{f'{r_angle:.0f}' if r_angle else '--'} "
             f"asym:{asymmetry:.0f} hip_drop:{hip_drop:.2f}")

    is_lunge = front_angle < LUNGE_ENTER and (asymmetry > LUNGE_SYMMETRY or l_angle is None or r_angle is None)

    # hip_drop removed: same reason as squat — 0 when hips off-screen
    if not _lunge_active and is_lunge:
        _lunge_active = True
    elif _lunge_active and front_angle > LUNGE_EXIT:
        _lunge_active = False

    if _lunge_active:
        return 'active', front_side, front_angle, debug
    elif front_angle < LUNGE_EXIT:
        return 'partial', front_side, front_angle, debug
    else:
        return 'rest', front_side, front_angle, debug


# ── EXERCISE 5: Back Extension ───────────────────────────────────────────────
BACK_EXT_ENTER = 0.10
BACK_EXT_EXIT  = 0.05

_back_ext_active = False

def detect_back_extension(lm):
    """
    Prone, lift chest off floor.
    Returns: (phase, lift_amount, debug)
    """
    global _back_ext_active

    orientation = get_body_orientation(lm)
    if orientation not in ('prone', 'all_fours'):
        return 'wrong_position', 0, f"orient:{orientation} — lie face down"

    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    if sh_y is None or hp_y is None:
        return None, 0, "landmarks missing"

    lift = hp_y - sh_y

    debug = f"sh_y:{sh_y:.2f} hp_y:{hp_y:.2f} lift:{lift:.3f}"

    if not _back_ext_active and lift > BACK_EXT_ENTER:
        _back_ext_active = True
    elif _back_ext_active and lift < BACK_EXT_EXIT:
        _back_ext_active = False

    if _back_ext_active:
        return 'active', lift, debug
    elif lift > BACK_EXT_EXIT:
        return 'partial', lift, debug
    else:
        return 'rest', lift, debug


# ── Main classifier ──────────────────────────────────────────────────────────
def classify_exercise(lm, target_exercise=None):
    orientation = get_body_orientation(lm)

    # ── Locked to specific exercise ───────────────────────────────────────
    if target_exercise == 'squat':
        phase, angle, debug = detect_squat(lm)
        return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                'orientation': orientation, 'debug': debug}

    if target_exercise == 'arm_raise':
        phase, side, debug = detect_arm_raise(lm)
        return {'exercise': 'Arm Raise', 'phase': phase, 'side': side,
                'orientation': orientation, 'debug': debug}

    if target_exercise == 'bird_dog':
        phase, side, conf, debug = detect_bird_dog(lm)
        return {'exercise': 'Bird Dog', 'phase': phase, 'side': side,
                'confidence': conf, 'orientation': orientation, 'debug': debug}

    if target_exercise == 'deep_lunge':
        phase, side, angle, debug = detect_deep_lunge(lm)
        return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                'angle': angle, 'orientation': orientation, 'debug': debug}

    if target_exercise == 'back_extension':
        phase, lift, debug = detect_back_extension(lm)
        return {'exercise': 'Back Extension', 'phase': phase, 'lift': lift,
                'orientation': orientation, 'debug': debug}

    # ── Auto-detect by orientation ────────────────────────────────────────
    if orientation == 'upright':
        # Distinguish squat vs lunge vs arm raise
        l_angle = _knee_angle(lm, 'L')
        r_angle = _knee_angle(lm, 'R')

        knee_vals = [a for a in (l_angle, r_angle) if a is not None]
        knees_mostly_straight = (not knee_vals) or (min(knee_vals) > 152)

        # Check arm raise first (can happen at any knee angle)
        sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
        if sh_y and knees_mostly_straight:
            l_raised = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < sh_y - 0.06
            r_raised = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < sh_y - 0.06
            if l_raised or r_raised:
                phase, side, debug = detect_arm_raise(lm)
                return {'exercise': 'Arm Raise', 'phase': phase, 'side': side,
                        'orientation': orientation, 'debug': debug}

        # Squat vs lunge: check symmetry
        if l_angle and r_angle:
            asymmetry = abs(l_angle - r_angle)
            front_angle = min(l_angle, r_angle)
            if asymmetry > AUTO_LUNGE_ASYMMETRY and front_angle < AUTO_LUNGE_FRONT_MAX:
                phase, side, angle, debug = detect_deep_lunge(lm)
                return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                        'angle': angle, 'orientation': orientation, 'debug': debug}
            else:
                phase, angle, debug = detect_squat(lm)
                return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                        'orientation': orientation, 'debug': debug}

        # Default upright = squat
        phase, angle, debug = detect_squat(lm)
        return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                'orientation': orientation, 'debug': debug}

    if orientation == 'all_fours':
        # Deep squats can temporarily look like all-fours when hips and shoulders
        # converge in y. If knees are clearly bent, keep evaluating squat/lunge
        # before falling back to bird dog.
        l_angle = _knee_angle(lm, 'L')
        r_angle = _knee_angle(lm, 'R')
        knee_vals = [a for a in (l_angle, r_angle) if a is not None]

        if knee_vals and min(knee_vals) < SQUAT_EXIT:
            if l_angle and r_angle:
                asymmetry = abs(l_angle - r_angle)
                front_angle = min(l_angle, r_angle)
                if asymmetry > AUTO_LUNGE_ASYMMETRY and front_angle < AUTO_LUNGE_FRONT_MAX:
                    phase, side, angle, debug = detect_deep_lunge(lm)
                    return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                            'angle': angle, 'orientation': orientation, 'debug': debug}

            phase, angle, debug = detect_squat(lm)
            return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                    'orientation': orientation, 'debug': debug}

        phase, side, conf, debug = detect_bird_dog(lm)
        return {'exercise': 'Bird Dog', 'phase': phase, 'side': side,
                'confidence': conf, 'orientation': orientation, 'debug': debug}

    if orientation == 'prone':
        # Squats/lunges can occasionally look prone from camera perspective.
        # If knees are visible and clearly bent, prefer lower-body classifiers.
        l_angle = _knee_angle(lm, 'L')
        r_angle = _knee_angle(lm, 'R')
        knee_vals = [a for a in (l_angle, r_angle) if a is not None]

        if knee_vals and min(knee_vals) < 165:
            if l_angle and r_angle:
                asymmetry = abs(l_angle - r_angle)
                front_angle = min(l_angle, r_angle)
                if asymmetry > AUTO_LUNGE_ASYMMETRY and front_angle < AUTO_LUNGE_FRONT_MAX:
                    phase, side, angle, debug = detect_deep_lunge(lm)
                    return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                            'angle': angle, 'orientation': orientation, 'debug': debug}

            phase, angle, debug = detect_squat(lm)
            return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                    'orientation': orientation, 'debug': debug}

        phase, lift, debug = detect_back_extension(lm)
        return {'exercise': 'Back Extension', 'phase': phase, 'lift': lift,
                'orientation': orientation, 'debug': debug}

    if orientation == 'unknown':
        # Fallback for imperfect camera framing: still try lower-body detection
        # from knee angles before giving up to Detecting.
        l_angle = _knee_angle(lm, 'L')
        r_angle = _knee_angle(lm, 'R')
        knee_vals = [a for a in (l_angle, r_angle) if a is not None]

        if knee_vals:
            if l_angle and r_angle:
                asymmetry = abs(l_angle - r_angle)
                front_angle = min(l_angle, r_angle)
                if asymmetry > AUTO_LUNGE_ASYMMETRY and front_angle < AUTO_LUNGE_FRONT_MAX:
                    phase, side, angle, debug = detect_deep_lunge(lm)
                    return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                            'angle': angle, 'orientation': orientation, 'debug': debug}

            phase, angle, debug = detect_squat(lm)
            return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                    'orientation': orientation, 'debug': debug}

    return {'exercise': 'Detecting...', 'phase': None,
            'orientation': orientation, 'debug': f'orient:{orientation}'}

# ── Session reset — call this between sessions ────────────────────────────────
def reset_hysteresis():
    """
    Resets all module-level hysteresis flags.
    Must be called at the start of every new session.
    Without this, _squat_active=True from session 1 will block session 2.
    """
    global _squat_active, _arm_active, _bird_dog_active, _lunge_active, _back_ext_active
    global _smoothers
    _squat_active     = False
    _arm_active       = False
    _bird_dog_active  = False
    _lunge_active     = False
    _back_ext_active  = False
    _smoothers.clear()   # also clear rolling angle history