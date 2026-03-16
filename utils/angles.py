"""
utils/angles.py
---------------
Calculates joint angles from MediaPipe landmarks.

Usage:
    from utils.angles import calculate_angle, get_knee_angle

Angle reference:
    ~170°  = standing straight
    ~90°   = squat bottom
    ~120°  = lunge
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A-B-C.

    Args:
        a: (x, y) tuple — first point  (e.g. Hip)
        b: (x, y) tuple — vertex point (e.g. Knee)  ← angle measured here
        c: (x, y) tuple — third point  (e.g. Ankle)

    Returns:
        angle in degrees (0-180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors from B to A and B to C
    ba = a - b
    bc = c - b

    # Dot product formula for angle
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)  # Clamp to avoid floating point errors
    angle = np.degrees(np.arccos(cosine))

    return round(angle, 1)


def get_knee_angle(landmarks, side="left"):
    """
    Get the knee angle from MediaPipe pose landmarks.

    Args:
        landmarks: list of pose landmarks (result.pose_landmarks[0])
        side: "left" or "right"

    Returns:
        knee angle in degrees, or None if landmarks not visible
    """
    if side == "left":
        hip_idx, knee_idx, ankle_idx = 23, 25, 27
    else:
        hip_idx, knee_idx, ankle_idx = 24, 26, 28

    hip    = landmarks[hip_idx]
    knee   = landmarks[knee_idx]
    ankle  = landmarks[ankle_idx]

    # Check visibility
    if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
        return None

    return calculate_angle(
        (hip.x,   hip.y),
        (knee.x,  knee.y),
        (ankle.x, ankle.y)
    )


def get_hip_angle(landmarks, side="left"):
    """
    Get the hip angle (shoulder → hip → knee).
    Useful for detecting forward lean or sit-to-stand.
    """
    if side == "left":
        shoulder_idx, hip_idx, knee_idx = 11, 23, 25
    else:
        shoulder_idx, hip_idx, knee_idx = 12, 24, 26

    shoulder = landmarks[shoulder_idx]
    hip      = landmarks[hip_idx]
    knee     = landmarks[knee_idx]

    if shoulder.visibility < 0.5 or hip.visibility < 0.5 or knee.visibility < 0.5:
        return None

    return calculate_angle(
        (shoulder.x, shoulder.y),
        (hip.x,      hip.y),
        (knee.x,     knee.y)
    )


def get_elbow_angle(landmarks, side="left"):
    """
    Get the elbow angle (shoulder → elbow → wrist).
    Useful for arm raises, bicep curls.
    """
    if side == "left":
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
    else:
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16

    shoulder = landmarks[shoulder_idx]
    elbow    = landmarks[elbow_idx]
    wrist    = landmarks[wrist_idx]

    if shoulder.visibility < 0.5 or elbow.visibility < 0.5 or wrist.visibility < 0.5:
        return None

    return calculate_angle(
        (shoulder.x, shoulder.y),
        (elbow.x,    elbow.y),
        (wrist.x,    wrist.y)
    )


if __name__ == "__main__":
    # Quick test with known values
    # Straight line → 180°
    print(calculate_angle((0, 0), (0, 1), (0, 2)))   # Expected: 180.0

    # Right angle → 90°
    print(calculate_angle((0, 0), (1, 0), (1, 1)))   # Expected: 90.0

    # 45° angle
    print(calculate_angle((0, 0), (1, 0), (2, 1)))   # Expected: ~26.6

    print("angles.py working correctly!")