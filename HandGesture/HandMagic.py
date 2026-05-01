"""HandMagic — energy beams between fingertips of both hands.

When both hands are visible, glowing plasma/electric arcs flow between the
fingertips. Thumb-thumb, index-index, etc. The beams jitter, branch, and
pulse based on motion. No geometric shapes, no fireballs — just pure energy.

Requirements: pip install opencv-python mediapipe numpy
Run: python HandMagic.py
Controls: 'q' to quit
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

TIP_IDS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Each finger pair has a base hue (degrees). Hue drifts over time + with motion.
FINGER_BASE_HUES = [200, 280, 320, 30, 90]  # cyan, purple, magenta, orange, green

SMOOTH_ALPHA = 0.55


def hsv_to_bgr(h_deg, s=1.0, v=1.0):
    h = (h_deg % 360) / 2.0
    hsv = np.uint8([[[int(h), int(s * 255), int(v * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def landmark_to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))


def draw_hand_skeleton(frame, hand_lm, w, h):
    lm = hand_lm.landmark
    for a, b in HAND_BONES:
        pa = landmark_to_px(lm[a], w, h)
        pb = landmark_to_px(lm[b], w, h)
        cv2.line(frame, pa, pb, (70, 70, 80), 1, cv2.LINE_AA)


def jittered_path(p1, p2, segments, jitter):
    """Return a list of points along p1->p2 with perpendicular noise (lightning)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy / length, dx / length  # perpendicular unit

    pts = [p1]
    for i in range(1, segments):
        t = i / segments
        bx = p1[0] + dx * t
        by = p1[1] + dy * t
        # Jitter falls off near endpoints (shape: sin)
        falloff = math.sin(t * math.pi)
        offset = random.uniform(-jitter, jitter) * falloff
        pts.append((int(bx + nx * offset), int(by + ny * offset)))
    pts.append(p2)
    return pts


def draw_beam(frame, p1, p2, color, intensity=1.0, branches=True):
    """Draw a single plasma beam with multi-pass glow + lightning jitter."""
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if length < 5:
        return

    segments = max(8, int(length / 18))
    jitter = min(18, length * 0.06) * intensity

    main_path = jittered_path(p1, p2, segments, jitter)
    pts_arr = np.array(main_path, dtype=np.int32).reshape(-1, 1, 2)

    # Outer glow halo
    overlay = frame.copy()
    cv2.polylines(overlay, [pts_arr], False, color, 18, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18 * intensity, frame, 1 - 0.18 * intensity, 0, frame)

    # Mid glow
    overlay = frame.copy()
    cv2.polylines(overlay, [pts_arr], False, color, 9, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4 * intensity, frame, 1 - 0.4 * intensity, 0, frame)

    # Main bolt
    cv2.polylines(frame, [pts_arr], False, color, 4, cv2.LINE_AA)

    # White-hot core
    core_color = tuple(min(255, int(c * 0.3 + 200)) for c in color)
    cv2.polylines(frame, [pts_arr], False, core_color, 2, cv2.LINE_AA)

    # Random forking branches
    if branches and random.random() < 0.7 * intensity:
        n_branches = random.randint(1, 2)
        for _ in range(n_branches):
            idx = random.randint(2, len(main_path) - 3)
            origin = main_path[idx]
            branch_len = random.uniform(20, 55) * intensity
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            angle += random.uniform(-1.0, 1.0)
            end = (
                int(origin[0] + math.cos(angle) * branch_len),
                int(origin[1] + math.sin(angle) * branch_len),
            )
            bsegs = max(3, int(branch_len / 10))
            bpath = jittered_path(origin, end, bsegs, branch_len * 0.15)
            barr = np.array(bpath, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [barr], False, color, 2, cv2.LINE_AA)
            cv2.polylines(frame, [barr], False, core_color, 1, cv2.LINE_AA)


def draw_endpoint_orb(frame, pt, color, base_radius=10, pulse=0.0):
    radius = int(base_radius + 3 * math.sin(pulse))
    overlay = frame.copy()
    cv2.circle(overlay, pt, radius + 12, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.3 + 180)) for c in color)
    cv2.circle(frame, pt, max(2, radius // 2), bright, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, max(1, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    window_name = "HandMagic - Energy Beams"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    smoothed_tips = {}
    prev_tips = {}  # for motion intensity
    motion_amount = {fi: 0.0 for fi in range(5)}

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()
        t_elapsed = now - start_time

        # Darken background heavily so beams glow
        frame = cv2.addWeighted(frame, 0.45, np.zeros_like(frame), 0.55, 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        per_hand_tips = {}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
                raw_label = hand_info.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"
                draw_hand_skeleton(frame, hand_lm, w, h)

                tips = {}
                for fi, tid in enumerate(TIP_IDS):
                    raw = landmark_to_px(hand_lm.landmark[tid], w, h)
                    key = (label, fi)
                    if key in smoothed_tips:
                        sx = SMOOTH_ALPHA * raw[0] + (1 - SMOOTH_ALPHA) * smoothed_tips[key][0]
                        sy = SMOOTH_ALPHA * raw[1] + (1 - SMOOTH_ALPHA) * smoothed_tips[key][1]
                    else:
                        sx, sy = raw
                    smoothed_tips[key] = (sx, sy)
                    tips[fi] = (int(sx), int(sy))
                per_hand_tips[label] = tips

        if "Left" in per_hand_tips and "Right" in per_hand_tips:
            for fi in range(5):
                if fi not in per_hand_tips["Left"] or fi not in per_hand_tips["Right"]:
                    continue
                p1 = per_hand_tips["Left"][fi]
                p2 = per_hand_tips["Right"][fi]

                # Track motion: change in distance/position drives intensity
                pkey = ("L", fi)
                qkey = ("R", fi)
                motion = 0.0
                if pkey in prev_tips and qkey in prev_tips:
                    motion = (math.hypot(p1[0] - prev_tips[pkey][0], p1[1] - prev_tips[pkey][1]) +
                              math.hypot(p2[0] - prev_tips[qkey][0], p2[1] - prev_tips[qkey][1]))
                prev_tips[pkey] = p1
                prev_tips[qkey] = p2

                # Smooth motion -> intensity
                motion_amount[fi] = 0.7 * motion_amount[fi] + 0.3 * motion
                intensity = min(1.6, 0.7 + motion_amount[fi] * 0.04)

                # Hue drifts with time and motion
                hue = (FINGER_BASE_HUES[fi] + t_elapsed * 25 + motion_amount[fi] * 2) % 360
                color = hsv_to_bgr(hue, s=1.0, v=1.0)

                draw_beam(frame, p1, p2, color, intensity=intensity, branches=True)

                pulse = t_elapsed * 4 + fi
                draw_endpoint_orb(frame, p1, color, base_radius=9, pulse=pulse)
                draw_endpoint_orb(frame, p2, color, base_radius=9, pulse=pulse + math.pi)

            cv2.putText(frame, "Energy linked", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 220),
                        2, cv2.LINE_AA)
        else:
            # Single-hand fallback: faint sparks at each fingertip
            for label, tips in per_hand_tips.items():
                for fi, pt in tips.items():
                    hue = (FINGER_BASE_HUES[fi] + t_elapsed * 25) % 360
                    color = hsv_to_bgr(hue, s=1.0, v=1.0)
                    pulse = t_elapsed * 4 + fi
                    draw_endpoint_orb(frame, pt, color, base_radius=7, pulse=pulse)

            cv2.putText(frame, "Show both hands to channel energy",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (180, 180, 180), 2, cv2.LINE_AA)

        # Drop stale smoothed tips when hand disappears
        active = set()
        for label, tips in per_hand_tips.items():
            for fi in tips:
                active.add((label, fi))
        for k in list(smoothed_tips.keys()):
            if k not in active:
                del smoothed_tips[k]

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
