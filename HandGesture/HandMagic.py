"""HandMagic — energy beams between fingertips of both hands.

When both hands are visible, glowing plasma/electric arcs flow between the
fingertips. Thumb-thumb, index-index, etc. The beams jitter, branch, and
pulse based on motion. No geometric shapes, no fireballs — just pure energy.

Requirements: pip install opencv-python mediapipe numpy
Run:          python HandMagic.py [--camera N]
Controls:     's' = save snapshot, 'h' = toggle skeleton, 'q' = quit
"""

import argparse
import colorsys
import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
SMOOTH_ALPHA    = 0.55    # EMA weight for new fingertip position
BG_DARKEN       = 0.45    # fraction of live frame kept (heavier than HandGesture)
MIN_DETECT_CONF = 0.6
MIN_TRACK_CONF  = 0.6

MOTION_DECAY    = 0.7     # how quickly motion intensity decays per frame
MOTION_GAIN     = 0.3     # blend weight for new motion sample
MOTION_CLAMP    = 0.01    # values below this snap to 0 to stop drift
HUE_DRIFT_SPEED = 25      # degrees of hue rotation per second
INTENSITY_BASE  = 0.7     # minimum beam intensity
INTENSITY_SCALE = 0.04    # how strongly motion drives intensity
INTENSITY_MAX   = 1.6

TIP_IDS          = [4, 8, 12, 16, 20]
FINGER_NAMES     = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_BASE_HUES = [200, 280, 320, 30, 90]  # cyan, purple, magenta, orange, green

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# ──────────────────────────────────────────────────────────────────────────────


def hsv_to_bgr(h_deg: float, s: float = 1.0, v: float = 1.0) -> tuple:
    """HSV (h in degrees 0-360) → BGR uint8 tuple. Uses colorsys — no numpy overhead."""
    r, g, b = colorsys.hsv_to_rgb((h_deg / 360.0) % 1.0, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def landmark_to_px(landmark, w: int, h: int) -> tuple:
    return int(landmark.x * w), int(landmark.y * h)


def draw_hand_skeleton(frame, hand_lm, w: int, h: int) -> None:
    lm = hand_lm.landmark
    for a, b in HAND_BONES:
        pa, pb = landmark_to_px(lm[a], w, h), landmark_to_px(lm[b], w, h)
        cv2.line(frame, pa, pb, (70, 70, 80), 1, cv2.LINE_AA)


def jittered_path(p1, p2, segments: int, jitter: float) -> list:
    """Points along p1→p2 with perpendicular noise — produces lightning effect."""
    dx, dy   = p2[0] - p1[0], p2[1] - p1[1]
    length   = math.hypot(dx, dy) + 1e-6
    nx, ny   = -dy / length, dx / length  # perpendicular unit vector

    pts = [p1]
    for i in range(1, segments):
        t      = i / segments
        falloff = math.sin(t * math.pi)   # jitter falls off near endpoints
        offset = random.uniform(-jitter, jitter) * falloff
        pts.append((
            int(p1[0] + dx * t + nx * offset),
            int(p1[1] + dy * t + ny * offset),
        ))
    pts.append(p2)
    return pts


def draw_beam(frame, overlay, p1, p2, color, intensity: float = 1.0, branches: bool = True) -> None:
    """Plasma beam: multi-pass glow + lightning jitter + random branches.
    Reuses a pre-allocated overlay buffer to avoid repeated frame.copy() calls.
    """
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if length < 5:
        return

    segments  = max(8, int(length / 18))
    jitter    = min(18, length * 0.06) * intensity
    main_path = jittered_path(p1, p2, segments, jitter)
    pts_arr   = np.array(main_path, dtype=np.int32).reshape(-1, 1, 2)

    # Outer glow halo
    np.copyto(overlay, frame)
    cv2.polylines(overlay, [pts_arr], False, color, 18, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18 * intensity, frame, 1 - 0.18 * intensity, 0, frame)

    # Mid glow
    np.copyto(overlay, frame)
    cv2.polylines(overlay, [pts_arr], False, color, 9, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4 * intensity, frame, 1 - 0.4 * intensity, 0, frame)

    # Main bolt + white-hot core
    core_color = tuple(min(255, int(c * 0.3 + 200)) for c in color)
    cv2.polylines(frame, [pts_arr], False, color, 4, cv2.LINE_AA)
    cv2.polylines(frame, [pts_arr], False, core_color, 2, cv2.LINE_AA)

    # Random forking branches
    if branches and random.random() < 0.7 * intensity:
        for _ in range(random.randint(1, 2)):
            origin     = main_path[random.randint(2, len(main_path) - 3)]
            branch_len = random.uniform(20, 55) * intensity
            angle      = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + random.uniform(-1.0, 1.0)
            end        = (
                int(origin[0] + math.cos(angle) * branch_len),
                int(origin[1] + math.sin(angle) * branch_len),
            )
            bpath = np.array(
                jittered_path(origin, end, max(3, int(branch_len / 10)), branch_len * 0.15),
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(frame, [bpath], False, color, 2, cv2.LINE_AA)
            cv2.polylines(frame, [bpath], False, core_color, 1, cv2.LINE_AA)


def draw_endpoint_orb(frame, overlay, pt, color, base_radius: int = 10, pulse: float = 0.0) -> None:
    radius = int(base_radius + 3 * math.sin(pulse))
    np.copyto(overlay, frame)
    cv2.circle(overlay, pt, radius + 12, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.3 + 180)) for c in color)
    cv2.circle(frame, pt, max(2, radius // 2), bright, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, max(1, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="HandMagic — energy beam visualizer")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    window_name = "HandMagic - Energy Beams"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    smoothed_tips: dict = {}
    prev_tips: dict     = {}
    motion_amount       = {fi: 0.0 for fi in range(5)}
    show_skeleton       = True
    save_msg            = ""
    save_msg_time       = 0.0

    # Allocated on first frame once dimensions are known
    overlay = None
    black   = None

    fps        = 0.0
    start_time = time.time()
    prev_time  = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # One-time buffer allocation
            if overlay is None:
                overlay = np.empty_like(frame)
                black   = np.zeros_like(frame)

            now       = time.time()
            t_elapsed = now - start_time
            fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now

            # Darken background heavily so beams glow (reuses pre-alloc'd black)
            cv2.addWeighted(frame, BG_DARKEN, black, 1.0 - BG_DARKEN, 0, frame)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            per_hand_tips: dict = {}

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
                    raw_label = hand_info.classification[0].label
                    label     = "Right" if raw_label == "Left" else "Left"

                    if show_skeleton:
                        draw_hand_skeleton(frame, hand_lm, w, h)

                    tips = {}
                    for fi, tid in enumerate(TIP_IDS):
                        raw                = landmark_to_px(hand_lm.landmark[tid], w, h)
                        key                = (label, fi)
                        prev_sx, prev_sy   = smoothed_tips.get(key, raw)
                        sx                 = SMOOTH_ALPHA * raw[0] + (1 - SMOOTH_ALPHA) * prev_sx
                        sy                 = SMOOTH_ALPHA * raw[1] + (1 - SMOOTH_ALPHA) * prev_sy
                        smoothed_tips[key] = (sx, sy)
                        tips[fi]           = (int(sx), int(sy))
                    per_hand_tips[label] = tips

            if "Left" in per_hand_tips and "Right" in per_hand_tips:
                for fi in range(5):
                    if fi not in per_hand_tips["Left"] or fi not in per_hand_tips["Right"]:
                        continue
                    p1 = per_hand_tips["Left"][fi]
                    p2 = per_hand_tips["Right"][fi]

                    pkey, qkey = ("L", fi), ("R", fi)
                    motion = 0.0
                    if pkey in prev_tips and qkey in prev_tips:
                        motion = math.dist(p1, prev_tips[pkey]) + math.dist(p2, prev_tips[qkey])
                    prev_tips[pkey] = p1
                    prev_tips[qkey] = p2

                    # Smooth motion → intensity; clamp near-zero to stop drift
                    m                 = MOTION_DECAY * motion_amount[fi] + MOTION_GAIN * motion
                    motion_amount[fi] = 0.0 if m < MOTION_CLAMP else m
                    intensity         = min(INTENSITY_MAX, INTENSITY_BASE + motion_amount[fi] * INTENSITY_SCALE)

                    hue   = (FINGER_BASE_HUES[fi] + t_elapsed * HUE_DRIFT_SPEED + motion_amount[fi] * 2) % 360
                    color = hsv_to_bgr(hue)

                    draw_beam(frame, overlay, p1, p2, color, intensity=intensity, branches=True)

                    pulse = t_elapsed * 4 + fi
                    draw_endpoint_orb(frame, overlay, p1, color, base_radius=9, pulse=pulse)
                    draw_endpoint_orb(frame, overlay, p2, color, base_radius=9, pulse=pulse + math.pi)

                cv2.putText(frame, "Energy linked", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 220), 2, cv2.LINE_AA)
            else:
                # Single-hand fallback: faint sparks at each fingertip
                for label, tips in per_hand_tips.items():
                    for fi, pt in tips.items():
                        hue   = (FINGER_BASE_HUES[fi] + t_elapsed * HUE_DRIFT_SPEED) % 360
                        color = hsv_to_bgr(hue)
                        draw_endpoint_orb(frame, overlay, pt, color,
                                          base_radius=7, pulse=t_elapsed * 4 + fi)

                cv2.putText(frame, "Show both hands to channel energy",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (180, 180, 180), 2, cv2.LINE_AA)

            # Prune smoothed positions for hands that have left the frame
            active_keys   = {(lbl, fi) for lbl, tips in per_hand_tips.items() for fi in tips}
            smoothed_tips = {k: v for k, v in smoothed_tips.items() if k in active_keys}

            # HUD
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
            if save_msg and now - save_msg_time < 2.0:
                cv2.putText(frame, save_msg, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_skeleton = not show_skeleton
            elif key == ord('s'):
                fname         = f"hand_magic_{int(now)}.png"
                cv2.imwrite(fname, frame)
                save_msg      = f"Saved {fname}"
                save_msg_time = now
                print(save_msg)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
