"""HandGesture — finger-to-finger connection lines between two hands.

When both hands are visible, 5 lines appear connecting matching fingertips
(thumb<->thumb, index<->index, ...). Each line's color hue is driven by its
rotation angle, so turning the hands cycles colors through the rainbow.

Requirements: pip install opencv-python mediapipe numpy
Run:          python HandGesture.py [--camera N]
Controls:     's' = save snapshot, 'h' = toggle skeleton, 'q' = quit
"""

import argparse
import colorsys
import math
import time

import cv2
import mediapipe as mp
import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
SMOOTH_ALPHA    = 0.55   # EMA weight for new fingertip position (0=frozen, 1=raw)
BG_DARKEN       = 0.78   # fraction of live frame kept (rest is darkened)
MIN_DETECT_CONF = 0.6
MIN_TRACK_CONF  = 0.6

TIP_IDS           = [4, 8, 12, 16, 20]
FINGER_NAMES      = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_HUE_OFFSETS = [0, 60, 120, 200, 280]  # per-finger hue offset (degrees)

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


def draw_hand_skeleton(frame, overlay, hand_lm, w: int, h: int) -> None:
    lm = hand_lm.landmark
    for a, b in HAND_BONES:
        pa, pb = landmark_to_px(lm[a], w, h), landmark_to_px(lm[b], w, h)
        cv2.line(frame, pa, pb, (90, 90, 90), 1, cv2.LINE_AA)
    for point in lm:
        cv2.circle(frame, landmark_to_px(point, w, h), 2, (180, 180, 180), -1, cv2.LINE_AA)


def draw_glow_line(frame, overlay, p1, p2, color, core_thickness: int = 3) -> None:
    """Multi-pass glow: wide faint outer → medium mid → bright core.
    Reuses a pre-allocated overlay buffer to avoid repeated frame.copy() calls.
    """
    np.copyto(overlay, frame)
    cv2.line(overlay, p1, p2, color, core_thickness + 14, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    np.copyto(overlay, frame)
    cv2.line(overlay, p1, p2, color, core_thickness + 7, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.line(frame, p1, p2, color, core_thickness + 2, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.5 + 130)) for c in color)
    cv2.line(frame, p1, p2, bright, core_thickness, cv2.LINE_AA)


def draw_endpoint_node(frame, overlay, pt, color, radius: int = 8) -> None:
    np.copyto(overlay, frame)
    cv2.circle(overlay, pt, radius + 6, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.4 + 150)) for c in color)
    cv2.circle(frame, pt, max(1, radius // 2), bright, -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="HandGesture — finger connection lines")
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

    window_name = "HandGesture - Finger Lines"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    smoothed_tips: dict = {}
    show_skeleton       = True
    save_msg            = ""
    save_msg_time       = 0.0

    # Allocated on first frame once dimensions are known
    overlay = None
    black   = None

    fps       = 0.0
    prev_time = time.time()

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
            fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Darken background so glows pop (reuses pre-alloc'd black)
            cv2.addWeighted(frame, BG_DARKEN, black, 1.0 - BG_DARKEN, 0, frame)

            per_hand_tips: dict = {}

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
                    raw_label = hand_info.classification[0].label
                    label     = "Right" if raw_label == "Left" else "Left"

                    if show_skeleton:
                        draw_hand_skeleton(frame, overlay, hand_lm, w, h)

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

            # Draw the 5 connecting lines when both hands present
            if "Left" in per_hand_tips and "Right" in per_hand_tips:
                left_tips  = per_hand_tips["Left"]
                right_tips = per_hand_tips["Right"]

                for fi in range(5):
                    if fi not in left_tips or fi not in right_tips:
                        continue
                    p1, p2 = left_tips[fi], right_tips[fi]

                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    length = math.hypot(dx, dy)

                    # atan2 doubled so 180° flips don't change color (line is undirected)
                    hue   = (math.degrees(math.atan2(dy, dx) * 2.0) + FINGER_HUE_OFFSETS[fi]) % 360
                    color = hsv_to_bgr(hue)

                    # Thickness scales with stretch so long lines feel taut
                    core = 2 + int(min(length, 600) / 200)
                    draw_glow_line(frame, overlay, p1, p2, color, core_thickness=core)
                    draw_endpoint_node(frame, overlay, p1, color, radius=7)
                    draw_endpoint_node(frame, overlay, p2, color, radius=7)

                    mx = (p1[0] + p2[0]) // 2
                    my = (p1[1] + p2[1]) // 2 - 10
                    cv2.putText(frame, FINGER_NAMES[fi], (mx - 22, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

                cv2.putText(frame, "5 lines active", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Show both hands to connect fingers",
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
                fname         = f"hand_lines_{int(now)}.png"
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
