"""HandGesture — finger-to-finger connection lines between two hands.

When both hands are visible, 5 lines appear connecting matching fingertips
(thumb<->thumb, index<->index, ...). Each line's color hue is driven by its
rotation angle, so turning the hands cycles colors through the rainbow.
You can stretch the lines by moving hands apart.

Requirements: pip install opencv-python mediapipe numpy
Run: python HandGesture.py
Controls: 's' = save snapshot, 'h' = toggle skeleton, 'q' = quit
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time

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

SMOOTH_ALPHA = 0.55  # exponential smoothing for fingertip positions

# Per-finger hue offset (degrees) so the 5 lines aren't identical at angle 0
FINGER_HUE_OFFSETS = [0, 60, 120, 200, 280]


def hsv_to_bgr(h_deg, s=1.0, v=1.0):
    """Convert HSV (h in degrees 0-360) to BGR uint8 tuple."""
    h = (h_deg % 360) / 2.0  # OpenCV H is 0-179
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
        cv2.line(frame, pa, pb, (90, 90, 90), 1, cv2.LINE_AA)
    for i in range(21):
        pt = landmark_to_px(lm[i], w, h)
        cv2.circle(frame, pt, 2, (180, 180, 180), -1, cv2.LINE_AA)


def draw_glow_line(frame, p1, p2, color, core_thickness=3):
    """Multi-pass glow line: wide faint outer, medium mid, bright core."""
    overlay = frame.copy()
    cv2.line(overlay, p1, p2, color, core_thickness + 14, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    overlay = frame.copy()
    cv2.line(overlay, p1, p2, color, core_thickness + 7, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.line(frame, p1, p2, color, core_thickness + 2, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.5 + 130)) for c in color)
    cv2.line(frame, p1, p2, bright, core_thickness, cv2.LINE_AA)


def draw_endpoint_node(frame, pt, color, radius=8):
    overlay = frame.copy()
    cv2.circle(overlay, pt, radius + 6, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
    bright = tuple(min(255, int(c * 0.4 + 150)) for c in color)
    cv2.circle(frame, pt, max(1, radius // 2), bright, -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    window_name = "HandGesture - Finger Lines"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    smoothed_tips = {}  # (label, fi) -> (x, y) smoothed
    show_skeleton = True
    save_msg = ""
    save_msg_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        # Slightly darken background so glows pop
        frame = cv2.addWeighted(frame, 0.78, np.zeros_like(frame), 0.22, 0)

        per_hand_tips = {}  # "Left"/"Right" -> {fi: (x,y)}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
                raw_label = hand_info.classification[0].label
                # Mirror flip means we invert the label
                label = "Right" if raw_label == "Left" else "Left"

                if show_skeleton:
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

        # Draw the 5 connecting lines when both hands present
        if "Left" in per_hand_tips and "Right" in per_hand_tips:
            left_tips = per_hand_tips["Left"]
            right_tips = per_hand_tips["Right"]

            for fi in range(5):
                if fi not in left_tips or fi not in right_tips:
                    continue
                p1 = left_tips[fi]
                p2 = right_tips[fi]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.hypot(dx, dy)
                # Angle in degrees, full 0-360 by mapping symmetric line to unique direction
                # Use atan2 doubled so 180-degree flips don't change color (line is undirected)
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad * 2.0)
                hue = (angle_deg + FINGER_HUE_OFFSETS[fi]) % 360

                color = hsv_to_bgr(hue, s=1.0, v=1.0)

                # Thickness scales mildly with length so stretched lines feel taut
                core = 2 + int(min(length, 600) / 200)
                draw_glow_line(frame, p1, p2, color, core_thickness=core)

                # Endpoint nodes
                draw_endpoint_node(frame, p1, color, radius=7)
                draw_endpoint_node(frame, p2, color, radius=7)

                # Finger label near midpoint
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2 - 10
                cv2.putText(frame, FINGER_NAMES[fi], (mx - 22, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (240, 240, 240), 1, cv2.LINE_AA)

            cv2.putText(frame, "5 lines active", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Show both hands to connect fingers",
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
            fname = f"hand_lines_{int(now)}.png"
            cv2.imwrite(fname, frame)
            save_msg = f"Saved {fname}"
            save_msg_time = now
            print(save_msg)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
