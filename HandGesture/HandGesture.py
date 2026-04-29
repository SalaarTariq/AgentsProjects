# HandGesture.py — Real-time hand gesture drawing with smooth Catmull-Rom final line
# Requirements: pip install opencv-python mediapipe numpy scipy
# (scipy optional — falls back to numpy-based Catmull-Rom if unavailable)
# Run: python HandGesture.py
# Controls: 's' to save final line as PNG, 'q' to quit

import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─── MediaPipe setup ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── Constants ────────────────────────────────────────────────────────────────
TIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

TRAIL_MAX_LEN = 12
SMOOTH_WINDOW = 5
MERGE_DIST = 70
SEPARATE_DIST = 110
DEBOUNCE_TIME = 0.3
ONE_HAND_WARN_TIME = 1.0

HAND_CONNECTION_COLOR = (0, 180, 0)
LANDMARK_COLOR = (0, 255, 255)
FINAL_LINE_COLOR = (255, 255, 0)
FINAL_LINE_GLOW_COLOR = (180, 180, 0)

GESTURE_COLORS = {
    "peace": (255, 105, 180),
    "fist": (0, 165, 255),
    "pointing": (0, 255, 255),
    "open_palm": (50, 255, 50),
}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def landmark_to_px(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))


def finger_angle(hand_landmarks, tip_id, mcp_id, w, h):
    """Angle in radians between (tip - mcp) and (wrist - mcp)."""
    lm = hand_landmarks.landmark
    tip = np.array([lm[tip_id].x * w, lm[tip_id].y * h])
    mcp = np.array([lm[mcp_id].x * w, lm[mcp_id].y * h])
    wrist = np.array([lm[0].x * w, lm[0].y * h])

    v1 = tip - mcp
    v2 = wrist - mcp
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_a, -1, 1))


def detect_gesture(hand_landmarks, w, h):
    angles = {}
    for i, name in enumerate(FINGER_NAMES):
        angles[name] = finger_angle(hand_landmarks, TIP_IDS[i], MCP_IDS[i], w, h)

    lm = hand_landmarks.landmark
    fingers_below = all(
        lm[TIP_IDS[i]].y > lm[MCP_IDS[i]].y for i in range(1, 5)
    )
    if lm[0].x < 0.5:
        thumb_folded = lm[TIP_IDS[0]].x > lm[MCP_IDS[0]].x
    else:
        thumb_folded = lm[TIP_IDS[0]].x < lm[MCP_IDS[0]].x
    if fingers_below and thumb_folded:
        return "fist"

    idx_ext = angles["index"] > 0.5
    mid_ext = angles["middle"] > 0.5
    ring_fold = angles["ring"] < 0.3
    pinky_fold = angles["pinky"] < 0.3

    if idx_ext and mid_ext and ring_fold and pinky_fold:
        return "peace"

    idx_ext_pt = angles["index"] > 0.5
    mid_fold = angles["middle"] < 0.3
    if idx_ext_pt and mid_fold and ring_fold and pinky_fold:
        return "pointing"

    if all(angles[f] > 0.7 for f in FINGER_NAMES):
        return "open_palm"

    return "open_palm"


# ─── Smoothing ────────────────────────────────────────────────────────────────

def smooth_deque(trail):
    if len(trail) < SMOOTH_WINDOW:
        return list(trail)
    arr = np.array(trail, dtype=np.float64)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    sx = np.convolve(arr[:, 0], kernel, mode='valid')
    sy = np.convolve(arr[:, 1], kernel, mode='valid')
    return [(int(x), int(y)) for x, y in zip(sx, sy)]


# ─── Catmull-Rom spline ──────────────────────────────────────────────────────

def catmull_rom_spline(points, num_interp=200):
    """Centripetal Catmull-Rom spline through a list of (x, y) points."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return points
    if n == 2:
        return [tuple(p) for p in np.linspace(pts[0], pts[1], num_interp).astype(int)]

    if HAS_SCIPY:
        t_knots = np.zeros(n)
        for i in range(1, n):
            t_knots[i] = t_knots[i - 1] + np.sqrt(np.linalg.norm(pts[i] - pts[i - 1]))
        if t_knots[-1] < 1e-6:
            return [tuple(p.astype(int)) for p in pts]
        cs_x = CubicSpline(t_knots, pts[:, 0], bc_type='natural')
        cs_y = CubicSpline(t_knots, pts[:, 1], bc_type='natural')
        t_fine = np.linspace(t_knots[0], t_knots[-1], num_interp)
        xs = cs_x(t_fine)
        ys = cs_y(t_fine)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    # Numpy fallback: centripetal Catmull-Rom
    extended = np.vstack([
        2 * pts[0] - pts[1],
        pts,
        2 * pts[-1] - pts[-2],
    ])
    result = []
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]

        d1 = max(np.linalg.norm(p1 - p0) ** 0.5, 1e-6)
        d2 = max(np.linalg.norm(p2 - p1) ** 0.5, 1e-6)
        d3 = max(np.linalg.norm(p3 - p2) ** 0.5, 1e-6)

        t0 = 0.0
        t1 = t0 + d1
        t2 = t1 + d2
        t3 = t2 + d3

        seg_pts = max(num_interp // (n - 1), 4)
        for j in range(seg_pts):
            t = t1 + (t2 - t1) * j / seg_pts

            a1 = (t1 - t) / (t1 - t0 + 1e-8) * p0 + (t - t0) / (t1 - t0 + 1e-8) * p1
            a2 = (t2 - t) / (t2 - t1 + 1e-8) * p1 + (t - t1) / (t2 - t1 + 1e-8) * p2
            a3 = (t3 - t) / (t3 - t2 + 1e-8) * p2 + (t - t2) / (t3 - t2 + 1e-8) * p3

            b1 = (t2 - t) / (t2 - t0 + 1e-8) * a1 + (t - t0) / (t2 - t0 + 1e-8) * a2
            b2 = (t3 - t) / (t3 - t1 + 1e-8) * a2 + (t - t1) / (t3 - t1 + 1e-8) * a3

            c = (t2 - t) / (t2 - t1 + 1e-8) * b1 + (t - t1) / (t2 - t1 + 1e-8) * b2
            result.append((int(c[0]), int(c[1])))

    result.append(tuple(pts[-1].astype(int)))
    return result


# ─── Styled trail drawing ────────────────────────────────────────────────────

def draw_dashed_line(img, p1, p2, color, thickness, dash_len=8, gap_len=6):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    drawn = 0.0
    while drawn < dist:
        s = (int(p1[0] + ux * drawn), int(p1[1] + uy * drawn))
        e_d = min(drawn + dash_len, dist)
        e = (int(p1[0] + ux * e_d), int(p1[1] + uy * e_d))
        cv2.line(img, s, e, color, thickness)
        drawn += dash_len + gap_len


def draw_wavy_line(img, p1, p2, color, thickness, amplitude=6, freq=0.15):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    steps = max(int(dist), 2)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        bx = p1[0] + dx * t
        by = p1[1] + dy * t
        offset = amplitude * math.sin(freq * i * 10)
        pts.append((int(bx + nx * offset), int(by + ny * offset)))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness)


def draw_zigzag_line(img, p1, p2, color, thickness, zz_size=7):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    num_segs = max(int(dist / (zz_size * 2)), 1)
    pts = [p1]
    for i in range(1, num_segs * 2 + 1):
        t = i / (num_segs * 2)
        bx = p1[0] + dx * t
        by = p1[1] + dy * t
        d = 1 if i % 2 == 1 else -1
        pts.append((int(bx + nx * zz_size * d), int(by + ny * zz_size * d)))
    pts.append(p2)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness)


def draw_styled_segment(img, p1, p2, gesture, color, thickness=2):
    if gesture == "peace":
        draw_dashed_line(img, p1, p2, color, thickness)
    elif gesture == "fist":
        draw_wavy_line(img, p1, p2, color, thickness)
    elif gesture == "pointing":
        draw_zigzag_line(img, p1, p2, color, thickness)
    else:
        cv2.line(img, p1, p2, color, thickness)


# ─── Hand skeleton drawing ───────────────────────────────────────────────────

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand_skeleton(frame, hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    for a, b in HAND_BONES:
        pa = landmark_to_px(lm[a], w, h)
        pb = landmark_to_px(lm[b], w, h)
        cv2.line(frame, pa, pb, HAND_CONNECTION_COLOR, 1, cv2.LINE_AA)
    for i in range(21):
        pt = landmark_to_px(lm[i], w, h)
        cv2.circle(frame, pt, 3, LANDMARK_COLOR, -1, cv2.LINE_AA)


# ─── Final line drawing with glow ────────────────────────────────────────────

def draw_final_spline(frame, spline_pts):
    if len(spline_pts) < 2:
        return
    pts_arr = np.array(spline_pts, dtype=np.int32).reshape((-1, 1, 2))

    overlay = frame.copy()
    cv2.polylines(overlay, [pts_arr], False, FINAL_LINE_GLOW_COLOR, 8, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cv2.polylines(frame, [pts_arr], False, FINAL_LINE_COLOR, 3, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    window_name = "Gesture Drawing – Clean Version"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Per-fingertip trail: key = (hand_label, finger_index)
    trails = {}
    # Per-fingertip gesture: same key
    trail_gestures = {}
    # Last known fingertip positions for occlusion fallback
    last_known_tips = {}

    # Merge-separate state machine
    merge_state = "apart"
    merge_start_time = 0.0
    separate_start_time = 0.0
    proximity_entered = False
    separation_entered = False

    # Single-hand warning timer
    one_hand_since = 0.0
    one_hand_active = False

    # Final spline
    final_spline = None
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
        detected = {}  # label -> (landmarks, gesture, tips_dict)
        active_keys = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = hand_info.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"

                draw_hand_skeleton(frame, hand_lm, w, h)

                gesture = detect_gesture(hand_lm, w, h)
                tips = {}
                for fi, tid in enumerate(TIP_IDS):
                    pos = landmark_to_px(hand_lm.landmark[tid], w, h)
                    tips[fi] = pos
                    key = (label, fi)
                    active_keys.add(key)
                    last_known_tips[key] = pos

                    if key not in trails:
                        trails[key] = deque(maxlen=TRAIL_MAX_LEN)
                    trails[key].append(pos)
                    trail_gestures[key] = gesture

                detected[label] = (hand_lm, gesture, tips)

                palm = landmark_to_px(hand_lm.landmark[0], w, h)
                cv2.putText(
                    frame, gesture,
                    (palm[0] - 30, palm[1] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
                )

        # Use last known positions for briefly occluded tips
        for key in list(trails.keys()):
            if key not in active_keys and key in last_known_tips:
                trails[key].append(last_known_tips[key])

        # Draw trails with per-finger gesture style and fade
        for key, trail in trails.items():
            smoothed = smooth_deque(trail)
            if len(smoothed) < 2:
                continue
            gesture = trail_gestures.get(key, "open_palm")
            base_color = GESTURE_COLORS.get(gesture, (255, 255, 255))

            for i in range(1, len(smoothed)):
                alpha = i / len(smoothed)
                c = tuple(int(ch * alpha) for ch in base_color)
                thick = max(1, int(2 * alpha))
                draw_styled_segment(frame, smoothed[i - 1], smoothed[i], gesture, c, thick)

        # Clean up trails for keys that have been inactive too long
        stale = [k for k in trails if k not in active_keys and len(trails[k]) == 0]
        for k in stale:
            del trails[k]
            trail_gestures.pop(k, None)

        # ── Single-hand warning ───────────────────────────────────────────
        num_hands = len(detected)
        if num_hands == 1:
            if not one_hand_active:
                one_hand_since = now
                one_hand_active = True
            elif now - one_hand_since > ONE_HAND_WARN_TIME:
                cv2.putText(
                    frame, "Need two hands for final line",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA,
                )
        else:
            one_hand_active = False

        # ── Merge-separate state machine using palm centers (landmark 0) ──
        if "Left" in detected and "Right" in detected:
            lp = landmark_to_px(detected["Left"][0].landmark[0], w, h)
            rp = landmark_to_px(detected["Right"][0].landmark[0], w, h)
            dist = math.hypot(lp[0] - rp[0], lp[1] - rp[1])

            if merge_state == "apart":
                if dist < MERGE_DIST:
                    if not proximity_entered:
                        merge_start_time = now
                        proximity_entered = True
                    elif now - merge_start_time >= DEBOUNCE_TIME:
                        merge_state = "merged"
                        proximity_entered = False
                        separation_entered = False
                else:
                    proximity_entered = False

            elif merge_state == "merged":
                if dist > SEPARATE_DIST:
                    if not separation_entered:
                        separate_start_time = now
                        separation_entered = True
                    elif now - separate_start_time >= DEBOUNCE_TIME:
                        merge_state = "apart"
                        separation_entered = False
                        proximity_entered = False

                        trails.clear()
                        trail_gestures.clear()

                        left_tips = detected["Left"][2]
                        right_tips = detected["Right"][2]

                        order_left = [0, 1, 2, 3, 4]      # thumb, index, middle, ring, pinky
                        order_right = [4, 3, 2, 1, 0]     # pinky, ring, middle, index, thumb

                        pts = []
                        missing = False
                        for fi in order_left:
                            key = ("Left", fi)
                            if fi in left_tips:
                                pts.append(left_tips[fi])
                            elif key in last_known_tips:
                                pts.append(last_known_tips[key])
                            else:
                                missing = True
                                break
                        if not missing:
                            for fi in order_right:
                                key = ("Right", fi)
                                if fi in right_tips:
                                    pts.append(right_tips[fi])
                                elif key in last_known_tips:
                                    pts.append(last_known_tips[key])
                                else:
                                    missing = True
                                    break

                        if missing:
                            cv2.putText(
                                frame, "Missing some fingers - cannot draw line",
                                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA,
                            )
                            final_spline = None
                        else:
                            final_spline = catmull_rom_spline(pts, num_interp=300)
                else:
                    separation_entered = False
        else:
            proximity_entered = False
            separation_entered = False
            if merge_state == "merged":
                merge_state = "apart"

        # State display
        cv2.putText(
            frame, f"State: {merge_state}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0) if merge_state == "merged" else (200, 200, 200),
            2, cv2.LINE_AA,
        )

        # Draw final spline
        if final_spline is not None:
            draw_final_spline(frame, final_spline)

        # Save message overlay
        if save_msg and now - save_msg_time < 2.0:
            cv2.putText(
                frame, save_msg,
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if final_spline is not None and len(final_spline) >= 2:
                save_img = np.zeros((h, w, 3), dtype=np.uint8)
                pts_arr = np.array(final_spline, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(save_img, [pts_arr], False, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imwrite("final_gesture_line.png", save_img)
                save_msg = "Saved final_gesture_line.png"
                save_msg_time = now
                print(save_msg)
            else:
                save_msg = "No line to save"
                save_msg_time = now
                print(save_msg)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
