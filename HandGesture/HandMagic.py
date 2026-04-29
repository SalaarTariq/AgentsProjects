# pip install opencv-python mediapipe numpy
# Run: python HandMagic.py
# Controls: 'q' to quit

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
from collections import deque

# ─── MediaPipe setup ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── Constants ────────────────────────────────────────────────────────────────
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

VELOCITY_HISTORY = 5
VELOCITY_SMOOTH = 3
FLICK_THRESHOLD = 35
FLICK_DEBOUNCE = 0.3

SHAPE_BASE_SPEED = 12
SHAPE_LIFETIME = 1.5
SHAPE_FADE_TIME = 0.5
MAX_SHAPES = 50

PARTICLES_PER_SHAPE = 6
MAX_PARTICLES = 200
PARTICLE_LIFETIME_FRAMES = 18  # ~0.3s at 60fps

HAND_BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_SHAPE_MAP = {
    "thumb": "fireball",
    "index": "triangle",
    "middle": "circle",
    "ring": "square",
    "pinky": "star",
}

SHAPE_COLORS = {
    "circle": (0, 0, 220),
    "triangle": (0, 180, 0),
    "square": (0, 230, 255),
    "star": (255, 100, 255),
    "fireball": (0, 100, 255),
}

FLAME_COLORS = [(0, 0, 255), (0, 80, 255), (0, 165, 255), (0, 200, 255), (0, 220, 255)]


# ─── Data classes ─────────────────────────────────────────────────────────────

class Shape:
    __slots__ = ['x', 'y', 'vx', 'vy', 'kind', 'color', 'birth', 'angle', 'size']

    def __init__(self, x, y, vx, vy, kind, color, size):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.kind = kind
        self.color = color
        self.birth = time.time()
        self.angle = random.uniform(0, 2 * math.pi)
        self.size = size


class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color']

    def __init__(self, x, y, vx, vy, life, size):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.life = life
        self.max_life = life
        self.size = size
        self.color = random.choice(FLAME_COLORS)


# ─── Drawing functions ────────────────────────────────────────────────────────

def draw_circle_shape(overlay, cx, cy, radius, color, alpha):
    c = tuple(int(ch * alpha) for ch in color)
    cv2.circle(overlay, (cx, cy), radius, c, -1, cv2.LINE_AA)


def draw_triangle_shape(overlay, cx, cy, side, color, alpha, angle):
    c = tuple(int(ch * alpha) for ch in color)
    pts = []
    for i in range(3):
        a = angle + i * (2 * math.pi / 3)
        px = int(cx + side * 0.6 * math.cos(a))
        py = int(cy + side * 0.6 * math.sin(a))
        pts.append([px, py])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], c, cv2.LINE_AA)


def draw_square_shape(overlay, cx, cy, size, color, alpha, angle):
    c = tuple(int(ch * alpha) for ch in color)
    half = size // 2
    corners = []
    for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        rx = dx * math.cos(angle) - dy * math.sin(angle)
        ry = dx * math.sin(angle) + dy * math.cos(angle)
        corners.append([int(cx + rx), int(cy + ry)])
    pts_arr = np.array(corners, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], c, cv2.LINE_AA)


def draw_star_shape(overlay, cx, cy, outer_r, color, alpha, angle):
    c = tuple(int(ch * alpha) for ch in color)
    inner_r = outer_r * 0.4
    pts = []
    for i in range(10):
        a = angle + i * (math.pi / 5)
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], c, cv2.LINE_AA)


def draw_shape(overlay, shape, alpha):
    cx, cy = int(shape.x), int(shape.y)
    kind = shape.kind
    color = shape.color
    s = shape.size

    if kind == "circle":
        draw_circle_shape(overlay, cx, cy, s, color, alpha)
    elif kind == "triangle":
        draw_triangle_shape(overlay, cx, cy, s, color, alpha, shape.angle)
    elif kind == "square":
        draw_square_shape(overlay, cx, cy, s, color, alpha, shape.angle)
    elif kind == "star":
        draw_star_shape(overlay, cx, cy, s, color, alpha, shape.angle)
    elif kind == "fireball":
        draw_circle_shape(overlay, cx, cy, s, color, alpha)
        draw_circle_shape(overlay, cx, cy, int(s * 0.6), (0, 200, 255), alpha * 0.8)


def draw_glow(overlay, shape, alpha):
    cx, cy = int(shape.x), int(shape.y)
    glow_size = int(shape.size * 2.2)
    glow_alpha = alpha * 0.25
    c = tuple(int(ch * glow_alpha) for ch in shape.color)
    cv2.circle(overlay, (cx, cy), glow_size, c, -1, cv2.LINE_AA)


def draw_particle(overlay, p):
    alpha = max(0.0, p.life / p.max_life)
    r = max(1, int(p.size * alpha))
    c = tuple(int(ch * alpha) for ch in p.color)
    cv2.circle(overlay, (int(p.x), int(p.y)), r, c, -1, cv2.LINE_AA)


# ─── Hand skeleton ────────────────────────────────────────────────────────────

def draw_hand(frame, landmarks, w, h):
    lm = landmarks.landmark
    for a, b in HAND_BONE_CONNECTIONS:
        pa = (int(lm[a].x * w), int(lm[a].y * h))
        pb = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(frame, pa, pb, (200, 200, 200), 1, cv2.LINE_AA)
    for tid in FINGERTIP_IDS:
        pt = (int(lm[tid].x * w), int(lm[tid].y * h))
        cv2.circle(frame, pt, 5, (255, 255, 0), -1, cv2.LINE_AA)


# ─── Cooldown indicator ──────────────────────────────────────────────────────

def draw_cooldown(frame, x, y, progress):
    """Draw a small arc showing cooldown progress (0 to 1)."""
    if progress >= 1.0:
        return
    radius = 12
    angle = int(360 * progress)
    cv2.ellipse(frame, (x, y + 20), (radius, radius), -90, 0, angle,
                (0, 140, 255), 2, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    window_name = "Hand Magic - Fire Shapes"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Position history per fingertip: key = (hand_index, finger_index)
    pos_history = {}
    last_fire_time = {}

    shapes = deque(maxlen=MAX_SHAPES)
    particles = deque(maxlen=MAX_PARTICLES)

    black_overlay = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if black_overlay is None:
            black_overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark background blend to make effects pop
        cv2.addWeighted(frame, 0.7, black_overlay, 0.3, 0, frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        now = time.time()

        if results.multi_hand_landmarks:
            for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                draw_hand(frame, hand_lm, w, h)
                lm = hand_lm.landmark

                for fi, tid in enumerate(FINGERTIP_IDS):
                    px = int(lm[tid].x * w)
                    py = int(lm[tid].y * h)
                    finger_key = (hand_idx, fi)

                    if finger_key not in pos_history:
                        pos_history[finger_key] = deque(maxlen=VELOCITY_HISTORY)
                    pos_history[finger_key].append((px, py))

                    hist = pos_history[finger_key]
                    if len(hist) < 2:
                        continue

                    # Compute smoothed velocity (average last VELOCITY_SMOOTH frames)
                    vels = []
                    count = min(VELOCITY_SMOOTH, len(hist) - 1)
                    for vi in range(1, count + 1):
                        dx = hist[-vi][0] - hist[-(vi + 1)][0]
                        dy = hist[-vi][1] - hist[-(vi + 1)][1]
                        vels.append((dx, dy))

                    avg_vx = sum(v[0] for v in vels) / len(vels)
                    avg_vy = sum(v[1] for v in vels) / len(vels)
                    speed = math.hypot(avg_vx, avg_vy)

                    # Cooldown indicator
                    last_t = last_fire_time.get(finger_key, 0)
                    elapsed = now - last_t
                    if elapsed < FLICK_DEBOUNCE:
                        progress = elapsed / FLICK_DEBOUNCE
                        draw_cooldown(frame, px, py, progress)

                    # Flick detection
                    if speed > FLICK_THRESHOLD:
                        if elapsed >= FLICK_DEBOUNCE:
                            last_fire_time[finger_key] = now

                            norm = speed + 1e-6
                            vx = (avg_vx / norm) * SHAPE_BASE_SPEED
                            vy = (avg_vy / norm) * SHAPE_BASE_SPEED

                            fname = FINGER_NAMES[fi]
                            kind = FINGER_SHAPE_MAP[fname]
                            color = SHAPE_COLORS[kind]

                            if kind == "fireball":
                                size = random.randint(16, 22)
                            elif kind == "circle":
                                size = random.randint(10, 15)
                            elif kind == "triangle":
                                size = 20
                            elif kind == "square":
                                size = 18
                            else:
                                size = 12

                            shapes.append(Shape(px, py, vx, vy, kind, color, size))

        # ── Update shapes and generate particles ─────────────────────────
        surviving_shapes = deque(maxlen=MAX_SHAPES)
        for s in shapes:
            age = now - s.birth
            if age > SHAPE_LIFETIME:
                continue

            s.x += s.vx
            s.y += s.vy
            s.angle += 0.05

            # Spawn flame particles trailing behind
            num_p = PARTICLES_PER_SHAPE + (3 if s.kind == "fireball" else 0)
            for _ in range(num_p):
                if len(particles) >= MAX_PARTICLES:
                    particles.popleft()
                spread = 8 if s.kind == "fireball" else 5
                ox = random.uniform(-spread, spread)
                oy = random.uniform(-spread, spread)
                pvx = -s.vx * random.uniform(0.1, 0.4) + random.uniform(-1.5, 1.5)
                pvy = -s.vy * random.uniform(0.1, 0.4) + random.uniform(-1.5, 1.5)
                ps = random.uniform(2, 5) if s.kind == "fireball" else random.uniform(1.5, 3.5)
                particles.append(Particle(s.x + ox, s.y + oy, pvx, pvy,
                                          PARTICLE_LIFETIME_FRAMES, ps))

            surviving_shapes.append(s)
        shapes = surviving_shapes

        # ── Update particles ──────────────────────────────────────────────
        surviving_particles = deque(maxlen=MAX_PARTICLES)
        for p in particles:
            p.life -= 1
            if p.life <= 0:
                continue
            p.x += p.vx
            p.y += p.vy
            p.vx *= 0.95
            p.vy *= 0.95
            surviving_particles.append(p)
        particles = surviving_particles

        # ── Draw everything on overlay then blend ─────────────────────────
        effects = np.zeros_like(frame)

        # Draw particles first (behind shapes)
        for p in particles:
            draw_particle(effects, p)

        # Draw shape glows
        for s in shapes:
            age = now - s.birth
            remaining = SHAPE_LIFETIME - age
            alpha = 1.0 if remaining > SHAPE_FADE_TIME else max(0, remaining / SHAPE_FADE_TIME)
            draw_glow(effects, s, alpha)

        # Draw shapes
        for s in shapes:
            age = now - s.birth
            remaining = SHAPE_LIFETIME - age
            alpha = 1.0 if remaining > SHAPE_FADE_TIME else max(0, remaining / SHAPE_FADE_TIME)
            draw_shape(effects, s, alpha)

        # Blend effects onto frame (additive-style)
        mask = effects.astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        result = np.clip(frame_f + effects.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        frame = result

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
