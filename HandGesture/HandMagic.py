# Fire Magic – Fixed Graphics
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
SHAPE_LIFETIME = 1.2
SHAPE_FADE_TIME = 0.4
MAX_SHAPES = 50

MAX_PARTICLES = 300
PARTICLE_LIFETIME = 0.4

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

# Flame colors: ONLY red, orange, yellow – NO GREEN
COLOR_BRIGHT_YELLOW = (0, 255, 255)
COLOR_YELLOW = (0, 230, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_DARK_RED = (0, 0, 150)
COLOR_WHITE_HOT = (200, 240, 255)

FLAME_BIRTH_COLORS = [COLOR_BRIGHT_YELLOW, COLOR_WHITE_HOT, COLOR_YELLOW]
FLAME_MID_COLORS = [COLOR_ORANGE, (0, 120, 255), (0, 200, 255)]
FLAME_END_COLORS = [COLOR_RED, COLOR_DARK_RED, (0, 50, 200)]

SHADOW_OFFSET = 3
SHADOW_ALPHA = 0.5


# ─── Particle ────────────────────────────────────────────────────────────────

class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'birth', 'lifetime', 'max_size', 'layer']

    def __init__(self, x, y, vx, vy, lifetime, max_size, layer=1):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.birth = time.time()
        self.lifetime = lifetime
        self.max_size = max_size
        self.layer = layer  # 0=back(red), 1=mid(orange), 2=front(yellow)


def particle_color_and_size(p, now):
    age = now - p.birth
    t = age / p.lifetime
    if t >= 1.0:
        return None, 0

    size = max(1, int(p.max_size * (1.0 - t)))

    if t < 0.25:
        color = random.choice(FLAME_BIRTH_COLORS)
    elif t < 0.6:
        color = random.choice(FLAME_MID_COLORS)
    else:
        color = random.choice(FLAME_END_COLORS)

    brightness = max(0.3, 1.0 - t * 0.7)
    color = tuple(max(0, min(255, int(c * brightness))) for c in color)
    return color, size


# ─── Shape ───────────────────────────────────────────────────────────────────

class Shape:
    __slots__ = ['x', 'y', 'vx', 'vy', 'kind', 'birth', 'angle', 'size']

    def __init__(self, x, y, vx, vy, kind, size):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.kind = kind
        self.birth = time.time()
        self.angle = math.atan2(vy, vx)
        self.size = size


# ─── Styled shape drawing ────────────────────────────────────────────────────

def draw_drop_shadow(frame, draw_fn, offset=SHADOW_OFFSET):
    overlay = frame.copy()
    draw_fn(overlay, offset_x=offset, offset_y=offset, shadow=True)
    cv2.addWeighted(overlay, SHADOW_ALPHA, frame, 1.0 - SHADOW_ALPHA, 0, frame)


def draw_circle_styled(frame, cx, cy, radius, alpha):
    # Drop shadow
    shadow_overlay = frame.copy()
    cv2.circle(shadow_overlay, (cx + SHADOW_OFFSET, cy + SHADOW_OFFSET),
               radius, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, SHADOW_ALPHA * alpha, frame, 1.0 - SHADOW_ALPHA * alpha, 0, frame)

    # Semi-transparent fill: deep red-orange
    fill_overlay = frame.copy()
    fill_color = (0, 40, 200)
    cv2.circle(fill_overlay, (cx, cy), radius, fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(fill_overlay, 0.6 * alpha, frame, 1.0 - 0.6 * alpha, 0, frame)

    # White outline
    outline_color = tuple(int(255 * alpha) for _ in range(3))
    cv2.circle(frame, (cx, cy), radius, outline_color, 2, cv2.LINE_AA)

    # Highlight dot (reflection)
    hx = cx - radius // 3
    hy = cy - radius // 3
    highlight_r = max(1, radius // 4)
    highlight_color = tuple(int(220 * alpha) for _ in range(3))
    cv2.circle(frame, (hx, hy), highlight_r, highlight_color, -1, cv2.LINE_AA)


def draw_triangle_styled(frame, cx, cy, size, alpha, angle):
    # Isosceles triangle pointing in direction of motion
    pts = []
    for i in range(3):
        a = angle + i * (2 * math.pi / 3)
        if i == 0:
            r = size * 0.8  # tip is longer
        else:
            r = size * 0.55
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        pts.append([px, py])
    pts_arr = np.array(pts, dtype=np.int32)

    # Shadow triangle offset
    shadow_pts = np.array([[p[0] + SHADOW_OFFSET, p[1] + SHADOW_OFFSET] for p in pts], dtype=np.int32)
    shadow_overlay = frame.copy()
    cv2.fillPoly(shadow_overlay, [shadow_pts], (0, 0, 0), cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, SHADOW_ALPHA * alpha, frame, 1.0 - SHADOW_ALPHA * alpha, 0, frame)

    # Orange fill
    fill_color = tuple(int(c * alpha) for c in (0, 130, 255))
    cv2.fillPoly(frame, [pts_arr], fill_color, cv2.LINE_AA)

    # Gold outline
    outline_color = tuple(int(c * alpha) for c in (0, 215, 255))
    cv2.polylines(frame, [pts_arr], True, outline_color, 2, cv2.LINE_AA)


def draw_square_styled(frame, cx, cy, size, alpha, angle):
    # Rotated square (diamond shape) – always 45 degrees rotated
    rot_angle = angle + math.pi / 4
    half = size // 2
    corners = []
    for dx, dy in [(-half, 0), (0, -half), (half, 0), (0, half)]:
        rx = dx * math.cos(rot_angle) - dy * math.sin(rot_angle)
        ry = dx * math.sin(rot_angle) + dy * math.cos(rot_angle)
        corners.append([int(cx + rx), int(cy + ry)])
    pts_arr = np.array(corners, dtype=np.int32)

    # Shadow
    shadow_pts = np.array([[p[0] + SHADOW_OFFSET, p[1] + SHADOW_OFFSET] for p in corners], dtype=np.int32)
    shadow_overlay = frame.copy()
    cv2.fillPoly(shadow_overlay, [shadow_pts], (0, 0, 0), cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, SHADOW_ALPHA * alpha, frame, 1.0 - SHADOW_ALPHA * alpha, 0, frame)

    # Dark blue fill
    fill_color = tuple(int(c * alpha) for c in (150, 50, 10))
    cv2.fillPoly(frame, [pts_arr], fill_color, cv2.LINE_AA)

    # Cyan glow outline (drawn twice for glow effect)
    glow_color = tuple(int(c * alpha * 0.4) for c in (255, 255, 0))
    cv2.polylines(frame, [pts_arr], True, glow_color, 4, cv2.LINE_AA)
    outline_color = tuple(int(c * alpha) for c in (255, 255, 0))
    cv2.polylines(frame, [pts_arr], True, outline_color, 2, cv2.LINE_AA)

    # Corner highlights
    for corner in corners:
        h_color = tuple(int(c * alpha) for c in (255, 255, 200))
        cv2.circle(frame, (corner[0], corner[1]), 2, h_color, -1, cv2.LINE_AA)


def draw_star_styled(frame, cx, cy, outer_r, alpha, angle):
    inner_r = outer_r * 0.4
    pts = []
    for i in range(10):
        a = angle + i * (math.pi / 5) - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
    pts_arr = np.array(pts, dtype=np.int32)

    # Shadow
    shadow_pts = np.array([[p[0] + SHADOW_OFFSET, p[1] + SHADOW_OFFSET] for p in pts], dtype=np.int32)
    shadow_overlay = frame.copy()
    cv2.fillPoly(shadow_overlay, [shadow_pts], (0, 0, 0), cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, SHADOW_ALPHA * alpha, frame, 1.0 - SHADOW_ALPHA * alpha, 0, frame)

    # Deep orange-red fill
    fill_color = tuple(int(c * alpha) for c in (0, 80, 220))
    cv2.fillPoly(frame, [pts_arr], fill_color, cv2.LINE_AA)

    # Bright orange outline
    outline_color = tuple(int(c * alpha) for c in (0, 165, 255))
    cv2.polylines(frame, [pts_arr], True, outline_color, 2, cv2.LINE_AA)


def draw_fireball_styled(frame, cx, cy, radius, alpha):
    # Shadow
    shadow_overlay = frame.copy()
    cv2.circle(shadow_overlay, (cx + SHADOW_OFFSET, cy + SHADOW_OFFSET),
               radius, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(shadow_overlay, SHADOW_ALPHA * alpha, frame, 1.0 - SHADOW_ALPHA * alpha, 0, frame)

    # Outer red glow
    outer_color = tuple(int(c * alpha * 0.5) for c in COLOR_RED)
    cv2.circle(frame, (cx, cy), int(radius * 1.4), outer_color, -1, cv2.LINE_AA)

    # Main body: dark red
    body_color = tuple(int(c * alpha) for c in (0, 30, 200))
    cv2.circle(frame, (cx, cy), radius, body_color, -1, cv2.LINE_AA)

    # Inner orange core
    inner_color = tuple(int(c * alpha) for c in COLOR_ORANGE)
    cv2.circle(frame, (cx, cy), int(radius * 0.6), inner_color, -1, cv2.LINE_AA)

    # Bright yellow center
    center_color = tuple(int(c * alpha) for c in COLOR_BRIGHT_YELLOW)
    cv2.circle(frame, (cx, cy), int(radius * 0.3), center_color, -1, cv2.LINE_AA)

    # White-hot highlight
    h_color = tuple(int(200 * alpha) for _ in range(3))
    cv2.circle(frame, (cx - radius // 4, cy - radius // 4),
               max(1, radius // 5), h_color, -1, cv2.LINE_AA)


def draw_shape(frame, shape, alpha):
    cx, cy = int(shape.x), int(shape.y)
    kind = shape.kind
    s = shape.size

    if kind == "circle":
        draw_circle_styled(frame, cx, cy, s, alpha)
    elif kind == "triangle":
        draw_triangle_styled(frame, cx, cy, s, alpha, shape.angle)
    elif kind == "square":
        draw_square_styled(frame, cx, cy, s, alpha, shape.angle)
    elif kind == "star":
        draw_star_styled(frame, cx, cy, s, alpha, shape.angle)
    elif kind == "fireball":
        draw_fireball_styled(frame, cx, cy, s, alpha)


# ─── Particle spawning ──────────────────────────────────────────────────────

def spawn_flame_particles(shape, particles, count):
    for _ in range(count):
        spread = 10 if shape.kind == "fireball" else 6
        ox = random.uniform(-spread, spread)
        oy = random.uniform(-spread, spread)

        # Particles move opposite to shape direction with random drift
        pvx = -shape.vx * random.uniform(0.15, 0.5) + random.uniform(-2.0, 2.0)
        pvy = -shape.vy * random.uniform(0.15, 0.5) + random.uniform(-2.0, 2.0)

        # Layered sizes: back=large red, mid=medium orange, front=small yellow
        layer = random.choices([0, 1, 2], weights=[3, 4, 3])[0]
        if layer == 0:
            max_size = random.uniform(4, 6)
        elif layer == 1:
            max_size = random.uniform(2.5, 4)
        else:
            max_size = random.uniform(1, 2.5)

        lifetime = PARTICLE_LIFETIME * random.uniform(0.7, 1.3)
        particles.append(Particle(shape.x + ox, shape.y + oy, pvx, pvy,
                                  lifetime, max_size, layer))


# ─── Particle drawing ────────────────────────────────────────────────────────

def draw_particles(frame, particles, now):
    for p in particles:
        age = now - p.birth
        t = age / p.lifetime
        if t >= 1.0:
            continue

        size = max(1, int(p.max_size * (1.0 - t)))

        # Color based on layer and age – strictly red/orange/yellow spectrum
        if p.layer == 2:  # front: yellow -> orange
            if t < 0.4:
                color = COLOR_BRIGHT_YELLOW
            else:
                color = COLOR_ORANGE
        elif p.layer == 1:  # middle: orange -> red
            if t < 0.3:
                color = COLOR_ORANGE
            elif t < 0.7:
                color = (0, 100, 255)  # red-orange
            else:
                color = COLOR_RED
        else:  # back: red -> dark red
            if t < 0.3:
                color = COLOR_RED
            else:
                color = COLOR_DARK_RED

        brightness = max(0.25, 1.0 - t * 0.75)
        color = tuple(max(0, min(255, int(c * brightness))) for c in color)

        cv2.circle(frame, (int(p.x), int(p.y)), size, color, -1, cv2.LINE_AA)


# ─── Hand skeleton ────────────────────────────────────────────────────────────

def draw_hand(frame, landmarks, w, h):
    lm = landmarks.landmark
    for a, b in HAND_BONE_CONNECTIONS:
        pa = (int(lm[a].x * w), int(lm[a].y * h))
        pb = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(frame, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
    for tid in FINGERTIP_IDS:
        pt = (int(lm[tid].x * w), int(lm[tid].y * h))
        cv2.circle(frame, pt, 5, (180, 105, 255), -1, cv2.LINE_AA)  # bright pink


def draw_cooldown(frame, x, y, progress):
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

    window_name = "Fire Magic - Fixed Graphics"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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

        # Darken camera background so shapes/flames pop
        cv2.addWeighted(frame, 0.6, black_overlay, 0.4, 0, frame)

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

                    vels = []
                    count = min(VELOCITY_SMOOTH, len(hist) - 1)
                    for vi in range(1, count + 1):
                        dx = hist[-vi][0] - hist[-(vi + 1)][0]
                        dy = hist[-vi][1] - hist[-(vi + 1)][1]
                        vels.append((dx, dy))

                    avg_vx = sum(v[0] for v in vels) / len(vels)
                    avg_vy = sum(v[1] for v in vels) / len(vels)
                    speed = math.hypot(avg_vx, avg_vy)

                    last_t = last_fire_time.get(finger_key, 0)
                    elapsed = now - last_t
                    if elapsed < FLICK_DEBOUNCE:
                        progress = elapsed / FLICK_DEBOUNCE
                        draw_cooldown(frame, px, py, progress)

                    if speed > FLICK_THRESHOLD:
                        if elapsed >= FLICK_DEBOUNCE:
                            last_fire_time[finger_key] = now

                            norm = speed + 1e-6
                            vx = (avg_vx / norm) * SHAPE_BASE_SPEED
                            vy = (avg_vy / norm) * SHAPE_BASE_SPEED

                            fname = FINGER_NAMES[fi]
                            kind = FINGER_SHAPE_MAP[fname]

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

                            new_shape = Shape(px, py, vx, vy, kind, size)
                            shapes.append(new_shape)

                            # Initial burst: 10-15 particles at spawn
                            spawn_flame_particles(new_shape, particles,
                                                  random.randint(10, 15))

        # ── Update shapes ────────────────────────────────────────────────
        surviving_shapes = deque(maxlen=MAX_SHAPES)
        for s in shapes:
            age = now - s.birth
            if age > SHAPE_LIFETIME:
                continue

            s.x += s.vx
            s.y += s.vy
            s.angle += 0.04

            # Continuous flame trail: 5 new particles per shape per frame
            spawn_flame_particles(s, particles, 5)

            surviving_shapes.append(s)
        shapes = surviving_shapes

        # ── Update particles ─────────────────────────────────────────────
        surviving_particles = deque(maxlen=MAX_PARTICLES)
        for p in particles:
            age = now - p.birth
            if age >= p.lifetime:
                continue
            p.x += p.vx
            p.y += p.vy
            p.vx *= 0.94
            p.vy *= 0.94
            surviving_particles.append(p)
        particles = surviving_particles

        # ── Draw particles (behind shapes) ───────────────────────────────
        # Sort: back layer first, front layer last (painters algorithm)
        sorted_particles = sorted(particles, key=lambda p: p.layer)
        draw_particles(frame, sorted_particles, now)

        # ── Draw shapes ──────────────────────────────────────────────────
        for s in shapes:
            age = now - s.birth
            remaining = SHAPE_LIFETIME - age
            alpha = 1.0 if remaining > SHAPE_FADE_TIME else max(0, remaining / SHAPE_FADE_TIME)
            draw_shape(frame, s, alpha)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
