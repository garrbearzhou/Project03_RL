"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
renderer.py - all Pygame drawing lives here (kept separate from the RL logic).

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

Nothing here changes how the game plays or how the agent learns. Fast-training mode (SPACE)
only draws every few loops so graphics barely slow training down.
"""
import bisect
import math
import random
import pygame

W, H = 800, 800
CAR_SCREEN_Y = int(H * 0.75)       # the car stays here; the world scrolls past it
_cam_x = 0.0                       # camera follows the car sideways (smoothly)

PAD_COLORS = {2: (255, 150, 0), 3: (0, 205, 255)}      # 2x = orange, 3x = cyan
CAR_SCALE = 1.25                  # the drawn car is a bit bigger than its (invisible) hitbox
PAD_USED = (105, 105, 110)
ROAD_COLORS = {None: (62, 62, 67), "ice": (158, 206, 232), "mud": (104, 74, 46), "mountain": (104, 98, 93)}
SPECKS = {None: ((46, 46, 51), (80, 80, 86)), "ice": ((225, 242, 252), (125, 178, 205)),
          "mud": ((76, 52, 32), (128, 94, 62)), "mountain": ((72, 68, 64), (132, 125, 118))}

_A = None          # assets (built on the first frame, once pygame is running)
_S = {"last_steps": 10 ** 9, "particles": [], "marks": [], "prev_heading": 0.0, "prev_speed": 0.0,
      "prev_rail": 0, "prev_air": 0, "wheels": None, "t": 0}


def _span(track, path):
    """Where to draw a road: the first road starts 200 px before the start line, the last runs 240 px past the finish."""
    y0 = path.y0 - 200 if path is track.paths[0] else path.y0
    y1 = path.y1 + 240 if path is track.paths[-1] else path.y1
    return y0, y1


def world_to_screen(env, wx, wy):
    """Camera follows the car: convert world coords to screen pixels."""
    return W // 2 + (wx - _cam_x), CAR_SCREEN_Y - (wy - env.y)


def _h(a, b=0, c=0):
    """Cheap deterministic 'random' number from coordinates (so decorations never flicker)."""
    return ((int(a) * 73856093) ^ (int(b) * 19349663) ^ (int(c) * 83492791)) & 0xFFFFFF


# =====================================================================================
#  Sprites (drawn once, then reused every frame)
# =====================================================================================
def _wrapped(surf, size, fn):
    """Call fn(dx, dy) at the 9 wrapped offsets so a tile has no visible seams."""
    for dx in (-size, 0, size):
        for dy in (-size, 0, size):
            fn(dx, dy)


def _make_grass():
    s = pygame.Surface((200, 200))
    s.fill((54, 124, 54))
    r = random.Random(3)
    for _ in range(30):                                                # soft darker / lighter patches
        x, y, w, h = r.randrange(200), r.randrange(200), r.randrange(30, 80), r.randrange(18, 46)
        col = (49, 114, 50) if r.random() < 0.5 else (61, 136, 59)
        _wrapped(s, 200, lambda dx, dy: pygame.draw.ellipse(s, col, (x + dx, y + dy, w, h)))
    for _ in range(700):                                               # grass blades
        x, y = r.randrange(200), r.randrange(200)
        col = (84, 156, 72) if r.random() < 0.5 else (40, 100, 42)
        pygame.draw.rect(s, col, (x, y, 1, 3))
    return s


def _make_tree(radius, seed):
    size = radius * 2 + 40
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    r = random.Random(seed)
    pygame.draw.ellipse(s, (0, 0, 0, 70), (c - radius + 9, c - radius + 13, radius * 2, radius * 2 - 4))   # shadow
    blobs = [(c + r.uniform(-1, 1) * radius * 0.55, c + r.uniform(-1, 1) * radius * 0.55, radius * r.uniform(0.5, 0.7))
             for _ in range(7)]
    pygame.draw.circle(s, (22, 80, 36), (c, c), int(radius * 0.95))
    for bx, by, br in blobs:
        pygame.draw.circle(s, (28, 96, 44), (int(bx), int(by)), int(br))
    for bx, by, br in blobs:
        pygame.draw.circle(s, (46, 132, 60), (int(bx - br * 0.15), int(by - br * 0.2)), int(br * 0.72))
    pygame.draw.circle(s, (92, 176, 92), (int(c - radius * 0.3), int(c - radius * 0.35)), max(3, int(radius * 0.26)))
    return s


def _make_bush(seed):
    s = pygame.Surface((64, 64), pygame.SRCALPHA)
    r = random.Random(seed)
    pygame.draw.ellipse(s, (0, 0, 0, 60), (12, 24, 44, 30))
    for _ in range(5):
        x, y, rad = 32 + r.randint(-9, 9), 30 + r.randint(-7, 7), r.randint(10, 14)
        pygame.draw.circle(s, (30, 100, 46), (x, y), rad)
        pygame.draw.circle(s, (56, 146, 70), (x - 2, y - 3), int(rad * 0.7))
    if r.random() < 0.6:                                               # a few flowers
        for _ in range(4):
            pygame.draw.circle(s, r.choice([(250, 220, 90), (245, 120, 150), (255, 255, 255)]),
                               (32 + r.randint(-12, 12), 28 + r.randint(-10, 10)), 2)
    return s


def _make_rock(seed):
    s = pygame.Surface((56, 48), pygame.SRCALPHA)
    r = random.Random(seed)
    pygame.draw.ellipse(s, (0, 0, 0, 60), (8, 24, 42, 20))
    pts = [(10 + r.randint(-3, 3), 30), (16, 14 + r.randint(-3, 3)), (30, 8 + r.randint(-3, 3)),
           (44, 16 + r.randint(-3, 3)), (48 + r.randint(-3, 3), 32), (30, 38)]
    pygame.draw.polygon(s, (116, 116, 124), pts)
    pygame.draw.polygon(s, (160, 160, 170), [pts[1], pts[2], pts[3], (30, 22)])
    pygame.draw.polygon(s, (70, 70, 78), pts, 2)
    return s


def _make_peak(w, hgt, seed):
    s = pygame.Surface((w + 24, hgt + 28), pygame.SRCALPHA)
    r = random.Random(seed)
    base = hgt + 14
    px = 12 + w // 2 + r.randint(-w // 8, w // 8)
    pygame.draw.ellipse(s, (0, 0, 0, 60), (4, base - 10, w + 16, 22))
    pygame.draw.polygon(s, (86, 86, 98), [(10, base), (px, 12), (px + 6, base)])                 # shaded face
    pygame.draw.polygon(s, (150, 150, 162), [(px, 12), (w + 14, base), (px + 6, base)])           # lit face
    pygame.draw.polygon(s, (250, 250, 255), [(px, 12), (px - w * 0.13, 12 + hgt * 0.27), (px - 2, 12 + hgt * 0.2),
                                             (px + w * 0.05, 12 + hgt * 0.3), (px + w * 0.12, 12 + hgt * 0.25)])
    pygame.draw.polygon(s, (60, 60, 72), [(10, base), (px, 12), (w + 14, base)], 2)
    return s


def _make_puddle(seed):
    s = pygame.Surface((84, 50), pygame.SRCALPHA)
    r = random.Random(seed)
    for _ in range(3):
        x, y, w, h = r.randint(4, 30), r.randint(6, 22), r.randint(30, 50), r.randint(16, 24)
        pygame.draw.ellipse(s, (120, 88, 58), (x - 2, y - 2, w + 4, h + 4))
        pygame.draw.ellipse(s, (82, 58, 38), (x, y, w, h))
        pygame.draw.ellipse(s, (104, 78, 52), (x + 5, y + 3, w // 3, h // 3))
    return s


def _make_car(body, dark, light, stripe):
    s = pygame.Surface((30, 56), pygame.SRCALPHA)
    for x, y in ((1, 9), (23, 9), (1, 37), (23, 37)):                   # wheels
        pygame.draw.rect(s, (24, 24, 28), (x, y, 6, 13), border_radius=2)
        pygame.draw.rect(s, (70, 70, 78), (x + 1, y + 2, 2, 9))
    pygame.draw.rect(s, (30, 30, 36), (7, 5, 16, 4))                    # front wing
    pygame.draw.rect(s, body, (4, 3, 22, 3))
    hull = [(11, 4), (19, 4), (23, 13), (24, 40), (21, 52), (9, 52), (6, 40), (7, 13)]
    pygame.draw.polygon(s, body, hull)
    pygame.draw.polygon(s, light, [(11, 6), (14, 6), (13, 48), (10, 46), (9, 14)])           # lit side
    pygame.draw.polygon(s, dark, [(19, 6), (22, 14), (23, 40), (20, 50), (17, 48), (18, 12)])   # shaded side
    pygame.draw.rect(s, stripe, (14, 4, 2, 47))                          # racing stripes
    pygame.draw.rect(s, stripe, (17, 4, 1, 47))
    pygame.draw.polygon(s, (24, 34, 66), [(11, 17), (19, 17), (20, 27), (10, 27)])           # windshield
    pygame.draw.polygon(s, (110, 150, 210), [(12, 18), (15, 18), (14, 22), (11, 22)])        # glass glint
    pygame.draw.rect(s, dark, (11, 27, 8, 8), border_radius=2)                               # roof / engine
    pygame.draw.polygon(s, (24, 34, 66), [(11, 36), (19, 36), (18, 40), (12, 40)])           # rear window
    pygame.draw.rect(s, (255, 240, 150), (8, 5, 3, 3))                                       # headlights
    pygame.draw.rect(s, (255, 240, 150), (19, 5, 3, 3))
    pygame.draw.rect(s, (26, 26, 32), (4, 50, 22, 4))                                        # rear spoiler
    pygame.draw.rect(s, (200, 40, 40), (4, 50, 3, 4))
    pygame.draw.rect(s, (200, 40, 40), (23, 50, 3, 4))
    pygame.draw.polygon(s, (20, 20, 24), hull, 1)
    return s


def _make_glow(w, h, color):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for i in range(7):
        pygame.draw.rect(s, (color[0], color[1], color[2], 16 + i * 17), (i * 3, i * 3, w - i * 6, h - i * 6),
                         border_radius=12)
    return s


def _make_box(font_big):
    s = pygame.Surface((44, 46), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0, 0, 0, 70), (6, 34, 34, 10))
    pygame.draw.rect(s, (176, 118, 8), (5, 10, 34, 28), border_radius=6)                    # side (3D look)
    pygame.draw.rect(s, (255, 206, 44), (5, 5, 34, 28), border_radius=6)                    # top
    pygame.draw.rect(s, (120, 78, 0), (5, 5, 34, 28), 2, border_radius=6)
    pygame.draw.polygon(s, (255, 244, 170), [(9, 8), (22, 8), (9, 20)])                      # gloss
    q = font_big.render("?", True, (90, 50, 0))
    s.blit(q, (22 - q.get_width() // 2, 19 - q.get_height() // 2))
    return s


def _build_scenery(track):
    """Trees, bushes, rocks, mountain peaks and mud puddles beside the road (placed once)."""
    rnd = random.Random(11)
    items = []                                                           # (y, x, kind, variant)

    def extent(y):
        roads = track.active(min(max(y, 0), track.LENGTH)) or [track.nearest(0, y)[0]]
        return roads, min(p.center_x(y) - p.width / 2 for p in roads), max(p.center_x(y) + p.width / 2 for p in roads)

    def in_zone(kind, y, pad=60):
        return any(z.kind == kind and z.y0 - pad <= y <= z.y1 + pad for z in track.terrain)

    for y in range(-300, track.LENGTH + 500, 56):
        roads, left, right = extent(y)
        for side, edge in ((-1, left), (1, right)):
            for ring in (0, 1):
                if rnd.random() > (0.78 if ring == 0 else 0.5):
                    continue
                d = 46 + ring * 150 + rnd.random() * 110
                roll = rnd.random()
                if in_zone("mountain", y):
                    kind = "rock" if roll < 0.5 else "bush"
                else:
                    kind = "tree" if roll < 0.66 else "bush" if roll < 0.88 else "rock"
                variant = rnd.randrange(3)
                items.append((y + rnd.uniform(-18, 18), edge + side * d, kind, variant))
        # trees in the grass island between two forked roads (when it is wide enough)
        ordered = sorted(roads, key=lambda p: p.center_x(y))
        for a, b in zip(ordered, ordered[1:]):
            free = (b.center_x(y) - b.width / 2) - (a.center_x(y) + a.width / 2)
            if free > 150 and rnd.random() < 0.7:
                mid = ((b.center_x(y) - b.width / 2) + (a.center_x(y) + a.width / 2)) / 2
                items.append((y, mid + rnd.uniform(-free / 6, free / 6), "tree" if free > 200 else "bush", rnd.randrange(3)))
    for z in track.terrain:
        if z.kind == "mountain":
            for y in range(int(z.y0) - 40, int(z.y1) + 40, 46):
                for side in (-1, 1):
                    items.append((y + rnd.uniform(-10, 10), z.path.center_x(y) + side * (z.path.width / 2 + 56 + rnd.random() * 40),
                                  "peak", rnd.randrange(3)))
        elif z.kind == "mud":
            for y in range(int(z.y0), int(z.y1), 70):
                for side in (-1, 1):
                    items.append((y + rnd.uniform(-10, 10), z.path.center_x(y) + side * (z.path.width / 2 + 36 + rnd.random() * 50),
                                  "puddle", rnd.randrange(3)))
    items.sort(key=lambda it: it[0])
    return items


def _build_assets(env):
    A = {"t": 0}
    A["f_small"] = pygame.font.SysFont("menlo,consolas,dejavusansmono", 14)
    A["f_norm"] = pygame.font.SysFont("menlo,consolas,dejavusansmono", 16)
    A["f_bold"] = pygame.font.SysFont("menlo,consolas,dejavusansmono", 16, bold=True)
    A["f_big"] = pygame.font.SysFont("menlo,consolas,dejavusansmono", 40, bold=True)
    A["f_box"] = pygame.font.SysFont("menlo,consolas,dejavusansmono", 26, bold=True)
    A["grass"] = _make_grass()
    A["sprites"] = {
        "tree": [_make_tree(r, 40 + r) for r in (34, 42, 50)],
        "bush": [_make_bush(s) for s in (1, 2, 3)],
        "rock": [_make_rock(s) for s in (4, 5, 6)],
        "peak": [_make_peak(w, h, s) for w, h, s in ((70, 62, 7), (86, 78, 8), (60, 80, 9))],
        "puddle": [_make_puddle(s) for s in (10, 11, 12)],
    }
    A["car"] = _make_car((214, 38, 44), (150, 18, 28), (240, 90, 90), (250, 250, 250))
    A["car_ghost"] = _make_car((140, 150, 255), (90, 100, 210), (200, 205, 255), (255, 255, 255))
    A["glow"] = {m: _make_glow(92, 72, c) for m, c in PAD_COLORS.items()}
    A["box"] = _make_box(A["f_box"])
    A["scenery"] = _build_scenery(env.track)
    A["scenery_y"] = [it[0] for it in A["scenery"]]
    A["panel_left"] = None
    return A


# =====================================================================================
#  Small drawing helpers
# =====================================================================================
def _text(screen, font, text, color, pos, shadow=True):
    if shadow:
        screen.blit(font.render(text, True, (0, 0, 0)), (pos[0] + 1, pos[1] + 1))
    screen.blit(font.render(text, True, color), pos)


def _panel(w, h, alpha=150):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (8, 12, 20, alpha), (0, 0, w, h), border_radius=12)
    pygame.draw.rect(s, (255, 255, 255, 60), (0, 0, w, h), 2, border_radius=12)
    return s


def _spawn(x, y, vx, vy, life, color, size):
    _S["particles"].append([x, y, vx, vy, life, life, color, size])


def _update_effects(env):
    """Spawn skid marks and particles from what the car is doing (purely visual)."""
    S = _S
    if env.steps < S["last_steps"]:                       # a new episode started: clear the leftovers
        S["particles"], S["marks"], S["wheels"] = [], [], None
        S["prev_heading"], S["prev_speed"] = env.heading, env.speed
        S["prev_rail"], S["prev_air"] = env.rail_hits, env.air_timer
    S["last_steps"] = env.steps
    S["t"] += 1
    ch, sh = math.cos(env.heading), math.sin(env.heading)

    def wheel(lx, ly):
        return env.x + lx * ch + ly * sh, env.y + (-lx * sh + ly * ch)

    rear = [wheel(-9, -16), wheel(9, -16)]
    turning = abs(env.heading - S["prev_heading"]) > 0.05 and env.speed > 110
    braking = env.speed < S["prev_speed"] - 7 and env.speed > 90
    if S["wheels"] and (turning or braking) and not env.airborne and env.stuck_timer == 0:
        for (px, py), (qx, qy) in zip(S["wheels"], rear):
            if abs(px - qx) < 40 and abs(py - qy) < 40:
                S["marks"].append((px, py, qx, qy, S["t"]))
    S["wheels"] = rear
    S["marks"] = [m for m in S["marks"] if S["t"] - m[4] < 260][-500:]

    bx, by = wheel(0, -24)
    if env.nitro_burning:                                  # nitro: blue-white exhaust
        for _ in range(3):
            _spawn(bx + random.uniform(-4, 4), by, random.uniform(-1, 1) - sh * 2, -3 - random.random() * 3, 12,
                   random.choice([(120, 190, 255), (200, 235, 255), (255, 255, 255)]), 5)
    elif env.boost_timer > 0:                              # boost pad trail
        col = PAD_COLORS.get(env.boost_mult, (255, 150, 0))
        for _ in range(2):
            _spawn(bx + random.uniform(-5, 5), by, random.uniform(-1, 1), -2 - random.random() * 2, 14, col, 4)
    terrain = env._terrain_here()
    if "mud" in terrain and env.speed > 40 and not env.airborne:
        for lx in (-9, 9):
            px, py = wheel(lx, -14)
            _spawn(px, py, random.uniform(-2, 2), random.uniform(-1, 2), 16, (96, 66, 40), 4)
    if "ice" in terrain and env.speed > 40 and not env.airborne:
        _spawn(env.x + random.uniform(-12, 12), env.y + random.uniform(-16, 16), 0, 0.3, 14, (235, 248, 255), 3)
    if env.rail_hits > S["prev_rail"]:                     # sparks when scraping a guard rail
        for _ in range(10):
            _spawn(env.x, env.y, random.uniform(-3, 3), random.uniform(-2, 4), 12, (255, 220, 90), 3)
    if S["prev_air"] > 0 and env.air_timer == 0:           # landing dust
        for _ in range(16):
            a = random.uniform(0, 6.28)
            _spawn(env.x, env.y, math.cos(a) * 2.5, math.sin(a) * 2.5, 18, (200, 184, 150), 6)
    if env.stuck_timer > 0 and S["t"] % 3 == 0:
        _spawn(env.x + random.uniform(-10, 10), env.y - 10, random.uniform(-1, 1), 1, 16, (96, 66, 40), 5)
    S["prev_heading"], S["prev_speed"] = env.heading, env.speed
    S["prev_rail"], S["prev_air"] = env.rail_hits, env.air_timer
    for p in S["particles"]:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= 1
    S["particles"] = [p for p in S["particles"] if p[4] > 0][-400:]


# =====================================================================================
#  The main draw function
# =====================================================================================
def draw(screen, font, env, info):
    global _cam_x, _A
    if _A is None:
        _A = _build_assets(env)
    A, track = _A, env.track
    _update_effects(env)
    t = _S["t"]

    if abs(env.x - _cam_x) > 250:                 # big jump (new episode / teleport): snap the camera
        _cam_x = env.x
    else:
        _cam_x += (env.x - _cam_x) * 0.15

    y_lo = env.y - (H - CAR_SCREEN_Y) - 80          # visible slice of the track
    y_hi = env.y + CAR_SCREEN_Y + 80

    # ---- Grass (a tiled texture that scrolls with the world) ----
    gx = int((W // 2 - _cam_x) % 200) - 200
    gy = int((CAR_SCREEN_Y + env.y) % 200) - 200
    for ox in range(gx, W, 200):
        for oy in range(gy, H, 200):
            screen.blit(A["grass"], (ox, oy))

    # ---- Scenery: trees, bushes, rocks, peaks, puddles (far ones first so near ones overlap them) ----
    lo = bisect.bisect_left(A["scenery_y"], y_lo - 80)
    hi = bisect.bisect_right(A["scenery_y"], y_hi + 80)
    for wy, wx, kind, variant in reversed(A["scenery"][lo:hi]):
        spr = A["sprites"][kind][variant]
        sx, sy = world_to_screen(env, wx, wy)
        screen.blit(spr, (sx - spr.get_width() // 2, sy - spr.get_height() // 2))

    # ---- Grass gap (a chasm): only a jump gets you across ----
    for g0, g1 in track.gaps:
        if g1 > y_lo and g0 < y_hi:
            gxs, gyb = world_to_screen(env, -104, g0)
            _, gyt = world_to_screen(env, -104, g1)
            h = gyb - gyt
            pygame.draw.rect(screen, (120, 86, 52), (gxs - 6, gyt - 6, 220, h + 12), border_radius=6)   # cliff edge
            for i in range(10):                                                                           # deepening dark bands
                shade = 52 - i * 5
                pygame.draw.rect(screen, (shade // 2, shade // 2, shade), (gxs + i * 2, gyt + i * h / 20 + 2,
                                                                          208 - i * 4, h - i * h / 10 - 4))
            for k in range(0, 208, 16):                                                                   # jagged rim
                pygame.draw.polygon(screen, (92, 64, 38), [(gxs + k, gyt), (gxs + k + 8, gyt + 10), (gxs + k + 16, gyt)])
                pygame.draw.polygon(screen, (92, 64, 38), [(gxs + k, gyb), (gxs + k + 8, gyb - 10), (gxs + k + 16, gyb)])
            label = A["f_bold"].render("GAP - JUMP!", True, (255, 255, 255))
            screen.blit(label, (gxs + 104 - label.get_width() // 2, (gyt + gyb) // 2 - label.get_height() // 2))

    # ---- Roads: striped curbs first, then asphalt (terrain-colored) on top, so overlapping roads merge cleanly ----
    step = 20
    quads = []
    for path in track.paths:
        hw = path.width / 2
        d0, d1 = _span(track, path)
        wy = max(d0, int(y_lo // step) * step)
        while wy < min(d1, y_hi):
            wy2 = min(wy + step, d1)
            x0, y0 = world_to_screen(env, path.center_x(wy), wy)
            x1, y1 = world_to_screen(env, path.center_x(wy2), wy2)
            fill = [(x0 - hw, y0 + 1), (x0 + hw, y0 + 1), (x1 + hw, y1), (x1 - hw, y1)]
            border = [(x0 - hw - 6, y0 + 1), (x0 + hw + 6, y0 + 1), (x1 + hw + 6, y1), (x1 - hw - 6, y1)]
            curb = (214, 40, 44) if (int(wy) // step) % 2 == 0 else (244, 244, 244)
            quads.append((border, fill, curb, ROAD_COLORS.get(track.terrain_at(path, wy), ROAD_COLORS[None])))
            wy = wy2
    for border, _, curb, _ in quads:
        pygame.draw.polygon(screen, curb, border)
    for _, fill, _, color in quads:
        pygame.draw.polygon(screen, color, fill)

    # asphalt grain, mud puddles, ice cracks, rock chips (deterministic, so nothing flickers)
    for pi, path in enumerate(track.paths):
        hw = path.width / 2
        d0, d1 = _span(track, path)
        wy = max(d0, int(y_lo // 14) * 14)
        while wy < min(d1, y_hi):
            kind = track.terrain_at(path, wy)
            dark, light = SPECKS[kind]
            sx, sy = world_to_screen(env, path.center_x(wy), wy)
            for i in range(4):
                hv = _h(wy, i, pi)
                off = ((hv % 1000) / 1000 - 0.5) * (path.width - 18)
                pygame.draw.rect(screen, dark if (hv >> 10) % 2 else light, (sx + off, sy + (hv >> 12) % 12, 2, 2))
            hv = _h(wy, 7, pi)
            off = ((hv % 1000) / 1000 - 0.5) * (path.width - 50)
            if kind == "mud" and (wy // 14) % 3 == 0:
                w, h = 18 + hv % 14, 8 + hv % 7
                pygame.draw.ellipse(screen, (122, 90, 60), (sx + off - 2, sy - 2, w + 4, h + 4))
                pygame.draw.ellipse(screen, (70, 48, 30), (sx + off, sy, w, h))
            elif kind == "ice" and (wy // 14) % 4 == 0:
                pygame.draw.lines(screen, (240, 250, 255), False,
                                  [(sx + off, sy), (sx + off + 9, sy - 7), (sx + off + 5, sy - 15), (sx + off + 16, sy - 22)], 2)
            elif kind == "mountain" and (wy // 14) % 3 == 0:
                pygame.draw.polygon(screen, (74, 70, 66), [(sx + off, sy + 8), (sx + off + 6, sy), (sx + off + 13, sy + 8)])
            wy += 14

    # ---- Coins: gold discs that disappear once collected ----
    for i, coin in enumerate(track.coins):
        if i in env.coins_used or not (y_lo < coin.y < y_hi):
            continue
        cx, cy = world_to_screen(env, coin.x, coin.y + coin.length / 2)
        bob = 3 * math.sin(t * 0.15 + i)
        cy -= bob
        pygame.draw.circle(screen, (70, 50, 0), (int(cx) + 2, int(cy) + 3), 11)
        pygame.draw.circle(screen, (214, 150, 16), (int(cx), int(cy)), 11)
        pygame.draw.circle(screen, (255, 214, 64), (int(cx), int(cy)), 8)
        pygame.draw.circle(screen, (255, 246, 190), (int(cx) - 3, int(cy) - 3), 3)

    # center line dashes
    for path in track.paths:
        d0, d1 = _span(track, path)
        wy = max(d0, int(y_lo // 60) * 60)
        while wy < min(d1, y_hi):
            sx, sy = world_to_screen(env, path.center_x(wy), wy)
            pygame.draw.rect(screen, (236, 220, 120), (sx - 2, sy - 15, 4, 30), border_radius=2)
            wy += 60

    # ---- Start line and finish line ----
    first, last = track.paths[0], track.paths[-1]
    sx, sy = world_to_screen(env, first.center_x(0), 0)
    pygame.draw.rect(screen, (255, 255, 255), (sx - 100, sy - 3, 200, 7))
    _text(screen, A["f_small"], "START", (255, 255, 255), (sx - 18, sy + 8))
    fx, fy = world_to_screen(env, last.center_x(env.TRACK_LENGTH), env.TRACK_LENGTH)
    sq = 20
    for i in range(int(last.width) // sq):
        for j in range(2):
            color = (255, 255, 255) if (i + j) % 2 == 0 else (20, 20, 20)
            pygame.draw.rect(screen, color, (fx - last.width / 2 + i * sq, fy - j * sq, sq, sq))
    for side in (-1, 1):                                                  # flags on poles
        px = fx + side * (last.width / 2 + 16)
        pygame.draw.line(screen, (60, 60, 66), (px, fy + 6), (px, fy - 44), 4)
        pygame.draw.polygon(screen, (230, 40, 50), [(px, fy - 44), (px + side * 24, fy - 36), (px, fy - 28)])
    _text(screen, A["f_bold"], "FINISH", (255, 255, 255), (fx - 26, fy - 62))

    # ---- Tire marks ----
    for x0, y0, x1, y1, born in _S["marks"]:
        if y_lo < y0 < y_hi:
            a = world_to_screen(env, x0, y0)
            b = world_to_screen(env, x1, y1)
            fade = max(0, 1 - (t - born) / 260)
            c = int(26 + 36 * (1 - fade))
            pygame.draw.line(screen, (c, c, c + 2), a, b, 3)

    # ---- Guard rails: metal rail with posts ----
    for rail in track.rails:
        hw = rail.path.width / 2 + 3
        for side, on in ((-1, rail.left), (1, rail.right)):
            if not on:
                continue
            ys = list(range(int(max(rail.y0, y_lo)) // 20 * 20, int(min(rail.y1, y_hi)) + 1, 20))
            pts = [world_to_screen(env, rail.path.center_x(wy) + side * hw, wy) for wy in ys]
            if len(pts) >= 2:
                pygame.draw.lines(screen, (70, 76, 88), False, pts, 9)
                pygame.draw.lines(screen, (205, 212, 222), False, pts, 5)
            for wy in range(int(max(rail.y0, y_lo)) // 40 * 40, int(min(rail.y1, y_hi)) + 1, 40):
                px, py = world_to_screen(env, rail.path.center_x(wy) + side * hw, wy)
                pygame.draw.circle(screen, (60, 66, 78), (int(px), int(py)), 5)
                pygame.draw.circle(screen, (150, 158, 170), (int(px) - 1, int(py) - 1), 2)

    # ---- Boost pads: glowing pad with arrows (they turn gray once used this episode) ----
    for i, pad in enumerate(track.pads):
        if not (y_lo < pad.y < y_hi):
            continue
        used = i in env.pads_used
        color = PAD_USED if used else PAD_COLORS[pad.mult]
        px, py = world_to_screen(env, pad.x, pad.y)                       # bottom-center of the pad
        if not used:
            glow = A["glow"][pad.mult]
            glow.set_alpha(150 + int(90 * math.sin(t * 0.18)))
            screen.blit(glow, (px - 46, py - pad.length // 2 - 36))
        rect = (px - pad.half_width, py - pad.length, pad.half_width * 2, pad.length)
        pygame.draw.rect(screen, tuple(int(c * 0.55) for c in color), (rect[0], rect[1] + 3, rect[2], rect[3]), border_radius=7)
        pygame.draw.rect(screen, color, rect, border_radius=7)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=7)
        shift = int(t * 0.4) % 10 if not used else 0                       # chevrons scroll forward
        for cxo in (-21, 21):
            for k in range(2):
                yy = py - 8 - k * 14 - shift % 10 // 2
                pygame.draw.lines(screen, (255, 255, 255) if not used else (170, 170, 175), False,
                                  [(px + cxo - 7, yy + 6), (px + cxo, yy), (px + cxo + 7, yy + 6)], 3)
        lab = A["f_bold"].render(f"{pad.mult}x", True, (20, 20, 20))
        screen.blit(lab, (px - lab.get_width() // 2, py - pad.length // 2 - lab.get_height() // 2))

    # ---- Jump ramps ----
    for i, ramp in enumerate(track.ramps):
        if not (y_lo < ramp.y < y_hi):
            continue
        used = i in env.ramps_used
        rx, ry = world_to_screen(env, ramp.x, ramp.y)
        hw, ln = ramp.half_width, ramp.length
        base = PAD_USED if used else (62, 104, 226)
        pygame.draw.polygon(screen, (0, 0, 0), [(rx - hw + 4, ry + 5), (rx + hw + 4, ry + 5), (rx + hw - 6, ry - ln + 5), (rx - hw + 14, ry - ln + 5)])
        pygame.draw.polygon(screen, tuple(int(c * 0.6) for c in base), [(rx - hw, ry), (rx + hw, ry), (rx + hw, ry - 6), (rx - hw, ry - 6)])
        pygame.draw.polygon(screen, base, [(rx - hw, ry - 6), (rx + hw, ry - 6), (rx + hw - 10, ry - ln), (rx - hw + 10, ry - ln)])
        for side in (-1, 1):                                               # hazard stripes on the sides
            for k in range(4):
                x = rx + side * (hw - 6) - side * k * 0
                col = (250, 214, 40) if k % 2 == 0 else (30, 30, 30)
                pygame.draw.polygon(screen, col, [(x - 4, ry - 6 - k * 8), (x + 4, ry - 6 - k * 8), (x + 4, ry - 12 - k * 8), (x - 4, ry - 12 - k * 8)])
        for k in range(3):
            pygame.draw.lines(screen, (255, 255, 255) if not used else (170, 170, 175), False,
                              [(rx - 12, ry - 12 - k * 8), (rx, ry - 20 - k * 8), (rx + 12, ry - 12 - k * 8)], 3)
        pygame.draw.polygon(screen, (255, 255, 255), [(rx - hw + 10, ry - ln), (rx + hw - 10, ry - ln), (rx + hw, ry - 6), (rx - hw, ry - 6)], 2)

    # ---- Mystery boxes: bobbing gold cubes that vanish once opened ----
    for i, box in enumerate(track.boxes):
        if i in env.boxes_used or not (y_lo < box.y < y_hi):
            continue
        bx, by = world_to_screen(env, box.x, box.y)
        bob = 4 * math.sin(t * 0.12 + i)
        screen.blit(A["box"], (bx - 22, by - 40 + bob))
        for k in range(3):                                                  # twinkling sparkles
            a = t * 0.08 + k * 2.1 + i
            sx, sy = bx + math.cos(a) * 26, by - 22 + math.sin(a) * 18 + bob
            if (t + k * 7) % 20 < 12:
                pygame.draw.lines(screen, (255, 255, 220), False, [(sx - 3, sy), (sx + 3, sy)], 2)
                pygame.draw.lines(screen, (255, 255, 220), False, [(sx, sy - 3), (sx, sy + 3)], 2)

    # ---- Walls: striped barriers (red/white = fixed, purple/white = MOVING, with arrows) ----
    for wall in track.walls:
        if not (y_lo < wall.y < y_hi):
            continue
        wx, wy_ = world_to_screen(env, wall.x_at(env.steps), wall.y)
        left, width = wx - wall.half_width, wall.half_width * 2
        top = wy_ - wall.thick
        pygame.draw.rect(screen, (0, 0, 0), (left + 4, top + 7, width, wall.thick), border_radius=5)          # shadow
        pygame.draw.rect(screen, (210, 210, 214), (left, top, width, wall.thick), border_radius=5)
        main = (150, 60, 214) if wall.moving else (214, 40, 44)
        for k in range(int(width // 20) + 1):
            w = min(20, width - k * 20)
            if w > 0 and k % 2 == 0:
                pygame.draw.rect(screen, main, (left + k * 20, top, w, wall.thick))
        pygame.draw.rect(screen, (255, 255, 255), (left, top, width, 5))                                      # lit top edge
        pygame.draw.rect(screen, (30, 30, 36), (left, top, width, wall.thick), 3, border_radius=5)
        if wall.moving:
            for side in (-1, 1):
                pygame.draw.polygon(screen, (255, 255, 255), [(wx + side * 10, top + 8), (wx + side * 22, top + 15), (wx + side * 10, top + 22)])

    # ---- Portals (hidden unless debug view is on with the P key) ----
    if info.get("show_portal") and env.portals_enabled:
        for portal in track.portals:
            corners = [world_to_screen(env, portal.path.center_x(wy) + off, wy)
                       for wy, off in [(portal.y0, portal.off0), (portal.y0, portal.off1),
                                       (portal.y1, portal.off1), (portal.y1, portal.off0)]]
            pygame.draw.polygon(screen, (170, 70, 220), corners)
            pygame.draw.polygon(screen, (255, 255, 255), corners, 2)

    # ---- Particles (trails, dust, sparks, mud splashes) ----
    for px, py, _, _, life, maxlife, col, size in _S["particles"]:
        if y_lo < py < y_hi:
            sx, sy = world_to_screen(env, px, py)
            pygame.draw.circle(screen, col, (int(sx), int(sy)), max(1, int(size * life / maxlife)))

    # ---- Car: shadow, flames, then the sprite (rotated; bigger and lifted while airborne) ----
    cx, cy = world_to_screen(env, env.x, env.y)
    scale, lift = CAR_SCALE, 0.0
    if env.airborne:
        progress = 1 - env.air_timer / env.AIR_STEPS
        hgt = math.sin(math.pi * progress)
        scale, lift = CAR_SCALE * (1 + 0.5 * hgt), 45 * hgt
    shadow = pygame.Surface((44, 64), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 90), (4, 4, 36, 56))
    shadow = pygame.transform.rotate(shadow, -math.degrees(env.heading))
    screen.blit(shadow, (cx - shadow.get_width() / 2 + 5, cy - shadow.get_height() / 2 + 5))
    ch, sh = math.cos(env.heading), math.sin(env.heading)

    def to_screen(lx, ly):                                       # local car coords -> screen
        return (cx + (lx * ch + ly * sh) * scale, cy - lift - (-lx * sh + ly * ch) * scale)

    if env.nitro_burning:
        ln = 40 + random.randint(0, 14)
        pygame.draw.polygon(screen, (60, 130, 255), [to_screen(-8, -26), to_screen(8, -26), to_screen(0, -26 - ln)])
        pygame.draw.polygon(screen, (210, 240, 255), [to_screen(-4, -26), to_screen(4, -26), to_screen(0, -26 - ln * 0.6)])
    if env.boost_timer > 0:
        flame = PAD_COLORS.get(env.boost_mult, (255, 150, 0))
        pygame.draw.polygon(screen, flame, [to_screen(-7, -26), to_screen(7, -26), to_screen(0, -26 - 12 * env.boost_mult)])
    sprite = A["car_ghost"] if env.ghost_timer > 0 else A["car"]
    if scale != 1.0:
        sprite = pygame.transform.smoothscale(sprite, (int(sprite.get_width() * scale), int(sprite.get_height() * scale)))
    sprite = pygame.transform.rotate(sprite, -math.degrees(env.heading))
    if env.ghost_timer > 0:
        sprite.set_alpha(150)
    screen.blit(sprite, (cx - sprite.get_width() / 2, cy - lift - sprite.get_height() / 2))

    # ---- White flash right after teleporting ----
    if env.portal_flash > 0:
        flash = pygame.Surface((W, H))
        flash.fill((255, 255, 255))
        flash.set_alpha(int(220 * env.portal_flash / env.PORTAL_FLASH_STEPS))
        screen.blit(flash, (0, 0))

    if info.get("show_hud", True):                  # press H to hide the HUD (nice for presenting)
        _draw_hud(screen, A, env, info)
    pygame.display.flip()


# =====================================================================================
#  HUD
# =====================================================================================
def _best_points_line(info):
    """Compare the best real run to a fair ceiling.

    No jackpots on that run  -> % of skill-only max (coins + fast finish).
    N jackpots on that run   -> % of (skill-only max + N jackpots), so lucky boxes
    do not read as "191% of the max".
    """
    best, skill_max = info.get("best_points", "-"), info.get("max_score")
    if best in (None, "-") or not skill_max:
        return f"Best points {best}"
    best = int(best)
    jackpot = int(info.get("box_jackpot", 1250))
    box_pts = int(info.get("best_box_points", 0) or 0)
    n_jackpots = box_pts // jackpot if jackpot else 0
    compare_max = skill_max + n_jackpots * jackpot
    pct = 100 * best / compare_max
    if n_jackpots:
        label = "jackpot" if n_jackpots == 1 else "jackpots"
        return f"Best points {best} = {pct:.0f}% of max with {n_jackpots} {label}"
    return f"Best points {best} = {pct:.0f}% of the max"

def _draw_hud(screen, A, env, info):
    routes = info["routes"]
    mode = "FAST TRAIN" if info["fast"] else "WATCH"
    lines = [
        f"Model: {info.get('model_label', 'With portals' if env.portals_enabled else 'No portals')}   (press M to switch)",
        f"Episode {info['episode']}   Epsilon {info['epsilon']:.2f}",
        f"Time {env.steps * env.DT:5.1f}s   Points {env.score}",
        f"Coins {env.coins_collected}   Box pts {env.box_points}   Time bonus {int(env.time_points) if env.finished else '-'}",
        f"Finishes {info['wins']}   Crashes {info['crashes']}   Best {info['best_time']}",
        _best_points_line(info),
        f"Max ~{info.get('max_score', '-')} (est.); {info.get('max_score_luck', '-')} if every box is a jackpot",
        f"Last: {info['last']}",
        f"Boosts {env.boosts_hit}   Jumps {env.jumps}   Rail bumps {env.rail_hits}",
        f"Boxes {env.boxes_opened} this run | {info.get('boxes_total', 0)} total | last: {env.last_box or '-'}",
        f"Route: Scenic {routes['Scenic']} | Shortcut {routes['Shortcut']}   Portal: "
        f"{info['portals'] if env.portals_enabled else 'OFF'}",
        f"Mode: {mode}{' (GREEDY)' if info['greedy'] else ''}   Random starts: "
        f"{'ON' if info['random_starts'] else 'OFF'}{'' if env.from_start else ' [practice]'}",
    ]
    # Extra footer space so the controls line stays inside the panel (not on the border).
    footer_h = 32
    left_h = 18 + len(lines) * 20 + footer_h
    screen.blit(_panel(536, left_h), (8, 8))
    for i, text in enumerate(lines):
        _text(screen, A["f_norm"], text, (240, 244, 250), (24, 18 + i * 20))
    _text(screen, A["f_small"], "SPACE fast  G greedy  R random  M model  P show portal  H hide  S save  ESC",
          (150, 165, 190), (24, 18 + len(lines) * 20 + 6), shadow=False)

    # right gauge panel: speed, nitro bar, active effects
    gx = 554
    screen.blit(_panel(W - gx - 8, left_h), (gx, 8))
    slow = env.slow_timer > 0 or "mud" in env._terrain_here()
    shown = int(env.speed * (0.75 if slow else 1.0))
    _text(screen, A["f_big"], f"{shown}", (255, 255, 255), (gx + 14, 14))
    _text(screen, A["f_small"], "px/s", (160, 175, 200), (gx + 14 + 24 * len(str(shown)) + 6, 38))
    bar = pygame.Rect(gx + 14, 64, W - gx - 8 - 28, 16)
    pygame.draw.rect(screen, (20, 24, 34), bar, border_radius=8)
    fillw = int(bar.width * env.nitro)
    if fillw > 4:
        pygame.draw.rect(screen, (70, 150, 255) if not env.nitro_burning else (200, 235, 255),
                         (bar.x, bar.y, fillw, bar.height), border_radius=8)
        pygame.draw.rect(screen, (170, 210, 255), (bar.x + 3, bar.y + 3, max(fillw - 6, 1), 4), border_radius=2)
    pygame.draw.rect(screen, (255, 255, 255), bar, 2, border_radius=8)
    _text(screen, A["f_small"], "NITRO", (255, 255, 255), (bar.x + 6, bar.y - 1))

    pills = []
    if env.airborne:
        pills.append(("AIRBORNE", (62, 104, 226), (255, 255, 255)))
    if env.boost_timer > 0:
        pills.append((f"BOOST {env.boost_mult}x", PAD_COLORS.get(env.boost_mult, (255, 150, 0)), (20, 20, 20)))
    if env.nitro_burning:
        pills.append(("NITRO!", (90, 160, 255), (255, 255, 255)))
    if env.nitro_cut:
        pills.append(("NITRO CUT", (200, 70, 70), (255, 255, 255)))
    for name, t_, bg, fg in (("ICE", env.ice_timer, (150, 205, 235), (20, 30, 50)), ("SLOW", env.slow_timer, (150, 120, 80), (255, 255, 255)),
                             ("GHOST", env.ghost_timer, (170, 170, 255), (20, 20, 60))):
        if t_ > 0:
            pills.append((f"{name} {t_ / 30:.1f}s", bg, fg))
    if env.stuck_timer > 0:
        pills.append((f"STUCK {env.stuck_timer / 30:.1f}s", (200, 70, 70), (255, 255, 255)))
    ter = {"ice": ("on ICE", (150, 205, 235), (20, 30, 50)), "mud": ("on MUD", (104, 74, 46), (255, 255, 255)),
           "mountain": ("MOUNTAIN", (110, 104, 100), (255, 255, 255))}
    for k in sorted(env._terrain_here()):
        pills.append(ter[k])
    px, py = gx + 14, 92
    for label, bg, fg in pills[:8]:
        surf = A["f_small"].render(label, True, fg)
        w = surf.get_width() + 14
        if px + w > W - 16:
            px, py = gx + 14, py + 24
        if py > left_h - 14:
            break
        pygame.draw.rect(screen, bg, (px, py, w, 20), border_radius=10)
        screen.blit(surf, (px + 7, py + 1))
        px += w + 6