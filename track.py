"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
track.py - the road network: paths (with forks and gaps), coins, boost pads, ramps, walls,
rails, mystery boxes, terrain zones, and the hidden portal.

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

HOW TO EDIT THE TRACK: each Path is a list of control points (y, x). The road smoothly
eases from one point to the next (like an S-curve). Two paths that start and end at the
same point form a FORK. Two paths with empty space between them form a GAP (grass) that
only a jump can cross. Keep boost pads on STRAIGHT stretches (same x for several points),
because the car can't turn sharply at high speed.
"""
import math


def _ease(t):
    """Smooth 0 -> 1 curve with zero slope at both ends (an S-shape)."""
    return (1 - math.cos(math.pi * t)) / 2


class Path:
    """One road. Looks up its center line and direction at every 1 px of distance."""

    def __init__(self, name, pts, width):
        self.name = name
        self.pts = pts                      # [(y, x), ...] control points
        self.width = width
        self.y0, self.y1 = int(pts[0][0]), int(pts[-1][0])
        n = self.y1 - self.y0 + 1
        self._cx = [0.0] * n
        self._ang = [0.0] * n
        seg = 0
        for i in range(n):
            y = self.y0 + i
            while seg < len(pts) - 2 and y > pts[seg + 1][0]:
                seg += 1
            (ya, xa), (yb, xb) = pts[seg], pts[seg + 1]
            t = max(0.0, min(1.0, (y - ya) / (yb - ya)))
            self._cx[i] = xa + (xb - xa) * _ease(t)
            slope = (xb - xa) * (math.pi / 2) * math.sin(math.pi * t) / (yb - ya)
            self._ang[i] = math.atan(slope)

    def _i(self, y):
        return max(0, min(len(self._cx) - 1, int(y) - self.y0))

    def center_x(self, y):
        """World x of this road's center at distance y (clamped at the road's ends)."""
        return self._cx[self._i(y)]

    def angle(self, y):
        """Direction the road points at y (0 = straight ahead, + = to the right)."""
        return self._ang[self._i(y)]

    def curvature_ahead(self, y, horizon=500):
        """Max |d(heading)/dy| over the next `horizon` px (0 on a straight)."""
        i0 = self._i(y)
        i1 = min(len(self._ang) - 1, i0 + int(horizon))
        if i1 <= i0:
            return 0.0
        best = 0.0
        ang = self._ang
        for i in range(i0, i1):
            d = abs(ang[i + 1] - ang[i])
            if d > best:
                best = d
        return best


class Coin:
    """A coin on the road. Each one can be collected once per episode."""
    HALF_WIDTH = 16
    LENGTH = 40                              # longer than one step at top boost speed, so the car can't skip it

    def __init__(self, path, y, offset=0.0):
        self.path, self.y = path, y
        self.x = path.center_x(y) + offset
        self.half_width = self.HALF_WIDTH
        self.length = self.LENGTH


class Pad:
    """A visible boost pad. mult = 2 or 3 (speed multiplier)."""
    HALF_WIDTH = 30
    LENGTH = 40                              # must be longer than the farthest the car moves in 1 step

    def __init__(self, path, y, mult, offset=0.0):
        self.path, self.y, self.mult = path, y, mult
        self.x = path.center_x(y) + offset
        self.half_width = self.HALF_WIDTH
        self.length = self.LENGTH
        self.steps = 30 if mult == 2 else 24  # how long the boost lasts (in sim steps)


class Ramp:
    """A jump ramp. Drive onto it and the car flies for a moment (no steering while airborne)."""
    LENGTH = 40

    def __init__(self, path, y, half_width=50, offset=0.0):
        self.path, self.y = path, y
        self.x = path.center_x(y) + offset
        self.half_width = half_width
        self.length = self.LENGTH


class Box:
    """A mystery box (shown as a gold "?"). Drive into it for a random good or bad effect."""
    HALF_WIDTH = 22
    LENGTH = 40

    def __init__(self, path, y, offset=0.0):
        self.path, self.y = path, y
        self.x = path.center_x(y) + offset
        self.half_width = self.HALF_WIDTH
        self.length = self.LENGTH


class Wall:
    """A wall across part of the road. Hitting it stops the car and slows it down, like a guard rail.
    A ghost, or a car in the air, passes through. amplitude > 0 makes it a MOVING wall.
    gate=True spans the whole road. Hitting that one ends the run; the way past is to jump it."""
    THICK = 30

    def __init__(self, path, y, offset=0.0, half_width=40, amplitude=0.0, period=3.0,
                 phase=0.0, gate=False):
        self.path, self.y, self.offset = path, y, offset
        self.half_width = path.width / 2 + 20 if gate else half_width
        self.amplitude, self.period, self.phase = amplitude, period, phase
        self.gate = gate
        self.thick = self.THICK
        self.moving = amplitude > 0
        self._base = path.center_x(y) + offset

    def x_at(self, steps, dt=1 / 30):
        """World x of the wall's center `steps` simulation steps after the episode began."""
        if not self.moving:
            return self._base
        return self._base + self.amplitude * math.sin(2 * math.pi * steps * dt / self.period + self.phase)


class Rail:
    """Guard rails along the edge(s) of a road section. They push the car back instead of crashing."""

    def __init__(self, path, y0, y1, left=True, right=True):
        self.path, self.y0, self.y1, self.left, self.right = path, y0, y1, left, right


class Terrain:
    """A stretch of road with special ground: "ice" (half turning), "mud" (3/4 speed, can get
    stuck) or "mountain" (looks straight, but the climb makes the distance feel 1.5x longer)."""

    def __init__(self, path, y0, y1, kind):
        self.path, self.y0, self.y1, self.kind = path, y0, y1, kind


class Portal:
    """A HIDDEN teleporter: an invisible patch near the edge of a road. The agent is never told it
    exists. Driving over it (once per run) jumps the car PORTAL_DISTANCE px ahead (see environment.py)."""

    def __init__(self, path, y0, y1, off0, off1):
        self.path, self.y0, self.y1 = path, y0, y1
        self.off0, self.off1 = off0, off1            # px from the road's center line (negative = left)


class Track:
    LENGTH = 12000                           # finish line distance
    SECTION_BINS = 8                         # 1500 px slices: early, middle, and late, not every 500 px
    FORK_ZONE = (2400, 5700)                 # y-range where the Scenic / Shortcut fork exists

    ROUTE_Y = 3300                           # where we record which fork path the car chose

    def __init__(self, only_route=None):
        """Build the track. only_route="Scenic" or "Shortcut" removes the other fork (for route experiments)."""
        trunk1 = Path("Main road", [(0, 0), (600, 0), (900, 70), (1250, -70), (1600, 0), (2400, 0)], 200)
        scenic = Path("Scenic", [(2400, 0), (3100, -170), (3500, -240), (3900, -110), (4300, -240),
                                 (4600, -170), (5000, -170), (5700, 0)], 200)
        shortcut = Path("Shortcut", [(2400, 0), (3100, 170), (5000, 170), (5700, 0)], 160)
        # The final road is split in two with a GAP of grass between (8400 to 8650)
        trunk2a = Path("Main road", [(5700, 0), (6500, 0), (6900, -80), (7300, 80), (7700, -60),
                                     (8000, 0), (8400, 0)], 200)
        trunk2b = Path("Main road", [(8650, 0), (9000, 0)], 200)
        # Another stretch of the same kind of road: a straight, an S-curve, then a straight to the finish.
        trunk3 = Path("Main road", [(9000, 0), (9700, 0), (10150, 90), (10600, -90),
                                    (11050, 70), (11450, 0), (12000, 0)], 200)
        self.paths = [trunk1, scenic, shortcut, trunk2a, trunk2b, trunk3]
        self.gaps = [(8400, 8650)]
        self.fork_names = ("Scenic", "Shortcut")

        # Hidden portals on an outer edge. Each one jumps the car PORTAL_DISTANCE px ahead. Small patches (30 x 10 px)
        # so the agent has to find them, not drive over them by accident.
        self.portals = [
            Portal(scenic, 3795, 3825, -87, -77),          # outer-left edge of Scenic (a small 30 x 10 px patch)
            Portal(shortcut, 3165, 3195, 56, 66),          # outer-right edge of Shortcut
            Portal(trunk3, 10495, 10525, -87, -77),        # outer-left edge of the last S-curve
        ]

        # Coins. One route can collect the trunk coins plus one fork. They sit on the
        # road, a few of them off the center line so steering toward them is a real choice.
        self.coins = [
            Coin(trunk1, 250),
            Coin(trunk1, 700, offset=-45),
            Coin(trunk1, 1100),
            Coin(trunk1, 1500, offset=40),
            Coin(trunk1, 2050),
            Coin(trunk1, 2350, offset=-35),
            Coin(scenic, 2800),
            Coin(scenic, 3400, offset=-40),
            Coin(scenic, 4200),
            Coin(scenic, 4700, offset=35),
            Coin(scenic, 5300),
            Coin(shortcut, 2900, offset=30),
            Coin(shortcut, 3700),
            Coin(shortcut, 4300, offset=-25),
            Coin(shortcut, 4900),
            Coin(trunk2a, 5800),
            Coin(trunk2a, 6500, offset=40),
            Coin(trunk2a, 7200),
            Coin(trunk2a, 7800, offset=-30),
            Coin(trunk2b, 8850),
            Coin(trunk3, 9080),
            Coin(trunk3, 9480, offset=35),
            Coin(trunk3, 9900),
            Coin(trunk3, 10550, offset=-40),
            Coin(trunk3, 11200),
            Coin(trunk3, 11820),
        ]
        # Boost pads (orange = 2x, cyan = 3x)
        self.pads = [
            Pad(trunk1, 1700, 2),         # boost + jump combo on the first straight
            Pad(shortcut, 3300, 3),       # big boost into a long jump on the Shortcut
            Pad(scenic, 4650, 3),
            Pad(trunk2a, 5900, 2),        # boost into the jump over the gate wall
            Pad(trunk2a, 8100, 3),        # boost into the jump across the grass gap
            Pad(trunk3, 9150, 2),         # boost into the jump on the last straight
        ]
        # Jump ramps (the ramp points the car along the road as it takes off).
        # Keep a straight, wall-free landing zone after each one!
        self.ramps = [
            Ramp(trunk1, 1760),                      # paired with the 2x pad
            Ramp(shortcut, 3360, half_width=40),     # paired with the 3x pad (lands ~540 px later)
            Ramp(trunk2a, 6050),                     # jump over the gate wall right after it
            Ramp(trunk2a, 8300),                     # jump the 250 px gap: needs about 2x speed or more
            Ramp(trunk3, 9210),                      # lands near 9570, still on the straight
        ]
        # Walls: static ones in the middle block the road; moving ones slide side to side
        self.walls = [
            Wall(trunk1, 450, offset=0, half_width=40),                          # blocks the middle
            Wall(shortcut, 4650, amplitude=20, half_width=30, period=4.0),       # moving, narrow road
            Wall(scenic, 4900, amplitude=50, half_width=40, period=4.0),         # moving, wide road
            Wall(trunk2a, 6120, gate=True),                                      # full-width: jump it
            Wall(trunk3, 10820, amplitude=40, half_width=35, period=4.0),       # moving wall on the S-curve
        ]
        # Guard rails (no crash, just a bump) on tricky sections
        self.rails = [
            Rail(trunk1, 650, 1450, left=True, right=True),
            Rail(scenic, 3300, 4000, left=False, right=True),
            Rail(trunk3, 10000, 11100, left=True, right=True),
        ]
        # Mystery boxes (random good or bad effect, once per box per run)
        self.boxes = [
            Box(trunk1, 2250, offset=-40),
            Box(scenic, 3900, offset=0),
            Box(scenic, 4400, offset=40),
            Box(shortcut, 4200, offset=0),
            Box(trunk2a, 6650, offset=40),
            Box(trunk2a, 7550, offset=-30),
            Box(trunk3, 9600, offset=-30),
            Box(trunk3, 10950),
        ]
        # Terrain
        self.terrain = [
            Terrain(trunk1, 780, 1300, "ice"),
            Terrain(scenic, 4050, 4500, "ice"),
            Terrain(shortcut, 4000, 4300, "mud"),
            Terrain(trunk2a, 6700, 7050, "mud"),
            Terrain(shortcut, 4350, 4440, "mountain"),
            Terrain(trunk2a, 7800, 8050, "mountain"),
            Terrain(trunk3, 9900, 10250, "mountain"),
            Terrain(trunk3, 10320, 10740, "ice"),
            Terrain(trunk3, 11220, 11540, "mud"),
        ]

        # Optional: keep only one fork path (and drop objects that lived on the other).
        if only_route is not None:
            if only_route not in self.fork_names:
                raise ValueError(f"only_route must be one of {self.fork_names}, got {only_route!r}")
            drop = "Shortcut" if only_route == "Scenic" else "Scenic"
            self.paths = [p for p in self.paths if p.name != drop]
            for attr in ("coins", "pads", "ramps", "boxes", "walls", "rails", "terrain", "portals"):
                setattr(self, attr, [o for o in getattr(self, attr) if o.path.name != drop])
            self.fork_names = (only_route,)

        for lst in (self.coins, self.pads, self.ramps, self.boxes, self.walls):
            lst.sort(key=lambda o: o.y)

        # Parallel y-arrays for binary search in environment.py (must stay in sync with the lists).
        self.coin_ys = [o.y for o in self.coins]
        self.pad_ys = [o.y for o in self.pads]
        self.ramp_ys = [o.y for o in self.ramps]
        self.box_ys = [o.y for o in self.boxes]
        self.wall_ys = [o.y for o in self.walls]
        self.coin_len = max((o.length for o in self.coins), default=0)
        self.pad_len = max((o.length for o in self.pads), default=0)
        self.ramp_len = max((o.length for o in self.ramps), default=0)
        self.box_len = max((o.length for o in self.boxes), default=0)
        self.wall_thick = Wall.THICK

        # Which paths exist at each distance (precomputed so lookups are fast)
        self._active = [[p for p in self.paths if p.y0 <= y <= p.y1] for y in range(self.LENGTH + 1)]

    def active(self, y):
        return self._active[max(0, min(self.LENGTH, int(y)))]

    def in_gap(self, y):
        return any(g0 < y < g1 for g0, g1 in self.gaps)

    def terrain_at(self, path, y):
        """Terrain kind on `path` at distance y (or None)."""
        for z in self.terrain:
            if z.path is path and z.y0 <= y <= z.y1:
                return z.kind
        return None

    def nearest(self, x, y):
        """The road whose center is closest to x. Returns (path, x minus its center).
        In a gap (no road), it uses the road closest in distance."""
        act = self.active(y)
        if not act:
            act = [min(self.paths, key=lambda p: (max(p.y0 - y, y - p.y1, 0), abs(x - p.center_x(y))))]
        best, best_d = None, 1e9
        for p in act:
            d = abs(x - p.center_x(y))
            if d < best_d:
                best, best_d = p, d
        return best, x - best.center_x(y)

    def on_road(self, x, y, car_half):
        """True if the car (center x, half-width car_half) is fully on any active road."""
        if y < 0:
            return False
        for p in self.active(y):
            if abs(x - p.center_x(y)) + car_half <= p.width / 2:
                return True
        return False

    def road_ahead(self, path, y, look):
        """Which road the car will be on `look` px from now (follows the road past a path's end)."""
        yl = y + look
        if yl <= path.y1:
            return path
        act = self.active(yl)
        if not act:                              # looking across a gap: keep using this road
            return path
        ref = path.center_x(yl)
        return min(act, key=lambda p: abs(p.center_x(yl) - ref))