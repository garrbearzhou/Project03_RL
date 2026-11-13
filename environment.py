"""
environment.py - the racing "world" (the Environment in the agent-environment loop).

AI USE DISCLOSURE (Honor Code): This file was generated with help from Claude (Anthropic)
and upgraded over several versions (curves, hidden portal, track-section state, boost pads,
forks, walls, guard rails, jumps, grass gap, mystery boxes, nitro, terrain).
I read through it, and I understand how it works.
<-- Add your own notes here about what you changed or wrote yourself.

The environment knows the physics and the rules. The AGENT does not know them:
it only sees a small discretized state, picks an action, and gets a reward.
The road layout itself lives in track.py.
"""
import math
import random

from track import Track


class RacingEnv:
    CAR_HALF_WIDTH = 10       # px, used for the "off the road" check

    # ---- Car physics ----
    DT = 1 / 30               # seconds per simulation step
    MAX_SPEED = 300           # px/s (normal top speed)
    ACCEL = 200               # px/s^2 when throttle is held
    BRAKE = 300               # px/s^2 when braking
    DRAG = 40                 # px/s^2 natural slow-down when coasting
    BOOST_DECAY = 400         # px/s^2 slow-down back to normal speed after a boost ends
    TURN_RATE = 2.0           # rad/s at full steering
    MAX_STEPS = 2400          # episode time limit (80 simulated seconds)

    # ---- Jumps and rails ----
    AIR_STEPS = 18            # a jump lasts 0.6 s. Flight distance = speed x 0.6 s
    RAIL_SLOWDOWN = 0.8       # speed multiplier when the car scrapes a guard rail
    RAIL_PENALTY = 0.5        # small reward penalty per scrape (so rails aren't a free ride)
    # ---- Points (this is what the agent is trying to maximize) ----
    # Official score = coins collected × COIN_VALUE
    #                + one point per step still left on the clock, but ONLY if you finish
    #                − CRASH_PENALTY if you crash
    # A fast finish is worth far more than every coin on one route (about 400 coin points,
    # versus roughly 1200 time points for a 40 second finish). PROGRESS_SCALE is not part of
    # the score. It is a small "you moved forward" learning signal so a 9000 px track is
    # learnable. Scraping a rail is also a small reward penalty on top of the score.
    COIN_VALUE = 25
    TIME_POINT = 1            # points per unused step when the car finishes
    CRASH_PENALTY = 100
    PROGRESS_SCALE = 0.02
    STEP_COST = 0.0           # optional small cost per step (0 = off). Keep STEP_COST x MAX_STEPS below CRASH_PENALTY
                              # (so <= 0.04), otherwise crashing early can score better than running out of time.
                              # Tested at 0.05 and with gamma 0.995: neither beat the defaults in a short test.

    # ---- Nitro (actions 6-8: nitro straight / left / right): charges slowly, burns fast ----
    NITRO_CHARGE = 1 / 300    # per step -> 10 seconds to fill the bar
    NITRO_DRAIN = 1 / 90      # per step while burning -> a full bar lasts 3 seconds
    NITRO_MULT = 2            # nitro pushes you toward 2x speed
    NITRO_ACCEL = 600         # px/s^2 extra acceleration while burning
    NITRO_CUT_HEADING = 0.30  # nitro cuts out if the car points more than this many radians (~17 degrees) off the road

    # ---- Mystery box effects ----
    # A box helps about twice as often as it hurts (each good effect is in the bag twice).
    # At normal speed, a finish still in hand, the rough point value of each roll is:
    #   BOOST +60, GHOST +15 plus a free pass through a wall, WARP FORWARD +45, FULL NITRO +45
    #   ICE -10, SLOW -12, WARP BACK -20, NITRO DRAIN -20
    # The average is about +22 points, close to one coin: worth hitting on the way, not a long detour.
    # A bad roll is a stumble. It does not last long enough to throw the car off the road.
    BOOST_STEPS = 60          # 2 s at 2x speed
    GHOST_STEPS = 60          # 2 s
    GHOST_SPEED = 1.25        # ghost also moves a bit faster, so it is never a blank item
    BOX_ICE_STEPS = 45        # 1.5 s
    BOX_ICE_GRIP = 0.75       # track ice stays at ICE_GRIP (0.5); a box only loosens the steering
    SLOW_STEPS = 60           # 2 s
    WARP_FORWARD = 450        # px
    WARP_BACK = 200           # px. Shorter than the forward warp, so the pair still pays
    NITRO_STEAL = 0.25        # fraction of the nitro bar removed
    GOOD_EFFECTS = ["BOOST", "GHOST", "WARP FORWARD", "FULL NITRO"]
    BAD_EFFECTS = ["ICE", "SLOW", "WARP BACK", "NITRO DRAIN"]

    # ---- Terrain ----
    ICE_GRIP = 0.5            # turning is half as effective on ice
    MUD_SPEED = 0.75          # mud: 3/4 speed
    SLOW_SPEED = 0.8          # "slow" mystery box: 4/5 speed for 2 seconds
    MUD_STUCK_CHANCE_PER_SEC = 0.05   # chance per second in mud of getting stuck...
    STUCK_STEPS = 60          # ...for 2 seconds
    MOUNTAIN_FACTOR = 1.5     # a climb: you cover distance 1.5x slower

    # ---- Actions the agent can choose from ----
    ACCEL_STRAIGHT, ACCEL_LEFT, ACCEL_RIGHT, COAST, BRAKE_ACT, NITRO, NITRO_LEFT, NITRO_RIGHT = range(8)
    N_ACTIONS = 8
    NITRO_ACTIONS = (NITRO, NITRO_LEFT, NITRO_RIGHT)    # accelerate AND burn nitro (can still steer)

    # ---- What the agent can "see" ----
    # The agent looks FURTHER ahead the faster it goes: LOOKAHEAD + LOOK_TIME x speed. At 300 px/s that is
    # 280 px, at 2x boost 460 px, at 3x boost 640 px. (A fixed 150 px was only 0.25 s of road at 600 px/s,
    # so the agent kept burning nitro into curves it could not yet see.)
    LOOKAHEAD = 100           # px ahead at a standstill
    LOOK_TIME = 0.6           # extra seconds of travel it can see
    SIGHT = 300               # px ahead it can spot pads, ramps, walls and boxes
    PAD_SIDE = 140            # ...pads/ramps/boxes only if within this many px sideways
    OFFSET_BINS = 9           # where the car is across the road
    HEADING_BINS = 7          # which way the car is pointing (relative to the road)
    # speed has 6 buckets: 4 below normal top speed, then "2x boost", then "3x boost"
    # "ahead" has 17 values: 0 nothing | 1-3 boost pad left/ahead/right | 4-5 wall in my way,
    #     escape left/right | 6-8 jump ramp left/ahead/right | 9 airborne | 10-12 mystery box l/a/r
    #     | 13 (unused now; the portal no longer charges) | 14-16 coin left/ahead/right
    # "extras" has 6 values: (0 normal, 1 slippery, 2 ghost) x (nitro bar below / at least half full)

    # ---- Hidden portal: an immediate jump forward (visual flash only) ----
    PORTAL_FLASH_STEPS = 8
    PORTAL_DISTANCE = 600     # px ahead (was 1000: the trained agent used a portal in 100% of runs, so it was no
                              # longer a secret). The old 3 second charge is gone.

    def __init__(self, portals=True):
        self.portals_enabled = portals         # False = the "no portal" version of the game
        self.track = Track()
        self.TRACK_LENGTH = self.track.LENGTH
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, random_start=False):
        """Start a new episode.
        random_start=False: at the start line (this is the real race).
        random_start=True : somewhere random on the track - training practice, so the agent
                            sees every part of a long track instead of dying early every time."""
        self.from_start = not random_start
        if random_start:
            while True:
                y = random.uniform(0, self.TRACK_LENGTH - 300)
                roads = self.track.active(y)
                if roads:                                 # (not inside the grass gap)
                    break
            path = random.choice(roads)
            self.y = y
            self.x = path.center_x(y) + random.uniform(-0.5, 0.5) * (path.width / 2)
            self.heading = path.angle(y) + random.uniform(-0.2, 0.2)
            self.speed = random.uniform(0, self.MAX_SPEED)
        else:
            self.x = random.uniform(-60, 60)
            self.y = 0.0
            self.heading = random.uniform(-0.2, 0.2)   # radians, 0 = straight ahead, + = right
            self.speed = 0.0
        self.steps = 0
        self.score = 0
        self.time_points = 0
        self.finished = False
        self.crashed = False
        self.crash_reason = None      # "road", "wall" or "landing"
        self.portal_used = False      # the portal works once per episode
        self.portal_flash = 0
        self.pads_used = set()        # indexes of boost pads already touched this episode
        self.ramps_used = set()
        self.boxes_used = set()
        self.coins_used = set()
        self.coins_collected = 0
        self.boost_mult = 1
        self.boost_timer = 0
        self.boosts_hit = 0
        self.air_timer = 0            # > 0 while the car is flying
        self.jumps = 0
        self.landings = 0
        self.rail_hits = 0
        self.nitro = 0.5              # nitro bar, 0 to 1 (starts half full)
        self.nitro_burning = False
        self.nitro_cut = False        # True on steps where traction control cut the nitro
        self.ice_timer = 0            # mystery-box effects (steps remaining)
        self.slow_timer = 0
        self.ghost_timer = 0
        self.stuck_timer = 0          # stuck in the mud
        self.last_box = ""            # name of the last mystery effect (for the HUD)
        self.boxes_opened = 0
        self.route = None             # which fork path was taken ("Scenic" / "Shortcut")
        return self.get_state()

    @property
    def airborne(self):
        return self.air_timer > 0

    # ---------------------------------------------------------------- terrain
    def _terrain_here(self):
        """Set of terrain kinds under the car right now (a car must be on that road to count)."""
        kinds = set()
        for z in self.track.terrain:
            if z.y0 <= self.y <= z.y1 and abs(self.x - z.path.center_x(self.y)) <= z.path.width / 2 + 30:
                kinds.add(z.kind)
        return kinds

    # ------------------------------------------------------------------ state
    def get_state(self):
        """Turn the raw car numbers into a small tuple of buckets (the Q-table key).
        Everything is measured RELATIVE to the nearest road so the agent learns
        "stay on the road", not "go to x = 50"."""
        path, rel_x = self.track.nearest(self.x, self.y)
        terrain = self._terrain_here()
        slow_mult = self.MUD_SPEED if ("mud" in terrain or self.slow_timer > 0) else 1.0

        # 1) Where am I across the road? (-1 = left edge, +1 = right edge)
        offset = rel_x / (path.width / 2)
        o = int((offset + 1) / 2 * self.OFFSET_BINS)
        o = max(0, min(self.OFFSET_BINS - 1, o))

        # 2) Am I pointing along the road? (heading minus road direction)
        rel = max(-0.6, min(0.6, self.heading - path.angle(self.y)))
        hb = int((rel + 0.6) / 1.2 * self.HEADING_BINS)
        hb = max(0, min(self.HEADING_BINS - 1, hb))

        # 3) How fast am I really going? (0-3 normal, 4 = about 2x boost, 5 = 3x boost)
        speed = self.speed * slow_mult
        if speed <= self.MAX_SPEED * 1.05:
            sb = min(int(speed / self.MAX_SPEED * 4), 3)
        elif speed <= self.MAX_SPEED * 2.5:
            sb = 4
        else:
            sb = 5

        # 4) Which way does the road bend up ahead?
        look = self.LOOKAHEAD + self.LOOK_TIME * speed
        ahead = self.track.road_ahead(path, self.y, look)
        bend = ahead.angle(self.y + look) - path.angle(self.y)
        bb = 0 if bend < -0.1 else 2 if bend > 0.1 else 1

        # 5) Which slice of the track am I in? Lets the agent learn "do X HERE"
        slice_len = self.TRACK_LENGTH / self.track.SECTION_BINS
        section = max(0, min(self.track.SECTION_BINS - 1, int(self.y / slice_len)))

        # 6) What is the nearest thing coming up (pad / wall / ramp / box), or am I flying?
        # 7) Extras: slippery (ice) or ghost, and is the nitro bar at least half full?
        status = 1 if ("ice" in terrain or self.ice_timer > 0) else 2 if self.ghost_timer > 0 else 0
        extras = status * 2 + (1 if self.nitro >= 0.5 else 0)
        return (o, hb, sb, bb, section, self._ahead_hint(), extras)

    def _ahead_hint(self):
        """Describe the NEAREST object ahead with one number (see the table at the top)."""
        if self.airborne:
            return 9
        best_dy, best_code = None, 0

        def side(dx):                                 # 0 = to my left, 1 = straight ahead, 2 = to my right
            return 0 if dx < -25 else 2 if dx > 25 else 1

        for i, pad in enumerate(self.track.pads):
            dy = pad.y - self.y
            if i not in self.pads_used and 0 < dy <= self.SIGHT and abs(pad.x - self.x) <= self.PAD_SIDE:
                if best_dy is None or dy < best_dy:
                    best_dy, best_code = dy, 1 + side(pad.x - self.x)
        for i, ramp in enumerate(self.track.ramps):
            dy = ramp.y - self.y
            if i not in self.ramps_used and 0 < dy <= self.SIGHT and abs(ramp.x - self.x) <= self.PAD_SIDE:
                if best_dy is None or dy < best_dy:
                    best_dy, best_code = dy, 6 + side(ramp.x - self.x)
        for i, box in enumerate(self.track.boxes):
            dy = box.y - self.y
            if i not in self.boxes_used and 0 < dy <= self.SIGHT and abs(box.x - self.x) <= self.PAD_SIDE:
                if best_dy is None or dy < best_dy:
                    best_dy, best_code = dy, 10 + side(box.x - self.x)
        for i, coin in enumerate(self.track.coins):
            dy = coin.y - self.y
            if i not in self.coins_used and 0 < dy <= self.SIGHT and abs(coin.x - self.x) <= self.PAD_SIDE:
                if best_dy is None or dy < best_dy:
                    best_dy, best_code = dy, 14 + side(coin.x - self.x)
        for wall in self.track.walls:
            dy = wall.y - self.y
            if wall.gate or not (0 < dy <= self.SIGHT):
                continue                              # gate walls are handled by their ramp
            wx = wall.x_at(self.steps)
            if abs(wx - self.x) <= wall.half_width + 25 and (best_dy is None or dy < best_dy):
                best_dy = dy
                best_code = 4 if self.x < wx else 5   # wall is in my way: escape left (4) or right (5)
        return best_code

    # ------------------------------------------------------------ teleporting
    def _teleport(self, new_y, forward):
        """Move the car to distance new_y, onto the nearest road (never into a wall or the gap).
        Returns the distance moved."""
        old = self.y
        new_y = max(5.0, min(float(self.TRACK_LENGTH), new_y))
        for g0, g1 in self.track.gaps:
            if g0 <= new_y <= g1:
                new_y = g1 + 10 if forward else g0 - 10
        for wall in self.track.walls:
            if wall.y - 10 <= new_y <= wall.y + wall.thick + 10:
                new_y = wall.y + wall.thick + 20 if forward else wall.y - 20
        path, _ = self.track.nearest(self.x, new_y)
        self.y = new_y
        self.x = path.center_x(new_y)
        self.heading = path.angle(new_y)
        return new_y - old

    def _open_box(self):
        """Pick a random mystery effect and apply it. Good effects are twice as likely."""
        effect = random.choice(self.GOOD_EFFECTS + self.GOOD_EFFECTS + self.BAD_EFFECTS)
        self.last_box = effect
        self.boxes_opened += 1
        if effect == "BOOST":
            self.speed = max(self.speed, self.MAX_SPEED * 2)
            self.boost_mult, self.boost_timer = 2, self.BOOST_STEPS
        elif effect == "GHOST":
            self.ghost_timer = self.GHOST_STEPS
        elif effect == "FULL NITRO":
            self.nitro = 1.0
        elif effect == "ICE":
            self.ice_timer = self.BOX_ICE_STEPS
        elif effect == "SLOW":
            self.slow_timer = self.SLOW_STEPS
        elif effect == "NITRO DRAIN":
            self.nitro = max(0.0, self.nitro - self.NITRO_STEAL)
        elif effect == "WARP FORWARD":
            self._teleport(self.y + self.WARP_FORWARD, True)
        elif effect == "WARP BACK":
            self._teleport(self.y - self.WARP_BACK, False)

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply one action for one tick. Returns (next_state, reward, done, info)."""
        old_y = self.y
        was_airborne = self.airborne
        terrain = self._terrain_here()

        # Mud: a small chance each second of getting stuck for 2 seconds
        if ("mud" in terrain and self.stuck_timer == 0 and not was_airborne
                and random.random() < self.MUD_STUCK_CHANCE_PER_SEC * self.DT):
            self.stuck_timer = self.STUCK_STEPS
            self.boost_timer = 0
        stuck = self.stuck_timer > 0
        if stuck:
            self.stuck_timer -= 1
            self.speed = 0.0
        if was_airborne:
            self.air_timer -= 1

        ghost = self.ghost_timer > 0                       # (checked before the timers tick down)
        grip = 1.0
        if "ice" in terrain:
            grip = min(grip, self.ICE_GRIP)          # track ice: half steering
        if self.ice_timer > 0:
            grip = min(grip, self.BOX_ICE_GRIP)      # box ice: a short, milder slide
        slow_mult = 1.0
        if "mud" in terrain:
            slow_mult = min(slow_mult, self.MUD_SPEED)
        if self.slow_timer > 0:
            slow_mult = min(slow_mult, self.SLOW_SPEED)
        climb = self.MOUNTAIN_FACTOR if "mountain" in terrain else 1.0
        self.ice_timer = max(0, self.ice_timer - 1)
        self.slow_timer = max(0, self.slow_timer - 1)
        self.ghost_timer = max(0, self.ghost_timer - 1)

        eff_action = None if (was_airborne or stuck) else action   # no control in the air or when stuck

        # Throttle / brake / coast
        if eff_action in (self.ACCEL_STRAIGHT, self.ACCEL_LEFT, self.ACCEL_RIGHT) or eff_action in self.NITRO_ACTIONS:
            if self.speed < self.MAX_SPEED:
                self.speed = min(self.MAX_SPEED, self.speed + self.ACCEL * self.DT)
        elif eff_action == self.BRAKE_ACT:
            self.speed -= self.BRAKE * self.DT
        elif eff_action is not None:
            self.speed -= self.DRAG * self.DT

        # Nitro: burn it for a push toward 2x speed; otherwise it slowly recharges
        # Traction control: if the car is badly misaligned with the road, nitro cuts out (the action then
        # acts like a plain accelerate, and the bar does not drain). It stops runaway nitro slides.
        misaligned = False
        if eff_action in self.NITRO_ACTIONS and self.nitro > 0:
            near, _ = self.track.nearest(self.x, self.y)
            misaligned = abs(self.heading - near.angle(self.y)) > self.NITRO_CUT_HEADING
        self.nitro_cut = misaligned
        self.nitro_burning = eff_action in self.NITRO_ACTIONS and self.nitro > 0 and not misaligned
        if self.nitro_burning:
            self.nitro = max(0.0, self.nitro - self.NITRO_DRAIN)
            self.speed = min(self.MAX_SPEED * self.NITRO_MULT, self.speed + self.NITRO_ACCEL * self.DT)
        else:
            self.nitro = min(1.0, self.nitro + self.NITRO_CHARGE)

        # Boost pads: hold the boosted speed while the timer runs; afterwards ease back to normal
        if self.boost_timer > 0:
            self.boost_timer -= 1
            if eff_action != self.BRAKE_ACT:
                self.speed = max(self.speed, self.MAX_SPEED * self.boost_mult)
        elif self.speed > self.MAX_SPEED and eff_action != self.BRAKE_ACT and not self.nitro_burning:
            self.speed = max(self.MAX_SPEED, self.speed - self.BOOST_DECAY * self.DT)
        self.speed = max(0.0, self.speed)
        real_speed = self.speed * slow_mult * (self.GHOST_SPEED if ghost else 1.0)

        # Steering (a stopped car can't turn; no steering in the air; ice halves it)
        if eff_action is not None:
            steer = (-1 if eff_action in (self.ACCEL_LEFT, self.NITRO_LEFT)
                     else 1 if eff_action in (self.ACCEL_RIGHT, self.NITRO_RIGHT) else 0)
            self.heading += steer * self.TURN_RATE * min(real_speed / 100, 1.0) * grip * self.DT

        # Move the car along its heading (a mountain climb makes the same speed cover less track)
        self.x += real_speed / climb * math.sin(self.heading) * self.DT
        self.y += real_speed / climb * math.cos(self.heading) * self.DT
        self.steps += 1
        self.portal_flash = max(0, self.portal_flash - 1)

        landed = was_airborne and self.air_timer == 0
        coins_now = 0
        rail_cost = 0.0
        crash = None

        # Coins can be picked up on the ground or in the air (a jump should not skip one for free).
        for i, coin in enumerate(self.track.coins):
            if i in self.coins_used:
                continue
            if (old_y <= coin.y + coin.length and self.y >= coin.y
                    and abs(self.x - coin.x) <= coin.half_width + self.CAR_HALF_WIDTH):
                self.coins_used.add(i)
                self.coins_collected += 1
                coins_now += 1

        if not self.airborne:                       # ---- everything below only happens on the ground ----
            # Boost pads (always visible): touch one to jump to 2x or 3x speed
            for i, pad in enumerate(self.track.pads):
                if i in self.pads_used:
                    continue
                if (old_y <= pad.y + pad.length and self.y >= pad.y
                        and abs(self.x - pad.x) <= pad.half_width + self.CAR_HALF_WIDTH):
                    self.pads_used.add(i)
                    self.boosts_hit += 1
                    if self.MAX_SPEED * pad.mult > self.speed:       # a pad never slows you down
                        self.speed = self.MAX_SPEED * pad.mult
                        self.boost_mult = pad.mult
                        self.boost_timer = pad.steps

            # Jump ramps: drive onto one and the car takes off
            if not landed:
                for i, ramp in enumerate(self.track.ramps):
                    if i in self.ramps_used:
                        continue
                    if (old_y <= ramp.y + ramp.length and self.y >= ramp.y
                            and abs(self.x - ramp.x) <= ramp.half_width + self.CAR_HALF_WIDTH):
                        self.ramps_used.add(i)
                        self.jumps += 1
                        self.air_timer = self.AIR_STEPS
                        # the ramp guides the car: it leaves pointing along the road (no steering in the air)
                        self.heading = ramp.path.angle(self.y)
                        break

        if not self.airborne:
            # Walls: stop and slow the car, like a guard rail. A ghost passes through.
            # The car is pushed back in front of the wall and nudged toward the open side.
            if not ghost:
                for wall in self.track.walls:
                    wx = wall.x_at(self.steps)
                    if abs(self.x - wx) > wall.half_width + self.CAR_HALF_WIDTH:
                        continue
                    if landed or stuck:
                        hit = wall.y - 5 <= self.y <= wall.y + wall.thick + 5
                    else:
                        hit = old_y <= wall.y + wall.thick + 5 and self.y >= wall.y - 5
                    if not hit:
                        continue
                    if wall.gate:                                # full-width: nothing to steer around
                        crash = "wall"
                        break
                    self.y = wall.y - self.CAR_HALF_WIDTH - 1
                    side = 1 if self.x >= wx else -1
                    self.x += side * 6
                    cx = wall.path.center_x(self.y)
                    limit = wall.path.width / 2 - self.CAR_HALF_WIDTH - 0.5
                    self.x = max(cx - limit, min(cx + limit, self.x))
                    self.heading -= 0.6 * (self.heading - wall.path.angle(self.y))
                    self.speed *= self.RAIL_SLOWDOWN
                    self.rail_hits += 1
                    rail_cost += self.RAIL_PENALTY
                    break

            # Guard rails: scrape and bounce back instead of crashing
            if crash is None:
                for rail in self.track.rails:
                    if not (rail.y0 <= self.y <= rail.y1):
                        continue
                    p = rail.path
                    rel = self.x - p.center_x(self.y)
                    limit = p.width / 2 - self.CAR_HALF_WIDTH - 0.5      # (0.5 px safety margin)
                    if abs(rel) > p.width / 2 + 30:                      # not on this road
                        continue
                    if (rail.right and rel > limit) or (rail.left and rel < -limit):
                        self.x = p.center_x(self.y) + (limit if rel > 0 else -limit)
                        self.heading -= 0.6 * (self.heading - p.angle(self.y))   # pushed back along the road
                        self.speed *= self.RAIL_SLOWDOWN
                        self.rail_hits += 1
                        rail_cost += self.RAIL_PENALTY

            # Mystery boxes: random good or bad effect. Checked before a portal jump so the
            # jump does not sweep up every box along the 1000 px it skips.
            if crash is None:
                for i, box in enumerate(self.track.boxes):
                    if i in self.boxes_used:
                        continue
                    if (old_y <= box.y + box.length and self.y >= box.y
                            and abs(self.x - box.x) <= box.half_width + self.CAR_HALF_WIDTH):
                        self.boxes_used.add(i)
                        self._open_box()
                        break

            # Hidden portals: jump PORTAL_DISTANCE px ahead, once per episode. No freeze.
            if crash is None and self.portals_enabled and not self.portal_used:
                for portal in self.track.portals:
                    if not (old_y <= portal.y1 and self.y >= portal.y0 and self.y <= portal.path.y1):
                        continue
                    rel = self.x - portal.path.center_x(self.y)
                    if portal.off0 <= rel <= portal.off1:
                        self.portal_used = True
                        self.portal_flash = self.PORTAL_FLASH_STEPS
                        self._teleport(self.y + self.PORTAL_DISTANCE, True)
                        break

        # ---- Remember which fork path was taken ----
        if self.route is None and self.track.ROUTE_Y <= self.y <= 5000:
            path, _ = self.track.nearest(self.x, self.y)
            if path.name in self.track.fork_names:
                self.route = path.name

        # ---- Crash / finish / timeout ----
        done = False
        if crash is None and not self.airborne and not self.track.on_road(self.x, self.y, self.CAR_HALF_WIDTH):
            crash = "landing" if landed else "road"                  # left the road (or landed off it)
        if landed and crash is None:
            self.landings += 1
        if crash is not None:
            self.crashed = True
            self.crash_reason = crash
            done = True
        elif self.y >= self.TRACK_LENGTH:                            # crossed finish line
            self.finished = True
            done = True
        elif self.steps >= self.MAX_STEPS:                           # ran out of time
            done = True

        # Time points exist only for a finish: one point per step you did not use.
        # The learning reward is that score change, plus a small forward-progress signal
        # and the rail penalty. Progress and rails are not added into self.score.
        self.time_points = self.TIME_POINT * (self.MAX_STEPS - self.steps) if self.finished else 0
        self.score = int(self.coins_collected * self.COIN_VALUE + self.time_points)
        if self.crashed:
            self.score -= self.CRASH_PENALTY
        reward = (self.y - old_y) * self.PROGRESS_SCALE - rail_cost - self.STEP_COST + coins_now * self.COIN_VALUE
        if self.finished:
            reward += self.time_points
        elif self.crashed:
            reward -= self.CRASH_PENALTY

        info = {"time": self.steps * self.DT, "score": self.score,
                "finished": self.finished, "crashed": self.crashed, "crash_reason": self.crash_reason,
                "portal": self.portal_used, "route": self.route, "boosts": self.boosts_hit,
                "jumps": self.jumps, "landings": self.landings, "rail_hits": self.rail_hits,
                "boxes": self.boxes_opened, "from_start": self.from_start}
        return self.get_state(), reward, done, info