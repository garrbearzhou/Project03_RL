"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
tests.py - automatic checks for the game's rules and the learning code.

Run them with:      python tests.py          (or: python -m unittest tests -v)

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

Why tests? The game has lots of rules (walls, rails, jumps, portals, nitro, boxes...) and every time one is
changed it is easy to break another by accident. These tests describe how each rule is SUPPOSED to work.
"""
import os
import pickle
import random
import tempfile
import unittest
from unittest import mock

from agent import QLearningAgent
from environment import RacingEnv
from scoring import max_possible_score


# ----------------------------------------------------------------------------- helpers
def place(env, y, x_offset=0.0, speed=300.0, heading_offset=0.0, path_name=None, **attrs):
    """Put a fresh car on the road at distance y (centered, pointing along the road)."""
    env.reset()
    roads = env.track.active(y)
    road = [p for p in roads if p.name == path_name][0] if path_name else roads[0]
    env.y = float(y)
    env.x = road.center_x(y) + x_offset
    env.heading = road.angle(y) + heading_offset
    env.speed = speed
    for name, value in attrs.items():
        setattr(env, name, value)
    return road


def clear_track(env, keep=()):
    """Remove every obstacle/pickup from the track except the kinds listed in `keep`, for testing one rule alone."""
    t = env.track
    for kind, ys, length in (("coins", "coin_ys", "coin_len"), ("pads", "pad_ys", "pad_len"),
                             ("ramps", "ramp_ys", "ramp_len"), ("boxes", "box_ys", "box_len"),
                             ("walls", "wall_ys", None)):
        if kind in keep:
            continue
        setattr(t, kind, [])
        setattr(t, ys, [])
        if length:
            setattr(t, length, 0)
    for kind in ("rails", "terrain", "portals"):
        if kind not in keep:
            setattr(t, kind, [])


def run(env, action, steps):
    info = None
    for _ in range(steps):
        _, reward, done, info = env.step(action)
        if done:
            break
    return info


# ----------------------------------------------------------------------------- the track
class TrackTests(unittest.TestCase):
    def setUp(self):
        self.track = RacingEnv().track

    def test_roads_join_up(self):
        """Each road starts where another ends (or across the gap), at the same sideways position."""
        ends = {p.y1: p.center_x(p.y1) for p in self.track.paths}
        for p in self.track.paths:
            if p.y0 == 0:
                continue
            start_x = p.center_x(p.y0)
            if p.y0 in ends:
                self.assertAlmostEqual(ends[p.y0], start_x, places=6, msg=f"{p.name} at {p.y0}")
            else:                                         # must be the far side of a gap
                gap = [g for g in self.track.gaps if g[1] == p.y0][0]
                self.assertAlmostEqual(ends[gap[0]], start_x, places=6)

    def test_pickups_sit_on_their_road_and_not_in_the_gap(self):
        t = self.track
        for kind in (t.coins, t.pads, t.ramps, t.boxes):
            for o in kind:
                self.assertLessEqual(abs(o.x - o.path.center_x(o.y)) + o.half_width, o.path.width / 2, f"{type(o).__name__} at {o.y}")
                self.assertFalse(t.in_gap(o.y))

    def test_lists_are_sorted_and_lookup_arrays_match(self):
        t = self.track
        for objs, ys in ((t.coins, t.coin_ys), (t.pads, t.pad_ys), (t.ramps, t.ramp_ys),
                         (t.boxes, t.box_ys), (t.walls, t.wall_ys)):
            self.assertEqual([o.y for o in objs], ys)
            self.assertEqual(ys, sorted(ys))

    def test_portals_are_reachable(self):
        """A portal must be inside the strip the car's center can actually reach (road half-width minus car half-width)."""
        for portal in self.track.portals:
            reach = portal.path.width / 2 - RacingEnv.CAR_HALF_WIDTH
            self.assertLessEqual(abs(portal.off0), reach)
            self.assertLessEqual(abs(portal.off1), reach)

    def test_curvature_table_is_zero_on_straights_and_positive_in_curves(self):
        trunk2a = [p for p in self.track.paths if p.name == "Main road" and p.y0 == 5700][0]
        self.assertEqual(trunk2a.curvature_ahead(8300), 0.0)             # the straight before the gap
        self.assertGreater(trunk2a.curvature_ahead(6700), 0.004)         # the sharp second bend is in the next 500 px
        self.assertLess(trunk2a.curvature_ahead(6000), trunk2a.curvature_ahead(6700))   # ...and not yet at 6000

    def test_a_forced_route_removes_the_other_fork_path_and_its_objects(self):
        for keep, drop in (("Scenic", "Shortcut"), ("Shortcut", "Scenic")):
            t = RacingEnv(route=keep).track
            self.assertNotIn(drop, [p.name for p in t.paths])
            for kind in (t.coins, t.pads, t.ramps, t.boxes, t.walls, t.rails, t.terrain, t.portals):
                self.assertTrue(all(o.path.name != drop for o in kind))
            self.assertEqual([o.y for o in t.coins], t.coin_ys)
        # the whole track is still drivable from start to finish
        env = RacingEnv(route="Scenic")
        road, _ = env.track.nearest(0, 5000)
        self.assertTrue(env.track.on_road(road.center_x(5000), 5000, 10))

    def test_gap_has_no_road(self):
        g0, g1 = self.track.gaps[0]
        self.assertEqual(self.track.active((g0 + g1) / 2), [])
        self.assertFalse(self.track.on_road(0, (g0 + g1) / 2, 10))


# ----------------------------------------------------------------------------- the environment
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.env = RacingEnv()

    def test_state_is_a_valid_tuple(self):
        env = self.env
        bounds = (env.OFFSET_BINS, env.HEADING_BINS, 6, 5, env.track.SECTION_BINS, 17, 6)
        for ep in range(40):
            state = env.reset(random_start=ep % 2 == 1)
            done = False
            while not done:
                self.assertEqual(len(state), 7)
                for value, limit in zip(state, bounds):
                    self.assertTrue(0 <= value < limit, f"{state}")
                state, _, done, _ = env.step(random.randrange(env.N_ACTIONS))

    def test_every_reset_starts_on_the_road(self):
        for i in range(300):
            self.env.reset(random_start=i % 2 == 0)
            self.assertTrue(self.env.track.on_road(self.env.x, self.env.y, self.env.CAR_HALF_WIDTH))

    def test_same_seed_gives_same_run(self):
        def trajectory():
            random.seed(77)
            env = RacingEnv()
            out = []
            for _ in range(5):
                env.reset(random_start=True)
                for _ in range(80):
                    state, reward, done, _ = env.step(random.randrange(8))
                    out.append((state, round(reward, 6)))
                    if done:
                        break
            return out
        self.assertEqual(trajectory(), trajectory())

    # ---- pickups
    def test_a_coin_is_collected_once_and_pays_its_value(self):
        env = self.env
        clear_track(env)
        from track import Coin
        road = env.track.paths[0]
        env.track.coins = [Coin(road, 300)]             # exactly one coin on the track
        env.track.coin_ys, env.track.coin_len = [300], 40
        place(env, 285, speed=300.0)
        _, reward, _, _ = env.step(env.ACCEL_STRAIGHT)
        _, reward2, _, _ = env.step(env.ACCEL_STRAIGHT)
        self.assertEqual(env.coins_collected, 1)
        self.assertGreater(reward + reward2, env.COIN_VALUE - 1)
        run(env, env.ACCEL_STRAIGHT, 10)
        self.assertEqual(env.coins_collected, 1)

    def test_boost_pad_gives_2x_speed_then_eases_back(self):
        env = self.env
        clear_track(env, keep=("pads",))
        pad = [p for p in env.track.pads if p.mult == 2][0]
        place(env, pad.y - 15, speed=300.0, path_name=pad.path.name)
        run(env, env.ACCEL_STRAIGHT, 2)
        self.assertGreaterEqual(env.speed, 600 - 1)
        self.assertEqual(env.boosts_hit, 1)
        env.air_timer = 0
        for _ in range(pad.steps + 60):                  # long after the boost ends
            if env.step(env.ACCEL_STRAIGHT)[2]:
                break
        self.assertLess(env.speed, 301 + 1)

    def test_nitro_charges_in_ten_seconds(self):
        env = self.env
        env.reset()
        env.nitro, env.speed = 0.0, 0.0
        run(env, env.COAST, 300)
        self.assertAlmostEqual(env.nitro, 1.0, delta=0.01)

    def test_nitro_burns_down_and_pushes_toward_2x_speed(self):
        env = self.env
        clear_track(env)
        place(env, 5710, speed=300.0, nitro=1.0)         # a long straight, nothing in the way
        for _ in range(30):                              # 1 s of burning (about 400 px, so the road ahead stays straight)
            env.step(env.NITRO)
        self.assertAlmostEqual(env.nitro, 1.0 - 30 / 90, delta=0.03)
        self.assertGreater(env.speed, 550)

    def test_state_warns_when_going_too_fast_for_the_curve_ahead(self):
        env = self.env
        clear_track(env)
        place(env, 6700, speed=300.0)                      # the sharp bend is ahead; 300 px/s is fine
        self.assertLess(env.get_state()[3], 3)
        place(env, 6700, speed=600.0)                      # at 600 px/s the car could not follow it
        self.assertIn(env.get_state()[3], (3, 4))

    def test_being_too_fast_for_the_curve_costs_learning_reward_but_not_score(self):
        env = self.env
        clear_track(env)
        rewards = []
        for penalty in (0.0, env.DANGER_PENALTY):
            env.DANGER_PENALTY = penalty
            place(env, 6700, speed=600.0, boost_mult=2, boost_timer=40)       # held at 600 px/s
            _, reward, _, _ = env.step(env.ACCEL_STRAIGHT)
            rewards.append(reward)
            self.assertEqual(env.score, 0)                                    # (the score never changes)
        self.assertGreater(env._danger_excess, 0.4)
        self.assertLess(rewards[1], rewards[0] - 0.3)
        place(env, 6700, speed=300.0)
        env.step(env.ACCEL_STRAIGHT)
        self.assertEqual(env._danger_excess, 0.0)                             # no penalty at a safe speed

    def test_nitro_locks_out_when_the_road_ahead_is_too_sharp_for_the_speed(self):
        env = self.env
        clear_track(env)
        place(env, 6700, speed=300.0, nitro=1.0)
        env.step(env.NITRO)
        self.assertTrue(env.nitro_burning)                 # fine at 300 px/s
        place(env, 6700, speed=420.0, nitro=1.0)
        env.step(env.NITRO)
        self.assertTrue(env.nitro_cut)
        self.assertFalse(env.nitro_burning)

    def test_nitro_cuts_out_when_the_car_is_badly_misaligned(self):
        env = self.env
        clear_track(env)
        place(env, 5710, speed=300.0, heading_offset=0.5, nitro=1.0)
        env.step(env.NITRO)
        self.assertTrue(env.nitro_cut)
        self.assertFalse(env.nitro_burning)

    def test_mystery_box_effects(self):
        env = self.env
        env.reset()
        expectations = {
            "JACKPOT": lambda: self.assertEqual(env.box_points, env.BOX_POINTS),
            "BOOST": lambda: self.assertGreaterEqual(env.speed, 600),
            "GHOST": lambda: self.assertGreater(env.ghost_timer, 0),
            "FULL NITRO": lambda: self.assertEqual(env.nitro, 1.0),
            "ICE": lambda: self.assertGreater(env.ice_timer, 0),
            "SLOW": lambda: self.assertGreater(env.slow_timer, 0),
            "NITRO DRAIN": lambda: self.assertAlmostEqual(env.nitro, 0.5 - env.NITRO_STEAL, places=6),
            "WARP FORWARD": lambda: self.assertGreater(env.y, 1000),
            "WARP BACK": lambda: self.assertLess(env.y, 1000),
        }
        self.assertEqual(set(expectations), set(env.GOOD_EFFECTS + env.BAD_EFFECTS + [env.JACKPOT]))
        for effect, check in expectations.items():
            place(env, 1000, nitro=0.5)
            env.box_prizes[0] = effect                       # box 0 was dealt this prize for this run
            env._open_box(0)
            check()

    def test_every_run_deals_every_box_a_fresh_random_prize(self):
        """A box must not keep the same prize from run to run. Check: prizes match the pool's odds, and
        a box repeats its previous prize only about as often as pure chance (about 12%), not always."""
        env = self.env
        runs, n_boxes = 2500, len(env.track.boxes)
        self.assertGreater(n_boxes, 2)
        history = []
        for _ in range(runs):
            env.reset()
            self.assertEqual(len(env.box_prizes), n_boxes)
            history.append(list(env.box_prizes))
        pool = env.PRIZE_POOL
        for box in range(n_boxes):
            seen = [h[box] for h in history]
            for prize in set(pool):                                       # each prize appears at its expected rate
                expected = pool.count(prize) / len(pool)
                self.assertAlmostEqual(seen.count(prize) / runs, expected, delta=0.035, msg=f"box {box} {prize}")
            repeats = sum(a == b for a, b in zip(seen, seen[1:])) / (runs - 1)
            chance = sum((pool.count(p) / len(pool)) ** 2 for p in set(pool))
            self.assertAlmostEqual(repeats, chance, delta=0.035, msg=f"box {box} repeats")
        # different boxes in the same run are dealt independently (they are not all the same prize)
        self.assertGreater(sum(len(set(h)) > 1 for h in history) / runs, 0.99)

    def test_the_jackpot_is_one_slot_in_thirteen_and_good_prizes_outnumber_bad(self):
        env = self.env
        self.assertEqual(len(env.PRIZE_POOL), 13)
        self.assertEqual(env.PRIZE_POOL.count(env.JACKPOT), 1)
        good = sum(p in env.GOOD_EFFECTS or p == env.JACKPOT for p in env.PRIZE_POOL)
        bad = sum(p in env.BAD_EFFECTS for p in env.PRIZE_POOL)
        self.assertEqual((good, bad), (9, 4))
        self.assertEqual(env.BOX_POINTS, 50 * env.COIN_VALUE)                # "50 coins worth of points"

    def test_a_jackpot_box_adds_points_to_the_score_and_the_reward(self):
        env = self.env
        clear_track(env, keep=("boxes",))
        box = env.track.boxes[0]
        road = place(env, box.y - 15, x_offset=box.x - box.path.center_x(box.y - 15), path_name=box.path.name)
        env.box_prizes[0] = env.JACKPOT
        rewards = [env.step(env.ACCEL_STRAIGHT)[1] for _ in range(3)]
        self.assertEqual(env.box_points, env.BOX_POINTS)
        self.assertEqual(env.score, env.BOX_POINTS)                           # (no coins, no finish yet)
        self.assertGreater(max(rewards), env.BOX_POINTS - 5)                  # the reward carries the points
        self.assertEqual(env.last_box, env.JACKPOT)
        run(env, env.ACCEL_STRAIGHT, 10)
        self.assertEqual(env.box_points, env.BOX_POINTS)                      # each box pays once per run

    def test_box_points_are_part_of_the_final_score(self):
        env = self.env
        clear_track(env)
        place(env, 11995, speed=300.0)
        env.steps, env.box_points = 1000, env.BOX_POINTS
        _, _, done, info = env.step(env.ACCEL_STRAIGHT)
        self.assertEqual(info["score"], env.BOX_POINTS + env.TIME_POINT * (env.MAX_STEPS - 1001))

    # ---- the best possible score
    def test_max_possible_score_is_built_from_the_rules(self):
        env = self.env
        top = max_possible_score(env)
        self.assertEqual(top["score"], top["coin_points"] + top["time_points"])
        self.assertEqual(top["score_with_luck"], top["score"] + top["boxes"] * env.BOX_POINTS)
        self.assertGreater(top["score_with_luck"], top["score"])
        routes = {r: len([c for c in env.track.coins if c.path.name in (r, "Main road")]) for r in env.track.fork_names}
        self.assertEqual(top["coins"], max(routes.values()))                  # all the coins on the best route
        self.assertGreater(max_possible_score(env)["score"], max_possible_score(RacingEnv(portals=False))["score"])

    def test_no_real_run_can_beat_the_max_possible_score(self):
        """Skill-only runs (no boxes on the track) must stay at or under the skill-only maximum."""
        import settings
        random.seed(8)
        env = RacingEnv()
        env.track.boxes, env.track.box_ys, env.track.box_len = [], [], 0
        ceiling = max_possible_score(env)["score"]
        agent = QLearningAgent(env.N_ACTIONS, **settings.AGENT_KWARGS)
        if os.path.exists(settings.Q_FILE):                                   # the trained agent, if its table is here
            agent.load(settings.Q_FILE)
        agent.greedy = os.path.exists(settings.Q_FILE)
        agent.epsilon = 0.3
        best = -10 ** 9
        for ep in range(60):
            state, done = env.reset(), False
            while not done:
                state, _, done, info = env.step(agent.choose_action(state))
            best = max(best, info["score"])
        self.assertLessEqual(best, ceiling)

    # ---- walls, rails, road edges
    def test_a_wall_slows_the_car_but_does_not_end_the_run(self):
        env = self.env
        clear_track(env, keep=("walls",))
        wall = env.track.walls[0]                         # the fixed wall in the middle of the first straight
        place(env, wall.y - 40, x_offset=wall.x_at(0) - env.track.paths[0].center_x(wall.y - 40))
        run(env, env.ACCEL_STRAIGHT, 12)
        self.assertGreaterEqual(env.rail_hits, 1)
        self.assertFalse(env.crashed)

    def test_a_ghost_passes_through_walls(self):
        env = self.env
        clear_track(env, keep=("walls",))
        wall = env.track.walls[0]
        place(env, wall.y - 40, x_offset=wall.x_at(0) - env.track.paths[0].center_x(wall.y - 40), ghost_timer=200)
        run(env, env.ACCEL_STRAIGHT, 12)
        self.assertEqual(env.rail_hits, 0)
        self.assertGreater(env.y, wall.y + wall.thick)

    def test_the_gate_wall_ends_the_run_unless_you_jump(self):
        env = self.env
        clear_track(env, keep=("walls",))                 # no ramp: the gate cannot be jumped
        gate = [w for w in env.track.walls if w.gate][0]
        place(env, gate.y - 60, speed=300.0)
        info = run(env, env.ACCEL_STRAIGHT, 30)
        self.assertEqual(info["crash_reason"], "wall")

    def test_a_ramp_lets_you_fly_over_the_gate_wall(self):
        env = self.env
        clear_track(env, keep=("walls", "ramps"))
        ramp = [r for r in env.track.ramps if r.y < 6100 and r.y > 6000][0]
        gate = [w for w in env.track.walls if w.gate][0]
        place(env, ramp.y - 30, speed=300.0)
        info = run(env, env.ACCEL_STRAIGHT, 40)
        self.assertFalse(info["crashed"])
        self.assertEqual(env.jumps, 1)
        self.assertGreater(env.y, gate.y + gate.thick)

    def test_guard_rail_pushes_the_car_back_instead_of_crashing(self):
        env = self.env
        clear_track(env, keep=("rails",))
        rail = env.track.rails[0]
        y = (rail.y0 + rail.y1) / 2
        road = place(env, y, x_offset=88, heading_offset=0.3, speed=300.0)
        env.step(env.ACCEL_STRAIGHT)
        self.assertEqual(env.rail_hits, 1)
        self.assertFalse(env.crashed)
        self.assertLessEqual(abs(env.x - road.center_x(env.y)), road.width / 2 - env.CAR_HALF_WIDTH)

    def test_driving_off_the_road_is_a_crash(self):
        env = self.env
        clear_track(env)
        place(env, 2000, x_offset=95, heading_offset=0.3)
        info = run(env, env.ACCEL_STRAIGHT, 5)
        self.assertEqual(info["crash_reason"], "road")

    # ---- jumps
    def test_a_slow_jump_across_the_gap_falls_in(self):
        env = self.env
        clear_track(env, keep=("ramps",))
        ramp = [r for r in env.track.ramps if r.y > 8000][0]
        place(env, ramp.y - 20, speed=300.0)
        info = run(env, env.ACCEL_STRAIGHT, 60)
        self.assertEqual(info["crash_reason"], "landing")

    def test_a_boosted_jump_clears_the_gap(self):
        env = self.env
        clear_track(env, keep=("ramps",))
        ramp = [r for r in env.track.ramps if r.y > 8000][0]
        place(env, ramp.y - 20, speed=600.0, boost_mult=2, boost_timer=60)
        info = run(env, env.ACCEL_STRAIGHT, 40)
        self.assertFalse(info["crashed"], info)
        self.assertGreater(env.y, env.track.gaps[0][1])

    def test_a_ramp_points_the_car_along_the_road(self):
        """The ramp guides the car, so a slightly crooked approach still flies straight (there is no steering in the air)."""
        env = self.env
        clear_track(env, keep=("ramps",))
        ramp = env.track.ramps[0]
        road = place(env, ramp.y - 15, heading_offset=0.2, speed=300.0, path_name=ramp.path.name)
        run(env, env.ACCEL_STRAIGHT, 3)
        self.assertTrue(env.airborne)
        self.assertAlmostEqual(env.heading, road.angle(env.y), places=6)

    def test_no_steering_in_the_air(self):
        env = self.env
        clear_track(env)
        place(env, 5710, air_timer=10)
        heading = env.heading
        run(env, env.ACCEL_LEFT, 5)
        self.assertEqual(env.heading, heading)

    # ---- terrain
    def test_mud_slows_the_car(self):
        env = self.env
        clear_track(env, keep=("terrain",))
        env.MUD_STUCK_CHANCE_PER_SEC = 0.0
        mud = [z for z in env.track.terrain if z.kind == "mud"][0]
        mid = (mud.y0 + mud.y1) / 2
        place(env, mid, speed=300.0, path_name=mud.path.name)
        y0 = env.y
        env.step(env.ACCEL_STRAIGHT)
        mud_dy = env.y - y0
        place(env, 5710, speed=300.0)
        y0 = env.y
        env.step(env.ACCEL_STRAIGHT)
        self.assertAlmostEqual(mud_dy / (env.y - y0), env.MUD_SPEED, delta=0.05)

    def test_ice_halves_steering(self):
        env = self.env
        clear_track(env, keep=("terrain",))
        ice = [z for z in env.track.terrain if z.kind == "ice"][0]
        place(env, (ice.y0 + ice.y1) / 2, speed=300.0, path_name=ice.path.name)
        h0 = env.heading
        env.step(env.ACCEL_RIGHT)
        ice_turn = env.heading - h0
        place(env, 5710, speed=300.0)
        h0 = env.heading
        env.step(env.ACCEL_RIGHT)
        self.assertAlmostEqual(ice_turn / (env.heading - h0), env.ICE_GRIP, delta=0.05)

    def test_a_mountain_covers_less_track_per_step(self):
        env = self.env
        clear_track(env, keep=("terrain",))
        mountain = [z for z in env.track.terrain if z.kind == "mountain"][0]
        place(env, (mountain.y0 + mountain.y1) / 2, speed=300.0, path_name=mountain.path.name)
        y0 = env.y
        env.step(env.ACCEL_STRAIGHT)
        climb_dy = env.y - y0
        place(env, 5710, speed=300.0)
        y0 = env.y
        env.step(env.ACCEL_STRAIGHT)
        self.assertAlmostEqual(climb_dy / (env.y - y0), 1 / env.MOUNTAIN_FACTOR, delta=0.05)

    # ---- teleports and portals
    def test_warps_never_land_in_the_gap_or_a_wall(self):
        env = self.env
        place(env, 8200)
        env._teleport(8500, forward=True)
        self.assertFalse(env.track.in_gap(env.y))
        place(env, 8700)
        env._teleport(8500, forward=False)
        self.assertFalse(env.track.in_gap(env.y))
        for wall in env.track.walls:
            place(env, wall.y - 200)
            env._teleport(wall.y + 5, forward=True)
            self.assertFalse(wall.y - 10 <= env.y <= wall.y + wall.thick + 10)

    def test_a_portal_jumps_forward_once(self):
        env = self.env
        clear_track(env, keep=("portals",))
        portal = [p for p in env.track.portals if p.path.name == "Shortcut"][0]
        y = (portal.y0 + portal.y1) / 2
        place(env, y, x_offset=(portal.off0 + portal.off1) / 2, path_name="Shortcut", speed=300.0)
        env.step(env.ACCEL_STRAIGHT)
        self.assertTrue(env.portal_used)
        self.assertAlmostEqual(env.y, y + env.PORTAL_DISTANCE, delta=20)
        jumped_to = env.y
        env.y = y                                           # drive over it again: nothing happens
        env.x = portal.path.center_x(y) + (portal.off0 + portal.off1) / 2
        env.step(env.ACCEL_STRAIGHT)
        self.assertLess(env.y, jumped_to)

    def test_the_no_portal_version_has_no_working_portals(self):
        env = RacingEnv(portals=False)
        clear_track(env, keep=("portals",))
        portal = env.track.portals[0]
        y = (portal.y0 + portal.y1) / 2
        place(env, y, x_offset=(portal.off0 + portal.off1) / 2, path_name=portal.path.name, speed=300.0)
        env.step(env.ACCEL_STRAIGHT)
        self.assertFalse(env.portal_used)

    # ---- scoring
    def test_a_finish_scores_the_time_left_on_the_clock(self):
        env = self.env
        clear_track(env)
        place(env, 11995, speed=300.0)                 # 5 px from the finish line
        env.steps = 1000
        _, reward, done, info = env.step(env.ACCEL_STRAIGHT)
        self.assertTrue(done and info["finished"])
        self.assertEqual(info["score"], env.TIME_POINT * (env.MAX_STEPS - 1001))
        self.assertGreater(reward, info["score"])              # (reward also has the small progress signal)

    def test_a_crash_costs_the_crash_penalty(self):
        env = self.env
        clear_track(env)
        place(env, 2000, x_offset=95, heading_offset=0.3)
        _, reward, done, info = env.step(env.ACCEL_STRAIGHT)
        self.assertTrue(done and info["crashed"])
        self.assertEqual(info["score"], -env.CRASH_PENALTY)
        self.assertLessEqual(reward, -env.CRASH_PENALTY + 1)

    def test_running_out_of_time_ends_the_run_without_a_finish_bonus(self):
        env = self.env
        env.reset()
        env.steps = env.MAX_STEPS - 1
        env.speed = 0.0
        _, _, done, info = env.step(env.COAST)
        self.assertTrue(done)
        self.assertFalse(info["finished"] or info["crashed"])
        self.assertEqual(info["score"], 0)


# ----------------------------------------------------------------------------- the learning agent
class AgentTests(unittest.TestCase):
    def test_q_learning_update(self):
        agent = QLearningAgent(3, alpha=0.5, gamma=0.9)
        agent.q["a"] = [0.0, 0.0, 0.0]
        agent.q["b"] = [0.0, 2.0, 0.0]
        agent.learn("a", 0, 1.0, "b", False)                 # 0 + 0.5 * (1 + 0.9 * 2 - 0)
        self.assertAlmostEqual(agent.q["a"][0], 1.4)
        agent.learn("a", 1, 1.0, "b", True)                  # finished: the future is ignored
        self.assertAlmostEqual(agent.q["a"][1], 0.5)

    def test_greedy_agent_picks_the_best_action_and_does_not_learn(self):
        agent = QLearningAgent(3)
        agent.q["s"] = [0.0, 5.0, 1.0]
        agent.greedy = True
        self.assertEqual({agent.choose_action("s") for _ in range(50)}, {1})
        agent.learn("s", 0, 100.0, "s", True)
        self.assertEqual(agent.q["s"][0], 0.0)

    def test_minimum_epsilon_forces_extra_exploring_but_not_when_greedy(self):
        random.seed(4)
        agent = QLearningAgent(4, epsilon=0.0)
        agent.q["s"] = [0.0, 9.0, 0.0, 0.0]
        self.assertEqual({agent.choose_action("s") for _ in range(100)}, {1})               # no exploring
        self.assertGreater(len({agent.choose_action("s", min_epsilon=1.0) for _ in range(100)}), 1)
        agent.greedy = True
        self.assertEqual({agent.choose_action("s", min_epsilon=1.0) for _ in range(100)}, {1})

    def test_exploring_agent_tries_every_action(self):
        random.seed(3)
        agent = QLearningAgent(4, epsilon=1.0)
        self.assertEqual({agent.choose_action("s") for _ in range(200)}, {0, 1, 2, 3})

    def test_epsilon_decays_but_not_below_the_minimum(self):
        agent = QLearningAgent(2, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5)
        for _ in range(20):
            agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.05)

    def test_new_states_start_with_the_optimistic_values(self):
        agent = QLearningAgent(3, epsilon=0.0, initial_q=[0, 0, 8.0])     # (epsilon 0: no random exploring)
        agent.choose_action("new")
        self.assertEqual(agent.q["new"], [0, 0, 8.0])

    def test_save_load_round_trip_and_old_tables_are_upgraded(self):
        agent = QLearningAgent(3, initial_q=[0, 0, 8.0], epsilon=0.3)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "q.pkl")
            old = {"q": {"s": [1.0, 2.0]}, "epsilon": 0.2}      # a table saved by a version with 2 actions
            with open(path, "wb") as f:
                pickle.dump(old, f)
            agent.load(path)
            self.assertEqual(agent.q["s"], [1.0, 2.0, 8.0])
            self.assertEqual(agent.epsilon, 0.2)
            with open(path, "wb") as f:
                pickle.dump({"q": {"s": [0] * 5}, "epsilon": 0.1}, f)
            with self.assertRaises(ValueError):
                agent.load(path)


# ----------------------------------------------------------------------------- everything together
class IntegrationTests(unittest.TestCase):
    def test_a_short_training_run_fills_the_q_table_without_errors(self):
        random.seed(5)
        env = RacingEnv()
        agent = QLearningAgent(env.N_ACTIONS, epsilon=0.5, initial_q=[0, 0, 0, 0, 0, 8, 8, 8])
        for ep in range(25):
            state, done = env.reset(random_start=ep % 2 == 0), False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
        self.assertGreater(len(agent.q), 200)
        for row in agent.q.values():
            self.assertEqual(len(row), env.N_ACTIONS)


if __name__ == "__main__":
    unittest.main(verbosity=1)