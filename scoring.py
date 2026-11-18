"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
scoring.py - how many points is the BEST possible run worth? (shown in the game's HUD so you can compare)

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

Score = coins x COIN_VALUE + jackpot points + (steps left on the clock if you finish).
The best possible score is therefore "every coin on the best route" + "the fastest finish the rules allow".
The coins are easy to count. The fastest finish is the hard part, so this file bounds it from below with an
OPTIMISTIC model of the physics (an upper bound on the score: no real run can beat it):

  * the car cruises at MAX_SPEED the whole way (it ignores starting from 0, curves, mud, ice and mountains),
  * every boost pad on the route adds its full hold time PLUS the slow-down tail after it,
  * nitro is burned in the best possible bursts, and the bar is never wasted,
  * one portal jump (PORTAL_DISTANCE) is allowed if portals are on,
  * mystery boxes are ignored ("skill only"), OR every box on the route is a jackpot ("with luck").
    (A jackpot beats every other prize by far, so all-jackpots is the luckiest possible run.)

The real best runs score about 85% of the skill-only number, because the car must also steer.
"""
import math


def max_possible_score(env):
    """Return the best score a run could possibly get, as a dict (see the keys at the bottom).
    Uses the rules and constants of `env` (a RacingEnv), so it updates itself when you change the track or the rules."""
    E, track = type(env), env.track
    forks = track.fork_names

    def on_route(route, things):                 # everything on the shared roads plus this fork path
        return [o for o in things if o.path.name not in forks or o.path.name == route]

    # nitro: the most extra distance one second of burning can ever give (burst until 2x speed, then coast down)
    dv = E.MAX_SPEED * (E.NITRO_MULT - 1)                          # extra speed nitro can add
    burst = dv / E.NITRO_ACCEL                                     # seconds of burning to reach it
    extra_per_burn_second = (0.5 * E.NITRO_ACCEL * burst ** 2 + dv ** 2 / (2 * E.BOOST_DECAY)) / burst
    burn_per_sec = E.NITRO_DRAIN / E.DT                            # bar used per second of burning
    charge_per_sec = E.NITRO_CHARGE / E.DT                         # bar refilled per second

    best = None
    for route in forks:
        coins = len(on_route(route, track.coins))
        boxes = len(on_route(route, track.boxes))
        pad_extra = 0.0                                            # distance gained over cruising, per pad
        for pad in on_route(route, track.pads):
            dv_pad = E.MAX_SPEED * (pad.mult - 1)
            pad_extra += dv_pad * pad.steps * E.DT + dv_pad ** 2 / (2 * E.BOOST_DECAY)    # hold + slow-down tail
        portal = E.PORTAL_DISTANCE if env.portals_enabled else 0.0
        # distance = MAX*T + pad_extra + portal + extra_per_burn_second * burn_seconds(T)
        # burn_seconds(T) = (NITRO_START + charge_per_sec * T) / burn_per_sec
        denominator = E.MAX_SPEED + extra_per_burn_second * charge_per_sec / burn_per_sec
        numerator = track.LENGTH - pad_extra - portal - extra_per_burn_second * E.NITRO_START / burn_per_sec
        fastest_seconds = numerator / denominator
        min_steps = max(1, math.floor(fastest_seconds / E.DT))
        time_points = max(0, E.TIME_POINT * (E.MAX_STEPS - min_steps))
        coin_points = coins * E.COIN_VALUE
        jackpot_points = boxes * E.BOX_POINTS
        result = {"route": route, "coins": coins, "boxes": boxes, "coin_points": coin_points,
                  "min_steps": min_steps, "fastest_seconds": min_steps * E.DT, "time_points": time_points,
                  "jackpot_points": jackpot_points,
                  "score": coin_points + time_points,                              # skill only
                  "score_with_luck": coin_points + time_points + jackpot_points}   # every box a jackpot
        if best is None or result["score"] > best["score"]:
            best = result
    return best


if __name__ == "__main__":                      # python scoring.py  prints the breakdown
    from environment import RacingEnv
    for portals in (True, False):
        r = max_possible_score(RacingEnv(portals=portals))
        print(f"portals {'ON ' if portals else 'OFF'}: max score {r['score']} (skill only) / {r['score_with_luck']} (every box a jackpot)")
        print(f"    best route {r['route']}: {r['coins']} coins = {r['coin_points']} + finish in {r['fastest_seconds']:.1f} s = {r['time_points']} time points"
              f"  (+ {r['boxes']} jackpot boxes = {r['jackpot_points']})")