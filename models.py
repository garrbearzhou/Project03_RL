"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
models.py - the trained versions ("models") of the agent, and a quick way to compare them.

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

A "model" here is one saved Q-table plus the version of the game it was trained for:
  * With portals : q_table_items.pkl           (the hidden portals are switched on)
  * No portals   : q_table_items_noportal.pkl  (the hidden portals are switched off)
main.py shows a menu built from MODELS at start-up, and the M key switches between them while playing.
To add a third model, add one more entry to MODELS (and a Q-table file for it).
"""
import os
import random

import scoring
import settings
from agent import QLearningAgent
from environment import RacingEnv

MODELS = [
    {"key": "portal", "label": "With portals", "no_portal": False, "file": settings.Q_FILE,
     "blurb": "Hidden portals are on: one can jump the car 600 px ahead."},
    {"key": "noportal", "label": "No portals", "no_portal": True, "file": settings.Q_FILE_NO_PORTAL,
     "blurb": "Portals are off: the only way to win is to drive."},
]


def model_for(no_portal):
    """The model for the portal version (no_portal=False) or the no-portal version (no_portal=True)."""
    return MODELS[1 if no_portal else 0]


def has_saved_table(model):
    return os.path.exists(model["file"])


def quick_test(model, runs=30, seed=2024):
    """Play `runs` greedy races (no exploring, no learning) with the model's saved Q-table.
    Returns a dict with the results, or None if the model has no saved table yet.
    It saves and restores Python's random state, so it never disturbs the rest of the program."""
    if not has_saved_table(model):
        return None
    saved_state = random.getstate()
    try:
        random.seed(seed)
        env = RacingEnv(portals=not model["no_portal"])
        agent = QLearningAgent(env.N_ACTIONS, **settings.AGENT_KWARGS)
        agent.load(model["file"])
        agent.greedy = True
        finished, total, best = 0, 0, None
        for _ in range(runs):
            state, done = env.reset(), False
            while not done:
                state, _, done, info = env.step(agent.choose_action(state))
            finished += info["finished"]
            total += info["score"]
            best = info["score"] if best is None else max(best, info["score"])
        return {"runs": runs, "finished": finished, "avg_points": total / runs, "best_points": best,
                "states": len(agent.q), "max_score": scoring.max_possible_score(env)["score"]}
    finally:
        random.setstate(saved_state)