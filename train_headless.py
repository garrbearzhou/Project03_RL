"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
train_headless.py - train the agent WITHOUT a window (much faster than main.py).

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

Usage:  python train_headless.py 40000        (number of episodes; default 30000)
        python train_headless.py 40000 7      (optional 2nd number = random seed, to repeat a run)
        python train_headless.py 40000 --no-portal   (train the no-portal version; separate Q-table)
Every 5000 episodes the agent is tested (greedy, from the start line). The saved Q-table is the
one with the highest average points, because points (coins + time left at the finish) are the
objective, and Q-learning on a hard track can get worse again after a good moment.
Then run main.py to watch what it learned (it loads the same saved Q-table).
"""
import os
import random
import sys
import time

from agent import QLearningAgent
from environment import RacingEnv
import settings

EVAL_EVERY = 5000
EVAL_RUNS = 200                      # 200 test races per checkpoint (50 was too noisy: about +/-5% on a finish rate)


def evaluate(agent, env):
    """Greedy start-line races (no exploring, no learning).
    Returns (finishes, average points). Points are the objective."""
    was_greedy, agent.greedy = agent.greedy, True
    wins = steps = burning = 0
    total_points = 0
    for _ in range(EVAL_RUNS):
        state, done = env.reset(), False
        while not done:
            state, _, done, info = env.step(agent.choose_action(state))
            steps += 1
            burning += env.nitro_burning
        wins += info["finished"]
        total_points += info["score"]
    agent.greedy = was_greedy
    evaluate.nitro_share = 100 * burning / max(steps, 1)       # % of steps spent burning nitro
    return wins, total_points / EVAL_RUNS


def main(episodes, seed=None, no_portal=False):
    q_file = settings.Q_FILE_NO_PORTAL if no_portal else settings.Q_FILE
    if seed is not None:
        random.seed(seed)
    env, eval_env = RacingEnv(portals=not no_portal), RacingEnv(portals=not no_portal)
    agent = QLearningAgent(env.N_ACTIONS, **settings.AGENT_KWARGS)
    if os.path.exists(q_file):
        agent.load(q_file)
        print("Loaded saved Q-table (continuing from it).")

    best_points, t0 = -1e18, time.time()
    if os.path.exists(q_file):                 # don't overwrite a good saved table with a worse one
        loaded_wins, best_points = evaluate(agent, eval_env)
        print(f"Saved Q-table: {loaded_wins}/{EVAL_RUNS} finishes, average {best_points:.0f} points.")
    starts = wins = 0
    for ep in range(1, episodes + 1):
        state = env.reset(random_start=random.random() < settings.RANDOM_START_PROB)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
        if info["from_start"]:
            starts += 1
            wins += info["finished"]
        if ep % EVAL_EVERY == 0 or ep == episodes:
            wins_eval, avg_points = evaluate(agent, eval_env)
            note = ""
            if avg_points >= best_points:
                best_points = avg_points
                agent.save(q_file)
                note = "  <- best points so far, saved"
            print(f"episode {ep}: epsilon {agent.epsilon:.2f}, training finishes {wins}/{starts}, "
                  f"greedy test {wins_eval}/{EVAL_RUNS} finishes, {avg_points:.0f} avg points "
                  f"(nitro on {evaluate.nitro_share:.0f}% of steps), states {len(agent.q)}, {time.time() - t0:.0f}s{note}")
            starts = wins = 0
    print(f"Done. Best average points: {best_points:.0f}. Saved {q_file}")


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    nums = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    main(nums[0] if nums else 50000, nums[1] if len(nums) > 1 else None, "--no-portal" in flags)