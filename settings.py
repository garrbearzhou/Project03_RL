"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
settings.py - values shared by main.py (watch/train in a window) and train_headless.py.

Change learning knobs here so the windowed and headless trainers stay in sync.
Delete a .pkl file to force the agent to forget and relearn from scratch.
"""
Q_FILE = "q_table_items.pkl"      # saved Q-table (delete it to make the agent forget everything)
Q_FILE_NO_PORTAL = "q_table_items_noportal.pkl"   # the no-portal version keeps its own Q-table
RANDOM_START_PROB = 0.5           # during training, this fraction of episodes start at a random spot
# Extra explore rate while the car is in the fork. Tried 0.08 / 0.25; did not help route balance, so leave
# as None (main.py then passes no override). Set a float like 0.08 to force more exploring in the fork.
FORK_EPSILON = None
# alpha=learning rate, gamma=discount, epsilon*=exploration schedule.
# initial_q: last three entries are nitro actions (optimistic so they get tried early).
AGENT_KWARGS = dict(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                   initial_q=[0, 0, 0, 0, 0, 8.0, 8.0, 8.0])