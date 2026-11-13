"""settings.py - values shared by main.py (watch/train in a window) and train_headless.py."""
Q_FILE = "q_table_items.pkl"      # saved Q-table (delete it to make the agent forget everything)
Q_FILE_NO_PORTAL = "q_table_items_noportal.pkl"   # the no-portal version keeps its own Q-table
RANDOM_START_PROB = 0.5           # during training, this fraction of episodes start at a random spot
AGENT_KWARGS = dict(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                   initial_q=[0, 0, 0, 0, 0, 8.0, 8.0, 8.0])   # the 3 NITRO actions start optimistic so they get tried