"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
agent.py - tabular Q-learning agent.

AI USE DISCLOSURE (Honor Code): Scaffolded with help from Claude (Anthropic). I own the
learning hyperparameters, optimistic nitro initialization, save/load upgrade path, and the
min_epsilon override used for experiments. I can explain the Q-update and ε-greedy policy
without the code in front of me.

The Q-table starts EMPTY: the agent knows nothing about the game rules and learns only from
rewards. Update rule:

    Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
    target = r                         if the episode ended
    target = r + gamma * max_a' Q(s')  otherwise
"""
import pickle
import random


class QLearningAgent:
    """Learns a value for every (state, action) pair and picks actions with ε-greedy."""

    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 initial_q=None):
        self.n_actions = n_actions
        self.alpha = alpha                  # learning rate: how hard each new sample overwrites the old estimate
        self.gamma = gamma                  # discount: how much future reward matters vs immediate reward
        self.epsilon = epsilon              # explore probability (1.0 = fully random at the start)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = {}                         # state tuple -> list of action values (length n_actions)
        # Optimistic starting values (one per action). An action that starts HIGH gets tried at least a
        # few times; otherwise an action that starts at 0 can be ignored forever once the other
        # actions have learned positive values (this is why the agent barely used nitro).
        self.initial_q = list(initial_q) if initial_q else [0.0] * n_actions
        self.greedy = False                 # True = always exploit (demo / evaluation mode; no learning)

    def _values(self, state):
        """Return the action-value row for this state, creating it on first visit."""
        if state not in self.q:
            self.q[state] = list(self.initial_q)
        return self.q[state]

    def choose_action(self, state, min_epsilon=None):
        """Epsilon-greedy policy: explore sometimes, otherwise pick the best known action.

        min_epsilon (optional) raises the explore rate for this choice only (e.g. while in the fork).
        When several actions tie for best, one of them is chosen at random.
        """
        eps = self.epsilon if min_epsilon is None else max(self.epsilon, min_epsilon)
        if not self.greedy and random.random() < eps:
            return random.randrange(self.n_actions)
        values = self._values(state)
        best = max(values)
        return random.choice([a for a, v in enumerate(values) if v == best])

    def learn(self, state, action, reward, next_state, done):
        """One Q-learning update for the transition (state, action) -> (reward, next_state)."""
        if self.greedy:
            return
        values = self._values(state)
        # Terminal states have no future value; otherwise bootstrap from the best next action.
        target = reward if done else reward + self.gamma * max(self._values(next_state))
        values[action] += self.alpha * (target - values[action])

    def decay_epsilon(self):
        """Shrink exploration after each episode, but never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Write the Q-table and current epsilon to disk (companion dataset for the assignment)."""
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "epsilon": self.epsilon}, f)

    def load(self, path):
        """Load a saved Q-table. Older tables with fewer actions are padded automatically."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q, self.epsilon = data["q"], data["epsilon"]
        # A table saved by an older version has fewer actions per state (e.g. 6 instead of 8).
        # Pad the missing actions with their starting values so the old table still works.
        padded = 0
        for row in self.q.values():
            if len(row) < self.n_actions:
                row.extend(self.initial_q[len(row):])
                padded += 1
            elif len(row) > self.n_actions:
                raise ValueError(f"Saved Q-table has {len(row)} actions per state but this game has "
                                 f"{self.n_actions}. Delete the .pkl file to start fresh.")
        if padded:
            print(f"Upgraded {padded} states of an older Q-table to {self.n_actions} actions "
                  "(delete the .pkl to start fresh instead).")