"""
agent.py - tabular Q-learning agent.

AI USE DISCLOSURE (Honor Code): Generated with help from Claude (Anthropic) as a
starting point. <-- Add your own notes on what you changed/understand.

The Q-table starts EMPTY (all zeros): the agent knows nothing about the rules and
learns only from rewards.   Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
"""
import pickle
import random


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 initial_q=None):
        self.n_actions = n_actions
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # how much future reward matters
        self.epsilon = epsilon              # chance of exploring (random action)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = {}                         # state -> list of action values
        # Optimistic starting values (one per action). An action that starts HIGH gets tried at least a
        # few times; otherwise an action that starts at 0 can be ignored forever once the other
        # actions have learned positive values (this is why the agent barely used nitro).
        self.initial_q = list(initial_q) if initial_q else [0.0] * n_actions
        self.greedy = False                 # True = always exploit (demo mode)

    def _values(self, state):
        if state not in self.q:
            self.q[state] = list(self.initial_q)
        return self.q[state]

    def choose_action(self, state):
        """Epsilon-greedy policy: explore sometimes, otherwise pick the best known action."""
        if not self.greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        values = self._values(state)
        best = max(values)
        return random.choice([a for a, v in enumerate(values) if v == best])

    def learn(self, state, action, reward, next_state, done):
        """One Q-learning update."""
        if self.greedy:
            return
        values = self._values(state)
        target = reward if done else reward + self.gamma * max(self._values(next_state))
        values[action] += self.alpha * (target - values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "epsilon": self.epsilon}, f)

    def load(self, path):
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