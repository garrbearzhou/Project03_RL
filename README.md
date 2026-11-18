Project03_RL — PygAIme Racer
ADV CS III: Machine Learning and AI — Project 03: Reinforcement Learning (PyGAIme)

A Pygame racing game where a tabular Q-learning agent starts knowing nothing about the rules and improves by playing. The agent never reads the physics code; it only sees a small discrete state, chooses an action, and learns from rewards.

Objective
Train an AI racer to finish a long, forked track for points (coins + time left at the finish − crash penalty + optional jackpot boxes), not just “survive.” Over many episodes the Q-table fills in which actions work in which situations.

Project structure
File	Role
main.py	Pygame window: watch / interactive train
train_headless.py	Fast training (no window); saves best Q-table by average points
environment.py	Rules, physics, state encoding, rewards
track.py	Road layout (forks, gap, coins, pads, walls, portals, …)
agent.py	Tabular Q-learning (ε-greedy policy)
renderer.py	Drawing and HUD
settings.py	Shared hyperparameters and Q-table filenames
scoring.py	Optimistic maximum possible score
models.py	Portal vs no-portal trained models
tests.py	52 automated checks of rules + learning
ITERATIONS.md	Design log: what was tried and what worked
SCORE_CEILING.md	Notes on scoring.py output
q_table_items.pkl	Trained agent (portals on)
q_table_items_noportal.pkl	Trained agent (portals off)
How to run
Use the project venv (or any environment with pygame installed).

# 1) After code changes — quality gate
python tests.py

# 2) Score ceiling (skill-only vs all-jackpot luck)
python scoring.py

# 3) Train without a window (recommended for long runs)
python train_headless.py 30000                 # portals ON  -> q_table_items.pkl
python train_headless.py 30000 --no-portal     # portals OFF -> q_table_items_noportal.pkl

# 4) Watch / demo — menus: model, then Greedy vs Train
python main.py
# optional shortcuts that skip menus:
python main.py --portal --greedy
python main.py --no-portal --train
main.py controls
Key	Action
1 / 2 or UP/DOWN + ENTER	Choose on the start menus (model, then play mode)
M	Switch With portals ↔ No portals (saves current table first)
SPACE	Toggle fast training / watch mode
G	Greedy mode (no exploring; start-line races)
R	Random practice starts on/off
P	Reveal hidden portals
H	Hide / show HUD
S	Save Q-table
ESC	Quit (auto-saves)
Demo tip: Load a trained table → press G → watch finish rate and points on the HUD. Compare scores to python scoring.py (skill-only ceiling ≈ 2388 with portals).

Reinforcement learning design
Agent–environment loop
state = env.reset(...)
action = agent.choose_action(state) (ε-greedy)
next_state, reward, done, info = env.step(action)
agent.learn(state, action, reward, next_state, done)
Repeat until crash / finish / timeout; decay ε; next episode
State space (discretized)
Relative to the nearest road (so curves are learnable):

(offset, heading, speed, bend/danger, section, ahead_object, extras)

Section bins tell the agent where it is on the long track
Lookahead grows with speed
Danger flag when too fast for the curve ahead
Actions (8)
Accel straight / left / right, coast, brake, nitro straight / left / right.

Rewards vs score
Official score: coins×25 + jackpot points + time left (if finished) − 100 on crash
Learning reward: related score signals + small forward-progress + rail/danger penalties (so a 12,000 px track is learnable)
Policy / learning
Tabular Q-learning: (Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)])
ε-greedy exploration with decay
Nitro actions start with optimistic Q values so they get tried
train_headless.py keeps the best average-points checkpoint (Q-learning can get worse after a peak)
More experiment history: see ITERATIONS.md.

Testing and refinement
python tests.py          # rules + Q-learning unit tests
python scoring.py        # max possible score breakdown

The project was iterated many times (state features, nitro steering, danger signals, points objective, portal value, jackpots). Failed ideas (e.g. learning-rate decay, strong danger penalties) are recorded in ITERATIONS.md so they are not repeated blindly.

AI / collaboration disclosure (Honor Code)
AI assistance (Claude / Anthropic, and later Cursor) was used to help scaffold and extend the Pygame + Q-learning pipeline (track features, training loop, tests, refactoring).

I am responsible for understanding and defending the design: the agent–environment loop, state and action choices, reward vs score, training workflow, and the results in ITERATIONS.md. Inline AI USE DISCLOSURE comments appear at the top of the main Python files. Online RL/Pygame tutorials listed in the assignment PDF were used as general inspiration; this racer is a custom game (forks, portals, nitro, terrain, mystery boxes), not a copy of Snake/Brick Breaker.

Classmates may discuss ideas; code was not shared as a completed solution.

Submission notes for Mr. Cochran
GitHub repo: Project03_RL (push all project files + companion Q-tables)
Please add collaborator @juliancochran if not already added
Submit the repo https URL through Veracross
Companion datasets: q_table_items.pkl, q_table_items_noportal.pkl
Requirements
Python 3.10+ recommended
pygame
No GPU required (tabular Q-learning)
