Project03_RL — PygAIme Racer
Garrett Zhou
ADV CS III: Machine Learning and AI
Project 03: Reinforcement Learning (PyGAIme)

A Pygame racing game where a tabular Q-learning agent starts knowing nothing about the rules and improves by playing. The agent never reads the physics code. It only sees a small discrete state, chooses an action, and learns from rewards. The goal is to finish a long forked track for points (coins + time left at the finish − crash penalty + optional jackpot boxes).

### Project Objective
Build a reinforcement-learning game (not a copy of Snake / Brick Breaker) that:

## Starts with an empty Q-table (no built-in knowledge of the rules)
Improves over training episodes with tabular Q-learning and an ε-greedy policy
Lets you watch or train in a Pygame UI, including With portals vs No portals models
Supports fast headless training, automated tests, and a documented score ceiling so results are comparable
# Setup
Python 3.10+ recommended
cd Project03_RL
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pygame
Optional check that everything works:

python tests.py      # 52 rule + learning checks
python scoring.py    # prints max possible score
python main.py       # opens the UI menus

### How to Run
# 1) Tests (run after code changes)
python tests.py
# 2) Score ceiling
python scoring.py
Example output (skill-only vs all-jackpot luck):

Mode	Skill-only max	All jackpots
Portals ON	~2388	~11138
Portals OFF	~2344	~11094
See SCORE_CEILING.md for how to read those numbers.

# 3) Train (recommended: headless)
Portals ON  → saves q_table_items.pkl
python train_headless.py 30000

Portals OFF → saves q_table_items_noportal.pkl
python train_headless.py 30000 --no-portal

Headless training evaluates every 5000 episodes and keeps the best average-points checkpoint (Q-learning can get worse after a peak).

Do not open the UI on the same model while that model is training in headless mode — both write the same .pkl and can overwrite each other.

# 4) UI / demo
python main.py
Startup menus:

With portals or No portals
Greedy (demo) or Train (learning on)

## Shortcuts that skip menus:

python main.py --portal --greedy
python main.py --no-portal --train
In-game controls
Key	Action
SPACE	Fast training / watch mode
G	Toggle greedy
R	Random practice starts on/off
M	Switch portal ↔ no-portal model (saves first if learning)
P	Reveal hidden portal patches
H	Hide / show HUD
S	Save Q-table now
ESC	Quit (saves only if this session was learning)
Demo tip: choose a trained model → Greedy → watch finishes and points. Greedy demos do not overwrite the saved Q-table on quit.

### Reinforcement Learning Design
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

# Score vs learning reward
Official score (HUD): coins × 25 + jackpot points + time left (if finished) − 100 on crash
Learning reward: related score signals plus small forward-progress / rail / danger shaping so a ~12,000 px track is learnable
Policy / learning
Tabular Q-learning:
Q(s,a) ← Q(s,a) + α [r + γ max Q(s′,a′) − Q(s,a)]
ε-greedy exploration with decay
Nitro actions start with optimistic Q values so they get tried
Random-start practice episodes teach late track sections
Design experiments (what worked / what failed) are logged in ITERATIONS.md.

### Results (approximate)
Trained companion Q-tables are included in the repo:

Model	File	Approx. greedy eval
With portals	q_table_items.pkl	~84% finishes, ~1970 avg points
No portals	q_table_items_noportal.pkl	~50% finishes, ~1210 avg points (still improvable)
Skill-only ceilings are ~2388 (portals on) and ~2344 (portals off). Jackpot luck can push a single run much higher; the HUD compares best points to a fair ceiling (skill max + that run’s jackpots).

### Project Layout
File	Role
main.py	Pygame UI (model menu, greedy/train menu, live play)
train_headless.py	Fast training + best-checkpoint save
environment.py	Physics, state encoding, rewards, scoring rules
track.py	Roads, fork, gap, coins, pads, walls, portals, terrain
agent.py	Tabular Q-learning agent
renderer.py	Drawing + HUD
settings.py	Shared hyperparameters / Q-table filenames
scoring.py	Optimistic maximum possible score
models.py	Portal vs no-portal model helpers
tests.py	Automated checks (rules + learning)
ITERATIONS.md	Experiment / iteration log
SCORE_CEILING.md	Notes on scoring.py output
q_table_items.pkl	Trained agent (portals on)
q_table_items_noportal.pkl	Trained agent (portals off)
Dependencies
Package	Role
pygame	Game window, input, rendering
Python standard library	Q-table pickle I/O, math, unittest
No GPU required (tabular Q-learning).

### AI Assistance (Honor Code)
Claude (Anthropic) and Cursor helped scaffold and extend the Pygame + Q-learning pipeline (track features, training loop, tests, UI menus, refactors). I directed the design goals (points objective, nitro that can steer, danger flag, portal tuning, dual models) and verified behavior with tests.py and training runs.

Each main .py file has an AI USE DISCLOSURE header with file-specific notes. Online RL/Pygame tutorials from the assignment PDF were used as general inspiration; this racer is a custom game (forks, portals, nitro, terrain, mystery boxes), not a copy of a standard tutorial.

### Submission Notes
GitHub repo name: Project03_RL
Companion datasets: q_table_items.pkl, q_table_items_noportal.pkl
Please add collaborator @juliancochran
Submit the repo https URL through Veracross
