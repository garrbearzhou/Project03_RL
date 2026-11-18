# Project03_RL — PygAIme Racer

Garrett Zhou  
ADV CS III: Machine Learning and AI  
Project 03: Reinforcement Learning (PyGAIme)

**PygAIme Racer** is a custom Pygame racing game built around reinforcement learning. At the start of training the agent has an empty Q-table: it does not know that walls hurt, that coins are good, that nitro can throw it off a curve, or that a hidden portal even exists. It only receives a small discrete **state**, chooses among **8 actions**, and updates from a **reward** after every simulation step. Over thousands of episodes it learns a policy that can finish a long, forked track and score points competitively.

The track is much richer than a typical classroom Snake/Brick tutorial. It includes:

- A **Scenic vs Shortcut fork**, a **grass gap** that must be jumped, moving/static **walls**, **guard rails**, **boost pads**, **jump ramps**, **ice / mud / mountain** terrain, **mystery boxes** (including rare jackpots), and optional **hidden portals**
- An official **points** objective: coins × 25 + jackpot points + unused time if you finish − 100 for a crash (so finishing fast usually beats collecting every coin)
- A separate **learning reward** with small shaping signals (forward progress, danger, rails) so a ~12,000 px track is actually learnable with tabular Q-learning

There are two trained “models” (separate Q-tables): **With portals** and **No portals**. The Pygame UI lets you choose which model to load, then whether to run in **Greedy (demo)** mode (no learning — safe to show in class) or **Train** mode (updates the table). Serious training is done faster with `train_headless.py`, which periodically evaluates greedy start-line races and **keeps the best average-points checkpoint** because Q-learning on a hard track can get worse again after a good peak.

The project also includes `tests.py` (automated checks of rules and learning), `scoring.py` (an optimistic maximum possible score for the HUD), and `ITERATIONS.md` (a log of design experiments — including ideas that failed, like learning-rate decay or over-strong danger penalties).

---

## Project Objective

Use Python + Pygame to implement a full reinforcement-learning game pipeline that:

1. Starts with an empty Q-table (no built-in knowledge of the rules)
2. Improves over training episodes with tabular Q-learning and an ε-greedy policy
3. Uses a carefully designed **state space** (road-relative offset, heading, speed, bend/danger, track section, nearby objects, extras) and a **reward** aligned with the real score
4. Lets you **watch** or **train** in a Pygame UI, including **With portals** vs **No portals**
5. Supports fast headless training, automated tests, and a documented score ceiling so results are comparable
6. Goes beyond standard tutorials with forks, nitro, terrain, boxes, portals, and an iteration log of what was tuned

---

## Setup

```bash
# Python 3.10+ recommended
cd Project03_RL
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pygame
```

Optional check that everything works:

```bash
python tests.py      # 52 rule + learning checks
python scoring.py    # prints max possible score
python main.py       # opens the UI menus
```

---

## How to Run

### 1) Tests (run after code changes)

```bash
python tests.py
```

### 2) Score ceiling

```bash
python scoring.py
```

Example output (skill-only vs all-jackpot luck):

| Mode | Skill-only max | All jackpots |
|------|----------------|--------------|
| Portals ON | ~2388 | ~11138 |
| Portals OFF | ~2344 | ~11094 |

See `SCORE_CEILING.md` for how to read those numbers.

### 3) Train (recommended: headless)

```bash
# Portals ON  → saves q_table_items.pkl
python train_headless.py 30000

# Portals OFF → saves q_table_items_noportal.pkl
python train_headless.py 30000 --no-portal
```

Headless training evaluates every 5000 episodes and **keeps the best average-points checkpoint** (Q-learning can get worse after a peak).

**Do not** open the UI on the same model while that model is training in headless mode — both write the same `.pkl` and can overwrite each other.

### 4) UI / demo

```bash
python main.py
```

Startup menus:

1. **With portals** or **No portals**
2. **Greedy (demo)** or **Train (learning on)**

Shortcuts that skip menus:

```bash
python main.py --portal --greedy
python main.py --no-portal --train
```

#### In-game controls

| Key | Action |
|-----|--------|
| `SPACE` | Fast training / watch mode |
| `G` | Toggle greedy |
| `R` | Random practice starts on/off |
| `M` | Switch portal ↔ no-portal model (saves first if learning) |
| `P` | Reveal hidden portal patches |
| `H` | Hide / show HUD |
| `S` | Save Q-table now |
| `ESC` | Quit (saves only if this session was learning) |

**Demo tip:** choose a trained model → **Greedy** → watch finishes and points. Greedy demos do **not** overwrite the saved Q-table on quit.

---

## Reinforcement Learning Design

### Agent–environment loop

1. `state = env.reset(...)`
2. `action = agent.choose_action(state)` (ε-greedy)
3. `next_state, reward, done, info = env.step(action)`
4. `agent.learn(state, action, reward, next_state, done)`
5. Repeat until crash / finish / timeout; decay ε; next episode

### State space (discretized)

Relative to the nearest road (so curves are learnable):

`(offset, heading, speed, bend/danger, section, ahead_object, extras)`

- **Section bins** tell the agent where it is on the long track
- **Lookahead** grows with speed
- **Danger flag** when too fast for the curve ahead

### Actions (8)

Accel straight / left / right, coast, brake, nitro straight / left / right.

### Score vs learning reward

- **Official score (HUD):** `coins × 25 + jackpot points + time left (if finished) − 100 on crash`
- **Learning reward:** related score signals plus small forward-progress / rail / danger shaping so a ~12,000 px track is learnable

### Policy / learning

- Tabular Q-learning:  
  `Q(s,a) ← Q(s,a) + α [r + γ max Q(s′,a′) − Q(s,a)]`
- ε-greedy exploration with decay
- Nitro actions start with **optimistic** Q values so they get tried
- Random-start practice episodes teach late track sections

Design experiments (what worked / what failed) are logged in `ITERATIONS.md`.

---

## Results (approximate)

Trained companion Q-tables are included in the repo:

| Model | File | Approx. greedy eval |
|-------|------|---------------------|
| With portals | `q_table_items.pkl` | ~84% finishes, ~1970 avg points |
| No portals | `q_table_items_noportal.pkl` | ~50% finishes, ~1210 avg points (still improvable) |

Skill-only ceilings are ~2388 (portals on) and ~2344 (portals off). Jackpot luck can push a single run much higher; the HUD compares best points to a fair ceiling (skill max + that run’s jackpots).

---

## Project Layout

| File | Role |
|------|------|
| `main.py` | Pygame UI (model menu, greedy/train menu, live play) |
| `train_headless.py` | Fast training + best-checkpoint save |
| `environment.py` | Physics, state encoding, rewards, scoring rules |
| `track.py` | Roads, fork, gap, coins, pads, walls, portals, terrain |
| `agent.py` | Tabular Q-learning agent |
| `renderer.py` | Drawing + HUD |
| `settings.py` | Shared hyperparameters / Q-table filenames |
| `scoring.py` | Optimistic maximum possible score |
| `models.py` | Portal vs no-portal model helpers |
| `tests.py` | Automated checks (rules + learning) |
| `ITERATIONS.md` | Experiment / iteration log |
| `SCORE_CEILING.md` | Notes on `scoring.py` output |
| `q_table_items.pkl` | Trained agent (portals on) |
| `q_table_items_noportal.pkl` | Trained agent (portals off) |

---

## Dependencies

| Package | Role |
|---------|------|
| `pygame` | Game window, input, rendering |
| Python standard library | Q-table pickle I/O, math, unittest |

No GPU required (tabular Q-learning).

---

## AI Assistance (Honor Code)

Claude (Anthropic) and Cursor helped scaffold and extend the Pygame + Q-learning pipeline (track features, training loop, tests, UI menus, refactors). I directed the design goals (points objective, nitro that can steer, danger flag, portal tuning, dual models) and verified behavior with `tests.py` and training runs.

Each main `.py` file has an **AI USE DISCLOSURE** header with file-specific notes. Online RL/Pygame tutorials from the assignment PDF were used as general inspiration; this racer is a custom game (forks, portals, nitro, terrain, mystery boxes), not a copy of a standard tutorial.

---

## Submission Notes

- GitHub repo name: `Project03_RL`
- Companion datasets: `q_table_items.pkl`, `q_table_items_noportal.pkl`
- Please add collaborator `@juliancochran`
- Submit the repo **https** URL through Veracross
