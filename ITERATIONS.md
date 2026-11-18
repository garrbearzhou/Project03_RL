# Design log: what I changed, why, and what happened

How things were measured: greedy runs (no exploring) from the start line on fresh random seeds, usually 200 or 300 runs,
so a finish rate is good to about +/-3 to 5 points. "Points" is the game's official score (coins + time left when you
finish + jackpot boxes - crash penalty). I list every idea I tried, including the ones that did not help.

## Table of iterations

| # | Change | Why | Result |
|---|---|---|---|
| 1 | Straight road, 5 actions, state = (sideways position, heading, speed) | first working Q-learning agent | 100 of 100 finishes after about 1,000 episodes |
| 2 | Curves. State measured **relative to the road** (offset, heading minus road direction, bend ahead) | an absolute position is useless on a curving road | 187 of 200 finishes after 3,000 episodes |
| 3 | **Track-section bucket** added to the state (hidden portal exists) | the agent could not tell *where* it was, so "hug the edge" paid off at the portal and crashed everywhere else | without it: 13 to 136 of 200 finishes, portal unused. With it: 5 of 6 seeds reached 167 to 200 of 200 |
| 4 | Longer forked track + **random-start practice episodes** (half of training) | a long track kills the agent in the first curves, so it never sees the rest | every section gets practiced; start-line stats only count real races |
| 5 | Walls, rails, jumps, boxes, terrain. **Keep-the-best checkpointing** in `train_headless.py` | tabular Q-learning got *worse* between checkpoints (0/50 to 46/50 swings) | saved best table finished 49 of 50. Shrinking the learning rate over time made learning much worse, so I dropped it |
| 6 | **Nitro, v1**: one straight-only action | free speed boost | used on only 12% of steps; the bar was nearly full (0.99) at the end of a run; in 48% of states it had never been tried |
| 7 | Nitro v2: optimistic starting values for nitro | untried actions start at 0 and lose to learned positives | little change: 9 to 19% of steps |
| 8 | Nitro v3: **nitro + left / nitro + right** (8 actions) | burning in a straight line is risky, so the agent avoided it | 37% of steps; finishes 91% to 99%; average time 28.1 s to 12.6 s |
| 9 | **Points objective** (coins + time left - crash) with a small forward-progress learning signal | optimize what is actually scored, not "go far" | table trained for points: 83% finishes, 1,629 points, 10.2 of 26 coins, 24.1 s (the old +100 finish reward was not A/B tested) |
| 10 | Lookahead grows with speed (100 px + 0.6 s of travel) | a fixed 150 px was 0.25 s at 600 px/s, too late to react | alone: 74% finishes, 1,530 points; 55 of 78 failures still in the S-curve |
| 11 | **Nitro traction control**: nitro cuts out when the car is more than 0.3 rad off the road | the agent burned nitro through curves it could not follow | 97% finishes (290 of 300), 1,972 points, 15.7 coins; S-curve failures 55 to 5 |
| 12 | Danger flag in the state (too fast for the curve ahead), nitro lockout, small learning-only penalty | the S-curve (needs more turning than the car has at 600 px/s) still caused about half the failures | softened version: 95 to 96% finishes, 1,893 to 1,896 points. Overall failures 23 to 15 per 300 runs |
| 13 | Route balance: forced-route agents, then 3 extra Scenic coins | the agent always took the Shortcut | Scenic-only 1,774 vs Shortcut-only 1,836 points (about +61). Coins evened the value, but the habit stayed (see below) |
| 14 | Portal jump 1,000 px to 600 px, patches shrunk to 30 x 10 px; separate **no-portal table** trained with portals off | portals should not decide the game | portal 95%, 1,896 points; no-portal 94%, 1,861 points. A portal is worth very little |
| 15 | Code refactor (binary-search object lookups, small methods) + **52 tests** | faster experiments; catch rule bugs in seconds | 28.6 to 16.7 to 19.3 microseconds per step (about 1.6x faster), identical behavior (fingerprint check). Tests catch 5 of 5 deliberately broken rules |
| 16 | **Jackpot box** (50 coins = 1,250 points, 1 slot in 13), prizes **re-dealt every run**, **max possible score** in the HUD | boxes can now be worth a detour; compare real runs with the ceiling | average 1,896 to 2,032 with jackpots (96% finishes). Best skill-only run 2,078 = 84% of the 2,463 maximum. The agent opens only 1.4 boxes per run, so it has not learned boxes pay |

## Things that did not help (and what I learned)
- **Learning-rate decay** over time: much worse (it froze the table before it was good).
- **Gamma 0.995**: best checkpoint 182 of 200 (1,843 points) vs 190 of 200 (1,937) for gamma 0.99.
- **A per-step time cost of 0.05**: best checkpoint 176 of 200 (1,767 points) vs 191 of 200 (1,947). It is also unsafe: with
  0.05 x 2400 steps the cost of timing out is bigger than the crash penalty, so crashing early could look better. The option
  (`STEP_COST`) stays at 0 and its comment says to keep it at 0.04 or less.
- **Strong danger penalty** (1.5 with a threshold at 85% of the car's turn rate): the agent braked on 85% of steps at the
  start of the track and finished only 62%. The threshold sat at 99% of normal driving speed, so ordinary driving was punished.
- **Stricter nitro lockout** (80% of the turn rate): the saved table fell to 141 of 200 and recovered only to about 164 of 200.
- **Extra exploring at the fork** (0.25 and 0.08): the agent still took the Shortcut in 300 of 300 runs, and finish rates dropped.
  This looks like a learned habit (a classic exploration problem), not a real gap in route value.
- **Optimistic nitro values on their own**: almost no change. The real problem was that nitro could not steer.

## The scoring rules and the maximum score
Score = coins x 25 + jackpot points (1,250 each) + steps left on the clock (only if you finish) - 100 for a crash.
`python scoring.py` prints the best possible score and how it is built:

| | with portals | no portals |
|---|---|---|
| Skill-only maximum (boxes ignored) | **2,463** (25 coins = 625, finish in 18.7 s = 1,838) | 2,419 |
| If every box on the best route were a jackpot | 11,213 | 11,169 |

The fastest finish is an optimistic bound (constant 300 px/s, every pad's full boost and slow-down tail, nitro in the best
bursts, one portal jump, no curves or terrain slowdowns), so no real run can beat it. A test checks that.

## How to reproduce
```
python tests.py                                   # 52 checks of the rules and the learning code
python scoring.py                                 # the maximum possible score
python train_headless.py 30000 --alpha 0.03 --epsilon-min 0.02   # fine-tune with a gentle learning rate
python train_headless.py 30000 --route Scenic --out scenic_only.pkl   # route experiment
```
