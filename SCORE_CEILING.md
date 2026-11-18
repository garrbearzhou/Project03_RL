# Max possible score (`python scoring.py`)

Theoretical ceiling from the rules — not what the agent actually scores.
No real run should beat the **skill only** number.

## Portals ON
- **Skill-only max: 2388**
- Best optimistic route: **Scenic**
  - 22 coins × 25 = **550**
  - Fastest optimistic finish **18.7 s → 1838** time points
  - 550 + 1838 = **2388**
- **With luck: 11138**
  - Same run, but every mystery box on that route is a jackpot
  - 7 boxes × 1250 = **8750**
  - 2388 + 8750 = **11138**
- Portal helps a bit: the model assumes one **600 px** jump, so the optimistic time is slightly faster than with portals off.

## Portals OFF
- **Skill-only max: 2344** (= 550 coins + 1794 time)
- Finish bound is slower (**20.2 s**) because there’s no portal shortcut in the math
- **With luck: 11094** (same 8750 jackpot add-on)

## How to read it for the agent
| Number | Meaning |
|---|---|
| **2388 / 2344** | “Perfect driving” ceiling (coins + fast finish; boxes ignored) |
| **11138 / 11094** | Absurd luck ceiling (every box a jackpot) |
| **Scenic** | On this optimistic model, Scenic beats Shortcut |
| Your HUD run score | Compare to **~2388**, not 11138 |

If the agent scores ~1900–2100, that’s a large chunk of the skill ceiling; jackpot totals are mostly lottery, not skill.

Recompute anytime with:
```bash
python scoring.py
```
