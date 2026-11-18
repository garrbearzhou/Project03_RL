"""
Title: Project03_RL
Author: Garrett Zhou
Version: 11/17
main.py - Pygame entry point: Environment <-> Agent <-> Renderer.

AI USE DISCLOSURE: https://claude.ai/share/c3bb3a14-6477-4b27-9563-07eb366f5d20

At launch you pick With portals / No portals, then Greedy (demo) or Train (learning).
You can still pass --no-portal / --portal to skip the model menu, and --greedy / --train
to skip the play-mode menu.

Controls:
  1 / 2 or UP/DOWN+ENTER   choose on the start menus
  M      switch portal / no-portal model while playing (saves the current table first)
  SPACE  toggle fast training / watch mode
  G      greedy (no exploring; always starts at the start line)
  R      random practice starts on/off
  P      reveal hidden portal patches
  H      hide / show the HUD
  S      save Q-table
  ESC    quit (auto-saves)
"""
import os
import random
import sys
import time
import pygame

from environment import RacingEnv
from agent import QLearningAgent
import models
import renderer
import scoring
import settings

PLAY_MODES = [
    {
        "key": "greedy",
        "label": "Greedy (demo)",
        "greedy": True,
        "blurb": "No exploring, no learning. Safe to watch a trained table without changing it.",
    },
    {
        "key": "train",
        "label": "Train (learning on)",
        "greedy": False,
        "blurb": "Keeps updating the Q-table. Can overwrite your .pkl when you save/quit.",
    },
]


def _menu_fonts():
    return (
        pygame.font.SysFont("consolas", 28, bold=True),
        pygame.font.SysFont("consolas", 22),
        pygame.font.SysFont("consolas", 16),
    )


def _draw_choice_menu(screen, title, subtitle, options, selected):
    """Generic startup menu: title + numbered options with blurbs."""
    title_font, font, small = _menu_fonts()
    screen.fill((18, 22, 32))
    t = title_font.render(title, True, (240, 244, 250))
    screen.blit(t, (renderer.W // 2 - t.get_width() // 2, 80))
    s = font.render(subtitle, True, (160, 175, 200))
    screen.blit(s, (renderer.W // 2 - s.get_width() // 2, 130))

    for i, opt in enumerate(options):
        active = i == selected
        color = (255, 220, 120) if active else (220, 228, 240)
        prefix = ">" if active else " "
        y = 210 + i * 110
        screen.blit(font.render(f"{prefix}  {i + 1}. {opt['label']}", True, color), (90, y))
        for j, line in enumerate(opt.get("lines") or [opt.get("blurb", "")]):
            screen.blit(small.render(f"    {line}", True, (150, 165, 190)), (90, y + 28 + j * 22))

    hint = small.render("UP/DOWN or 1/2 to select    ENTER to confirm    ESC to quit", True, (150, 165, 190))
    screen.blit(hint, (renderer.W // 2 - hint.get_width() // 2, renderer.H - 60))
    pygame.display.flip()


def _run_choice_menu(screen, clock, title, subtitle, options, default=0):
    """Keyboard picker. Returns the selected option dict, or None if the user quits."""
    selected = default
    while True:
        _draw_choice_menu(screen, title, subtitle, options, selected)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(options)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_1:
                    selected = 0
                elif event.key == pygame.K_2 and len(options) > 1:
                    selected = 1
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    return options[selected]
        clock.tick(30)


def choose_model(screen, clock):
    """Return a MODELS entry from the startup menu (or from --portal / --no-portal)."""
    if "--no-portal" in sys.argv:
        return models.model_for(True)
    if "--portal" in sys.argv:
        return models.model_for(False)

    options = []
    for m in models.MODELS:
        saved = "trained table found" if models.has_saved_table(m) else "no saved table yet (will train from scratch)"
        options.append({
            **m,
            "lines": [m["blurb"], f"File: {m['file']}  —  {saved}"],
        })
    return _run_choice_menu(
        screen, clock,
        "Project03_RL — PygAIme Racer",
        "Step 1/2 — Choose a model",
        options,
    )


def choose_play_mode(screen, clock, model):
    """Return a PLAY_MODES entry: greedy demo vs learning (or from --greedy / --train)."""
    if "--greedy" in sys.argv:
        return PLAY_MODES[0]
    if "--train" in sys.argv:
        return PLAY_MODES[1]

    options = [{**mode, "lines": [mode["blurb"]]} for mode in PLAY_MODES]
    # Default highlight: greedy if a table exists (safer for demos), else train.
    default = 0 if models.has_saved_table(model) else 1
    return _run_choice_menu(
        screen, clock,
        "Project03_RL — PygAIme Racer",
        f"Step 2/2 — {model['label']}: choose play mode",
        options,
        default=default,
    )


def _fresh_info(env, agent):
    limits = scoring.max_possible_score(env)
    return {
        "max_score": limits["score"], "max_score_luck": limits["score_with_luck"],
        "episode": 1, "epsilon": agent.epsilon, "wins": 0, "crashes": 0,
        "best_time": "-", "last": "-", "fast": False, "greedy": agent.greedy,
        "portals": 0, "show_portal": False, "show_hud": True, "random_starts": True,
        "boxes_total": 0, "best_points": "-", "best_box_points": 0,
        "box_jackpot": env.BOX_POINTS,
        "routes": {"Scenic": 0, "Shortcut": 0},
        "model_label": ("No portals" if not env.portals_enabled else "With portals"),
    }


def load_model(model, agent=None):
    """Build env + agent for this model and load its Q-table if present."""
    env = RacingEnv(portals=not model["no_portal"])
    if agent is None:
        agent = QLearningAgent(env.N_ACTIONS, **settings.AGENT_KWARGS)
    else:
        # Switching models: replace the table rather than merging two different games.
        agent.q = {}
        agent.epsilon = settings.AGENT_KWARGS.get("epsilon", 1.0)
        agent.greedy = False
    q_file = model["file"]
    if os.path.exists(q_file):
        agent.load(q_file)
        print("Loaded saved Q-table:", q_file)
    else:
        print("No saved Q-table yet for", model["label"], f"({q_file}) — starting fresh.")
    return env, agent, q_file


def main():
    """Open the game window, pick model + play mode, then train/watch until ESC."""
    pygame.init()
    screen = pygame.display.set_mode((renderer.W, renderer.H))
    clock = pygame.time.Clock()

    model = choose_model(screen, clock)
    if model is None:
        pygame.quit()
        return

    play_mode = choose_play_mode(screen, clock, model)
    if play_mode is None:
        pygame.quit()
        return

    pygame.display.set_caption(f"Project03_RL - {model['label']} - {play_mode['label']}")
    font = pygame.font.SysFont("consolas", 18)

    env, agent, q_file = load_model(model)
    agent.greedy = play_mode["greedy"]
    info = _fresh_info(env, agent)
    # Greedy demos always start on the start line; training may use random practice starts.
    info["random_starts"] = not agent.greedy
    # If this session never learns, quitting will not overwrite the .pkl (safer for demos).
    session_writes_table = not play_mode["greedy"]
    best = None
    best_points = None

    def new_episode():
        # Practice starts (random y) teach late track sections; only from_start races count in HUD stats.
        practice = info["random_starts"] and not agent.greedy and random.random() < settings.RANDOM_START_PROB
        return env.reset(random_start=practice)

    def switch_model():
        """Save current table (if this session can write), flip model, load the other table."""
        nonlocal env, agent, q_file, model, info, best, best_points, state, session_writes_table
        if session_writes_table:
            agent.save(q_file)
            print("Saved", q_file)
        idx = 0 if model["key"] == "noportal" else 1
        model = models.MODELS[idx]
        was_greedy = agent.greedy
        env, agent, q_file = load_model(model, agent=agent)
        agent.greedy = was_greedy
        info = _fresh_info(env, agent)
        info["random_starts"] = not agent.greedy
        session_writes_table = not agent.greedy
        best = None
        best_points = None
        state = new_episode()
        pygame.display.set_caption(f"Project03_RL - {model['label']} - {'Greedy' if agent.greedy else 'Train'}")
        print("Switched to", model["label"], "| greedy =", agent.greedy)

    state = new_episode()
    running = True
    frame = 0
    print(f"Starting: {model['label']} | {play_mode['label']}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    info["fast"] = not info["fast"]
                elif event.key == pygame.K_g:
                    agent.greedy = not agent.greedy
                    info["greedy"] = agent.greedy
                    if agent.greedy:
                        info["random_starts"] = False
                    else:
                        session_writes_table = True  # user turned learning back on
                elif event.key == pygame.K_r:
                    info["random_starts"] = not info["random_starts"]
                elif event.key == pygame.K_h:
                    info["show_hud"] = not info["show_hud"]
                elif event.key == pygame.K_p:
                    info["show_portal"] = not info["show_portal"]
                elif event.key == pygame.K_m:
                    switch_model()
                elif event.key == pygame.K_s:
                    agent.save(q_file)
                    print("Saved", q_file)

        # Watch mode: 1 step per frame. Fast mode: as many steps as fit in ~30 ms.
        deadline = time.perf_counter() + 0.03
        while True:
            in_fork = env.track.FORK_ZONE[0] <= env.y <= env.track.FORK_ZONE[1]
            action = agent.choose_action(state, settings.FORK_EPSILON if in_fork else None)
            boxes_before = env.boxes_opened
            next_state, reward, done, step_info = env.step(action)
            info["boxes_total"] += env.boxes_opened - boxes_before
            agent.learn(state, action, reward, next_state, done)
            state = next_state

            if done:
                if step_info["from_start"]:
                    if step_info["portal"]:
                        info["portals"] += 1
                    if step_info["route"]:
                        info["routes"][step_info["route"]] += 1
                    if step_info["finished"]:
                        info["wins"] += 1
                        t = round(step_info["time"], 1)
                        pts = step_info["score"]
                        info["last"] = f"FINISH {t}s  {pts} pts"
                        if best is None or t < best:
                            best = t
                            info["best_time"] = f"{best}s"
                        if best_points is None or pts > best_points:
                            best_points = pts
                            info["best_points"] = str(best_points)
                            # Jackpot points on that best run (for a fair % vs skill max + those jackpots).
                            info["best_box_points"] = int(step_info.get("box_points", 0) or 0)
                    elif step_info["crashed"]:
                        info["crashes"] += 1
                        info["last"] = f"CRASH ({step_info['crash_reason']})"
                    else:
                        info["last"] = "TIME UP"
                if not agent.greedy:
                    agent.decay_epsilon()
                info["episode"] += 1
                state = new_episode()

            if not info["fast"] or time.perf_counter() > deadline:
                break

        info["epsilon"] = agent.epsilon
        info["model_label"] = model["label"]
        frame += 1
        if not info["fast"] or frame % 5 == 0:
            renderer.draw(screen, font, env, info)
        clock.tick(30 if not info["fast"] else 60)

    if session_writes_table:
        agent.save(q_file)
        print("Saved", q_file)
    else:
        print("Greedy demo ended — Q-table not overwritten.")
    pygame.quit()


if __name__ == "__main__":
    main()
