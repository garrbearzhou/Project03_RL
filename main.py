"""
main.py - runs the training loop: Environment <-> Agent <-> Renderer.

AI USE DISCLOSURE (Honor Code): Generated with help from Claude (Anthropic).
<-- Add your own notes here.

Run with --no-portal for the version of the game without portals (it has its own saved Q-table).
Controls:  SPACE = toggle fast training / watch mode,  G = greedy (no exploring, always starts
           at the start line),  R = random practice starts on/off,  P = reveal portal,  H = hide the HUD,
           S = save Q-table,  ESC = quit (auto-saves)
"""
import os
import random
import sys
import time
import pygame

from environment import RacingEnv
from agent import QLearningAgent
import renderer
import settings


def main():
    no_portal = "--no-portal" in sys.argv                 # python main.py --no-portal
    q_file = settings.Q_FILE_NO_PORTAL if no_portal else settings.Q_FILE
    pygame.init()
    screen = pygame.display.set_mode((renderer.W, renderer.H))
    pygame.display.set_caption("Project03_RL - PygAIme Racer" + (" (no portals)" if no_portal else ""))
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    env = RacingEnv(portals=not no_portal)
    agent = QLearningAgent(env.N_ACTIONS, **settings.AGENT_KWARGS)
    if os.path.exists(q_file):
        agent.load(q_file)
        print("Loaded saved Q-table:", q_file)

    info = {"episode": 1, "epsilon": agent.epsilon, "wins": 0, "crashes": 0,
            "best_time": "-", "last": "-", "fast": False, "greedy": False,
            "portals": 0, "show_portal": False, "show_hud": True, "random_starts": True, "boxes_total": 0,
            "best_points": "-", "routes": {"Scenic": 0, "Shortcut": 0}}
    best = None
    best_points = None

    def new_episode():
        practice = info["random_starts"] and not agent.greedy and random.random() < settings.RANDOM_START_PROB
        return env.reset(random_start=practice)

    state = new_episode()
    running = True
    frame = 0

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
                elif event.key == pygame.K_r:
                    info["random_starts"] = not info["random_starts"]
                elif event.key == pygame.K_h:
                    info["show_hud"] = not info["show_hud"]       # hide / show the HUD
                elif event.key == pygame.K_p:
                    info["show_portal"] = not info["show_portal"]   # reveal the hidden portal
                elif event.key == pygame.K_s:
                    agent.save(q_file)

        # Watch mode: 1 step per frame. Fast mode: as many steps as fit in ~30 ms.
        deadline = time.perf_counter() + 0.03
        while True:
            action = agent.choose_action(state)
            boxes_before = env.boxes_opened
            next_state, reward, done, step_info = env.step(action)
            info["boxes_total"] += env.boxes_opened - boxes_before      # mystery boxes opened (all runs)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

            if done:
                if step_info["from_start"]:                  # only real start-line races count in the stats
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
                    elif step_info["crashed"]:
                        info["crashes"] += 1
                        info["last"] = f"CRASH ({step_info['crash_reason']})"
                    else:
                        info["last"] = "TIME UP"
                agent.decay_epsilon()
                info["episode"] += 1
                state = new_episode()

            if not info["fast"] or time.perf_counter() > deadline:
                break

        info["epsilon"] = agent.epsilon
        frame += 1
        if not info["fast"] or frame % 5 == 0:      # fast training only draws every 5th loop (saves time)
            renderer.draw(screen, font, env, info)
        clock.tick(30 if not info["fast"] else 60)

    agent.save(q_file)
    pygame.quit()


if __name__ == "__main__":
    main()