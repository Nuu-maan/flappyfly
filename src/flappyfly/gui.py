import numpy as np
import pygame

from .evolve import Population
from .game import BIRD_R, BIRD_X, GAP, H, PIPE_W, W

PANEL_W, RASTER_H, RASTER_NEURONS = 420, 260, 200
FPS = 30
SKY, PIPE, TEXT, DIM, INK = (78, 192, 202), (85, 170, 60), (245, 245, 245), (150, 150, 160), (22, 22, 30)


def bird_color(i):
    return pygame.Color("#ffd23f").lerp(pygame.Color("#ff3f8f"), (i * 0.37) % 1.0)


def draw_game(surf, game):
    surf.fill(SKY)
    for px, gy in game.pipes:
        pygame.draw.rect(surf, PIPE, (px, 0, PIPE_W, gy - GAP / 2))
        pygame.draw.rect(surf, PIPE, (px, gy + GAP / 2, PIPE_W, H))
    for i, b in enumerate(game.birds):
        if b.alive:
            pygame.draw.circle(surf, bird_color(i), (BIRD_X, int(b.y)), BIRD_R)
            pygame.draw.circle(surf, INK, (BIRD_X, int(b.y)), BIRD_R, 2)


def draw_panel(surf, pop, font, small, raster):
    surf.fill(INK)
    lines = [
        f"birds alive   {sum(b.alive for b in pop.game.birds)}/{len(pop.game.birds)}",
        f"frame         {pop.game.frames}",
        f"deaths        {pop.deaths}",
        f"best ever     {pop.best_ever} frames",
        f"elite         {pop.elites[0][1]} frames",
    ]
    for k, line in enumerate(lines):
        surf.blit(font.render(line, True, TEXT), (16, 16 + 28 * k))
    for i, b in enumerate(pop.game.birds):
        y = 170 + 20 * i
        pygame.draw.circle(surf, bird_color(i), (24, y + 8), 7)
        label = f"{b.frames:5d} frames  score {b.score}" if b.alive else "dead"
        surf.blit(small.render(label, True, TEXT if b.alive else DIM), (40, y))

    if pop.history:
        hist = np.array(pop.history[-120:])
        x0, y0, w, h = 16, H - RASTER_H - 90, PANEL_W - 32, 70
        pygame.draw.rect(surf, (40, 40, 52), (x0, y0, w, h), 1)
        pts = [(x0 + k * w / max(1, len(hist) - 1), y0 + h - h * v / max(1, hist.max())) for k, v in enumerate(hist)]
        if len(pts) > 1:
            pygame.draw.lines(surf, (120, 220, 160), False, pts, 2)
        surf.blit(small.render("fitness per death", True, DIM), (x0, y0 - 18))

    surf.blit(raster, (0, H - RASTER_H))
    surf.blit(small.render("spikes, best living bird", True, DIM), (16, H - RASTER_H - 18))


def play(n_birds=None):
    pop = Population(**({} if n_birds is None else {"n_birds": n_birds}))
    shown = np.random.default_rng(0).choice(len(pop.base.feat_idx), RASTER_NEURONS, replace=False)

    pygame.init()
    screen = pygame.display.set_mode((W + PANEL_W, H))
    pygame.display.set_caption("flappyfly - evolving")
    font, small = pygame.font.SysFont("monospace", 20), pygame.font.SysFont("monospace", 15)
    raster = pygame.Surface((PANEL_W, RASTER_H))
    raster.fill(INK)
    game, panel = pygame.Surface((W, H)), pygame.Surface((PANEL_W, H))
    clock = pygame.time.Clock()

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                    pop.save_best()
            pop.step()
            best = max((i for i, b in enumerate(pop.game.birds) if b.alive), key=lambda i: pop.game.birds[i].frames, default=0)
            raster.scroll(dx=-2)
            pygame.draw.rect(raster, INK, (PANEL_W - 2, 0, 2, RASTER_H))
            for k, j in enumerate(shown):
                if pop.counts[best][j]:
                    raster.set_at((PANEL_W - 1, int(k * RASTER_H / RASTER_NEURONS)), bird_color(best))
            draw_game(game, pop.game)
            draw_panel(panel, pop, font, small, raster)
            screen.blit(game, (0, 0))
            screen.blit(panel, (W, 0))
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pop.save_best()
        pop.close()
