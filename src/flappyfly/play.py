import numpy as np
import pygame

from .brain import load
from .game import BIRD_R, BIRD_X, GAP, H, PIPE_W, W, Flappy
from .readout import Readout
from .sim import LIF
from .train import frame
from .vision import Retina

RASTER_W, RASTER_NEURONS = 400, 300
FPS = 30
KNOCKOUT_TYPE = "Tm1"


def draw_game(surf, game):
    surf.fill((78, 192, 202))
    for px, gy in game.pipes:
        pygame.draw.rect(surf, (85, 170, 60), (px, 0, PIPE_W, gy - GAP / 2))
        pygame.draw.rect(surf, (85, 170, 60), (px, gy + GAP / 2, PIPE_W, H))
    pygame.draw.circle(surf, (240, 200, 40), (BIRD_X, int(game.y)), BIRD_R)


def play():
    brain = load()
    retina = Retina(brain)
    readout = Readout.load()
    sim = LIF(brain.W)
    game = Flappy()
    rng = np.random.default_rng(0)
    shown = rng.choice(readout.feat_idx, RASTER_NEURONS, replace=False)
    knockout = None

    pygame.init()
    screen = pygame.display.set_mode((W + RASTER_W, H))
    pygame.display.set_caption("flappyfly")
    font = pygame.font.SysFont(None, 28)
    raster = pygame.Surface((RASTER_W, H))
    raster.fill((20, 20, 30))
    clock = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_k:
                knockout = None if knockout is not None else brain.neurons_of(KNOCKOUT_TYPE)
                sim.W = LIF(brain.W).W
                if knockout is not None:
                    rows = np.zeros(brain.n, dtype=bool)
                    rows[knockout] = True
                    sim.W.data[np.repeat(rows, np.diff(sim.W.indptr))] = 0
        counts = frame(sim, retina, game)
        _, alive = game.step(readout.act(counts))
        if not alive:
            game.reset()

        raster.scroll(dx=-2)
        pygame.draw.rect(raster, (20, 20, 30), (RASTER_W - 2, 0, 2, H))
        for i, neuron in enumerate(shown):
            if counts[neuron]:
                raster.set_at((RASTER_W - 1, int(i * H / RASTER_NEURONS)), (255, 230, 120))

        draw_game(screen, game)
        screen.blit(raster, (W, 0))
        label = f"score {game.score}   spikes/frame {counts.sum()}" + (f"   {KNOCKOUT_TYPE} KNOCKED OUT" if knockout is not None else "")
        screen.blit(font.render(label, True, (255, 255, 255)), (10, 10))
        pygame.display.flip()
        clock.tick(FPS)
