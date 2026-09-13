import pygame

from .game import Flappy
from .gui import ACCENT, DIM, FPS, INK, PANEL_X, TEXT, WIN_H, WIN_W, GameView, Panel
from .readout import MODEL, Readout
from .sim import LIF
from .train import frame
from .vision import setup


def showcase():
    """One fly, the best genome trained so far, restarting whenever it dies."""
    if not MODEL.exists():
        raise SystemExit("no trained readout yet: run `flappyfly play` first")
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("flappyfly - best fly")
    screen.fill(INK)
    screen.blit(pygame.font.Font(None, 30).render("loading the fly brain...", True, DIM), (WIN_W // 2 - 110, WIN_H // 2))
    pygame.display.flip()

    brain, retina = setup()
    readout = Readout.load()
    sim, game = LIF(brain.W), Flappy(seed=int(pygame.time.get_ticks()))
    view, panel = GameView(), Panel(brain, readout.feat_idx)
    clock = pygame.time.Clock()
    runs, best_score, scores = 1, 0, []
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                return
        counts = frame(sim, retina, game)
        deaths = []
        if not game.step(readout.act(sim, counts)):
            deaths = [(0, game.bird.y)]
            scores.append(game.bird.score)
            best_score = max(best_score, game.bird.score)
            runs += 1
            game.reset()
        screen.fill(INK)
        view.draw(screen, game, deaths)
        x, w = PANEL_X, WIN_W - PANEL_X - 12
        panel.header(screen, x, game.frames, clock.get_fps())
        panel.tiles(screen, x, w, [
            ("score", game.bird.score, ACCENT),
            ("best score", best_score, TEXT),
            ("runs", runs, TEXT),
            ("genome fitness", f"{readout.fitness} frames", TEXT),
            ("mean score", f"{sum(scores) / len(scores):.1f}" if scores else "-", TEXT)])
        panel.senses(screen, x, w, counts[readout.feat_idx], game.render())
        panel.spikes(screen, pygame.Rect(x, 364, w, WIN_H - 364 - 34), counts[readout.feat_idx])
        panel.text(screen, f"model: {MODEL}   q: quit", (x, WIN_H - 26), panel.small, DIM)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    showcase()
