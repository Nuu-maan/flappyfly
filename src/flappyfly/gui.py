import time

import numpy as np
import pygame

from .evolve import Population
from .game import BIRD_R, BIRD_X, GAP, H, PIPE_W, W

WIN_W, WIN_H = 1280, 720
GAME_W = int(W * WIN_H / H)
PANEL_X = GAME_W + 16
FPS = 30

INK = (14, 15, 22)
CARD = (24, 26, 36)
LINE = (44, 47, 62)
TEXT = (232, 234, 240)
DIM = (128, 132, 150)
ACCENT = (108, 220, 170)
WARN = (255, 120, 90)
BIRD_PALETTE = ["#ffd23f", "#ff6b6b", "#4ecdc4", "#c084fc", "#fb923c", "#38bdf8", "#a3e635", "#f472b6"]


def bird_color(i):
    return pygame.Color(BIRD_PALETTE[i % len(BIRD_PALETTE)])


def gradient(size, top, bottom):
    surf = pygame.Surface(size)
    h = size[1]
    for y in range(h):
        t = y / max(1, h - 1)
        surf.fill([int(a + (b - a) * t) for a, b in zip(top, bottom)], (0, y, size[0], 1))
    return surf


class GameView:
    def __init__(self):
        self.sky = gradient((W, H), (48, 120, 190), (140, 205, 230))
        self.canvas = pygame.Surface((W, H))
        self.ghosts = []
        self.scroll = 0

    def bird(self, surf, x, y, vy, color, wing_up):
        pygame.draw.circle(surf, INK, (x, y), BIRD_R + 2)
        pygame.draw.circle(surf, color, (x, y), BIRD_R)
        wing = pygame.Rect(0, 0, 14, 8)
        wing.center = (x - 3, y + (-6 if wing_up else 4))
        pygame.draw.ellipse(surf, color.lerp(INK, 0.35), wing)
        pygame.draw.circle(surf, (255, 255, 255), (x + 5, y - 4), 4)
        pygame.draw.circle(surf, INK, (x + 6, y - 4), 2)
        beak = [(x + 10, y), (x + 17, y + 2 + int(vy * 0.3)), (x + 10, y + 5)]
        pygame.draw.polygon(surf, (255, 150, 60), beak)

    def pipe(self, surf, px, gy):
        top, bottom = gy - GAP / 2, gy + GAP / 2
        for rect in (pygame.Rect(px, 0, PIPE_W, top), pygame.Rect(px, bottom, PIPE_W, H - bottom)):
            pygame.draw.rect(surf, (70, 150, 55), rect)
            pygame.draw.rect(surf, (120, 200, 90), (rect.x + 6, rect.y, 10, rect.h))
            pygame.draw.rect(surf, (40, 95, 35), rect, 2)
        for cy in (top - 22, bottom):
            cap = pygame.Rect(px - 4, cy, PIPE_W + 8, 22)
            pygame.draw.rect(surf, (85, 170, 65), cap)
            pygame.draw.rect(surf, (40, 95, 35), cap, 2)

    def draw(self, screen, game, deaths):
        c = self.canvas
        c.blit(self.sky, (0, 0))
        self.scroll = (self.scroll + 3) % 40
        for k in range(-1, W // 40 + 1):
            pygame.draw.rect(c, (60, 60, 80) if k % 2 else (80, 80, 100), (k * 40 - self.scroll, H - 14, 40, 14))
        pygame.draw.line(c, (60, 160, 80), (0, H - 14), (W, H - 14), 3)
        for px, gy in game.pipes:
            self.pipe(c, px, gy)
        for i, b in enumerate(game.birds):
            if b.alive:
                self.bird(c, BIRD_X, int(b.y), b.vy, bird_color(i), b.vy < 0)
        for i, y in deaths:
            self.ghosts.append([BIRD_X, int(np.clip(y, 0, H - 1)), bird_color(i), 18])
        for g in self.ghosts:
            r = BIRD_R + (18 - g[3])
            pygame.draw.circle(c, g[2].lerp(self.sky.get_at((0, min(H - 1, g[1]))), 1 - g[3] / 18), (g[0], g[1]), r, 2)
            g[3] -= 1
        self.ghosts = [g for g in self.ghosts if g[3] > 0]
        screen.blit(pygame.transform.smoothscale(c, (GAME_W, WIN_H)), (0, 0))


class Panel:
    def __init__(self, brain, feat_idx, pop=None):
        self.pop = pop
        self.brain, self.feat_idx = brain, feat_idx
        self.big = pygame.font.Font(None, 30)
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 17)
        self.t0 = time.time()
        feat = feat_idx
        self.eyes = []
        for eye in ("L", "R"):
            sel = np.flatnonzero((brain.side[feat] == eye) & np.isfinite(brain.hex[feat]).all(axis=1))
            hx = brain.hex[feat[sel]]
            xy = np.stack([hx[:, 0] - hx[:, 1] / 2, hx[:, 1] * 0.866], axis=1)
            xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-6)
            self.eyes.append((sel, xy))
        types = brain.type[feat]
        self.raster_idx = np.concatenate([np.flatnonzero(types == t)[:24] for t in np.unique(types)])
        self.raster_types = types[self.raster_idx]
        self.raster = pygame.Surface((0, 0))
        self.glow = [np.zeros(len(sel), dtype=np.float32) for sel, _ in self.eyes]

    def text(self, surf, s, pos, font=None, color=TEXT):
        surf.blit((font or self.font).render(s, True, color), pos)

    def card(self, surf, rect, title=None):
        pygame.draw.rect(surf, CARD, rect, border_radius=8)
        pygame.draw.rect(surf, LINE, rect, 1, border_radius=8)
        if title:
            self.text(surf, title.upper(), (rect.x + 10, rect.y + 7), self.small, DIM)

    def tile(self, surf, rect, label, value, color=TEXT):
        self.card(surf, rect)
        self.text(surf, label, (rect.x + 10, rect.y + 8), self.small, DIM)
        self.text(surf, str(value), (rect.x + 10, rect.y + 26), self.big, color)

    def eye(self, surf, rect, k, counts, title):
        self.card(surf, rect, title)
        sel, xy = self.eyes[k]
        self.glow[k] = self.glow[k] * 0.6 + counts[sel]
        r = pygame.Rect(rect.x + 12, rect.y + 28, rect.w - 24, rect.h - 40)
        pts = np.column_stack([r.x + xy[:, 0] * r.w, r.bottom - xy[:, 1] * r.h]).astype(int)
        g = np.clip(self.glow[k] / 2.5, 0, 1)
        for (x, y), v in zip(pts, g):
            col = (int(30 + 200 * v), int(40 + 180 * v), int(60 + 90 * v)) if v > 0.02 else (34, 37, 50)
            surf.fill(col, (x, y, 3, 3))

    def retina(self, surf, rect, img):
        self.card(surf, rect, "what the best fly sees")
        cell = min((rect.h - 44) // 16, (rect.w - 40) // 32)
        for c, (label, tint) in enumerate((("left eye: pipe", (120, 200, 90)), ("right eye: bird", (255, 210, 63)))):
            ox = rect.x + 12 + c * (16 * cell + 16)
            self.text(surf, label, (ox, rect.y + 26), self.small, DIM)
            for yy in range(16):
                for xx in range(16):
                    v = img[c, yy, xx]
                    col = tuple(int(t * v + 30 * (1 - v)) for t in tint) if v else (30, 32, 42)
                    surf.fill(col, (ox + xx * cell, rect.y + 42 + yy * cell, cell - 1, cell - 1))

    def chart(self, surf, rect):
        self.card(surf, rect, "survival per death (frames)")
        hist = self.pop.history[-200:]
        if len(hist) < 2:
            return
        r = pygame.Rect(rect.x + 44, rect.y + 30, rect.w - 56, rect.h - 44)
        top = max(60, max(hist))
        for k in range(4):
            y = r.bottom - r.h * k / 3
            pygame.draw.line(surf, LINE, (r.x, y), (r.right, y))
            self.text(surf, f"{int(top * k / 3)}", (rect.x + 8, y - 7), self.small, DIM)
        pts = [(r.x + k * r.w / (len(hist) - 1), r.bottom - r.h * v / top) for k, v in enumerate(hist)]
        pygame.draw.lines(surf, (70, 80, 110), False, pts, 1)
        win = 15
        if len(hist) > win:
            sm = np.convolve(hist, np.ones(win) / win, mode="valid")
            pts = [(r.x + (k + win - 1) * r.w / (len(hist) - 1), r.bottom - r.h * v / top) for k, v in enumerate(sm)]
            pygame.draw.lines(surf, ACCENT, False, pts, 2)

    def roster(self, surf, rect):
        self.card(surf, rect, "birds")
        best = max(1, self.pop.best_ever)
        for i, b in enumerate(self.pop.game.birds):
            y = rect.y + 30 + i * 20
            pygame.draw.circle(surf, bird_color(i), (rect.x + 20, y + 8), 6)
            if b.alive:
                self.text(surf, f"{b.frames:5d}", (rect.x + 34, y + 1), self.font)
                self.text(surf, f"{b.score} pipes", (rect.x + 88, y + 3), self.small, DIM)
                bar = pygame.Rect(rect.x + 160, y + 4, rect.w - 232, 9)
                pygame.draw.rect(surf, LINE, bar, border_radius=4)
                pygame.draw.rect(surf, bird_color(i), (bar.x, bar.y, int(bar.w * min(1, b.frames / best)), bar.h), border_radius=4)
                self.text(surf, self.pop.origin[i], (rect.right - 62, y + 3), self.small, DIM)
            else:
                self.text(surf, "respawning", (rect.x + 34, y + 3), self.small, WARN)

    def spikes(self, surf, rect, counts):
        self.card(surf, rect, f"spikes · {int(counts.sum())} / frame")
        r = pygame.Rect(rect.x + 70, rect.y + 28, rect.w - 82, rect.h - 40)
        if self.raster.get_size() != (r.w, r.h):
            self.raster = pygame.Surface((r.w, r.h))
            self.raster.fill((18, 19, 27))
        self.raster.scroll(dx=-2)
        self.raster.fill((18, 19, 27), (r.w - 2, 0, 2, r.h))
        rows = counts[self.raster_idx]
        for k in np.flatnonzero(rows):
            y = int(k * r.h / len(rows))
            self.raster.fill((255, 230, 120) if rows[k] < 2 else (255, 140, 90), (r.w - 2, y, 2, 2))
        surf.blit(self.raster, r.topleft)
        types = self.raster_types
        starts = np.flatnonzero(np.r_[True, types[1:] != types[:-1]])
        last_y = -99
        for s, e in zip(starts, np.r_[starts[1:], len(types)]):
            y = r.y + int(s * r.h / len(types))
            if e - s >= 12 and y - last_y >= 14:
                self.text(surf, types[s], (rect.x + 10, y), self.small, DIM)
                last_y = y

    def header(self, surf, x, frames, fps):
        self.text(surf, "flappyfly", (x, 12), self.big, ACCENT)
        self.text(surf, f"male Drosophila CNS connectome  ·  {self.brain.n:,} neurons simulated  ·  {self.brain.W.nnz:,} synapses", (x + 118, 18), self.small, DIM)
        elapsed = int(time.time() - self.t0)
        self.text(surf, f"{elapsed // 60:02d}:{elapsed % 60:02d}   frame {frames}   {fps:4.1f} fps", (x, 40), self.small, DIM)

    def tiles(self, surf, x, w, tiles):
        tw = (w - 8 * (len(tiles) - 1)) // len(tiles)
        for k, (label, value, color) in enumerate(tiles):
            self.tile(surf, pygame.Rect(x + k * (tw + 8), 62, tw, 58), label, value, color)

    def senses(self, surf, x, w, counts, img):
        row_y, row_h = 132, 220
        third = (w - 16) // 3
        self.eye(surf, pygame.Rect(x, row_y, third, row_h), 0, counts, "left eye · lamina activity")
        self.retina(surf, pygame.Rect(x + third + 8, row_y, third, row_h), img)
        self.eye(surf, pygame.Rect(x + 2 * (third + 8), row_y, third, row_h), 1, counts, "right eye · lamina activity")

    def draw(self, surf, counts, img, fps):
        pop = self.pop
        x, w = PANEL_X, WIN_W - PANEL_X - 12
        self.header(surf, x, pop.game.frames, fps)
        self.tiles(surf, x, w, [("alive", f"{sum(b.alive for b in pop.game.birds)}/{len(pop.game.birds)}", TEXT),
                 ("best ever", f"{pop.best_ever}", ACCENT),
                 ("elite", f"{pop.elites[0][1]}", TEXT),
                 ("deaths", f"{pop.deaths}", TEXT),
                 ("ridge refits", f"{pop.refits}" + (" ⟳" if pop.fitting else ""), WARN if pop.fitting else TEXT)])
        self.senses(surf, x, w, counts, img)

        row_y, row_h = 364, 150
        half = (w - 8) // 2
        self.chart(surf, pygame.Rect(x, row_y, half, row_h))
        self.roster(surf, pygame.Rect(x + half + 8, row_y, half, row_h + 46))
        self.spikes(surf, pygame.Rect(x, row_y + row_h + 8, half, WIN_H - row_y - row_h - 20), counts)
        self.text(surf, "s: save best genome   q: quit", (x + half + 8, WIN_H - 26), self.small, DIM)


def play(n_birds=None):
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("flappyfly - evolving")
    loading = pygame.font.Font(None, 30).render("loading the fly brain...", True, DIM)
    screen.fill(INK)
    screen.blit(loading, (WIN_W // 2 - 110, WIN_H // 2))
    pygame.display.flip()

    pop = Population(**({} if n_birds is None else {"n_birds": n_birds}))
    view, panel = GameView(), Panel(pop.brain, pop.base.feat_idx, pop)
    clock = pygame.time.Clock()
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                    pop.save_best()
            pop.step()
            best = max(range(len(pop.game.birds)), key=lambda i: (pop.game.birds[i].alive, pop.game.birds[i].frames))
            screen.fill(INK)
            view.draw(screen, pop.game, pop.last_deaths)
            panel.draw(screen, pop.counts[best], pop.game.render(best), clock.get_fps())
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pop.save_best()
        pop.close()
