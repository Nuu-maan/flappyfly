import multiprocessing as mp
import os
import threading

import numpy as np

from .game import Flappy, flap_margin
from .readout import MODEL, Readout
from .sim import LIF
from .train import STEPS_PER_FRAME, select_features
from .vision import setup

N_BIRDS = max(2, min(8, os.cpu_count() or 2))
SIGMA = 0.02
ELITES = 3
BUFFER = 12000
REFIT_EVERY = 1500
FROM_RIDGE = 0.5


def worker(brain, retina, feat_idx, conn):
    sim, readout = LIF(brain.W), Readout(feat_idx)
    while True:
        img = conn.recv()
        if img is None:
            return
        retina.see(sim, img)
        counts = sim.run(STEPS_PER_FRAME)
        conn.send((readout.features(sim, counts), counts[feat_idx]))


class Population:
    """Birds share one world. Each has its own brain process and its own readout genome.
    Dead birds respawn from a mutated elite or from a ridge readout refit on everything
    every bird has seen, labelled by the teacher bot (DAgger)."""

    def __init__(self, n_birds=N_BIRDS, seed=0):
        self.brain, self.retina = setup()
        self.rng = np.random.default_rng(seed)
        self.base = Readout.load() if MODEL.exists() else Readout(select_features(self.brain.W, self.retina))
        self.normalized = MODEL.exists()
        self.game = Flappy(seed, n_birds)
        self.genomes = [self.mutate((self.base.w, self.base.b)) for _ in range(n_birds)]
        self.elites = [((self.base.w, self.base.b), 0)]
        self.best_ever, self.deaths, self.history, self.refits = 0, 0, [], 0
        self.X, self.y = [], []
        self.fit_thread = None

        ctx = mp.get_context("fork")
        self.conns = []
        for _ in range(n_birds):
            here, there = ctx.Pipe()
            ctx.Process(target=worker, args=(self.brain, self.retina, self.base.feat_idx, there), daemon=True).start()
            self.conns.append(here)

    @property
    def scale(self):
        return float(np.abs(self.base.w).mean()) or 1.0

    def mutate(self, genome):
        w, b = genome
        return (w + self.rng.normal(0, SIGMA * self.scale, w.shape).astype(np.float32),
                b + self.rng.normal(0, SIGMA * self.scale * 10))

    def step(self):
        for i, conn in enumerate(self.conns):
            conn.send(self.game.render(i))
        feats, self.counts = zip(*(conn.recv() for conn in self.conns))
        flaps = [bool(self.base.score(f, w, b) > 0) for f, (w, b) in zip(feats, self.genomes)]
        for i, f in enumerate(feats):
            if self.game.birds[i].alive:
                self.X.append(f)
                self.y.append(flap_margin(self.game.state(i)))
        alive = self.game.step(flaps)
        for i, ok in enumerate(alive):
            if not ok:
                self.on_death(i)
        if len(self.X) > BUFFER:
            del self.X[: len(self.X) - BUFFER], self.y[: len(self.y) - BUFFER]
        if self.game.frames % REFIT_EVERY == 0 and self.fit_thread is None:
            self.fit_thread = threading.Thread(target=self.refit, args=(np.stack(self.X), np.array(self.y)), daemon=True)
            self.fit_thread.start()
        return alive

    def refit(self, X, y):
        """Genomes live in the base's normalized feature space, so mean/std are frozen after the first fit."""
        norm = (self.base.mean, self.base.std) if self.normalized else (None, None)
        ro = Readout(self.base.feat_idx).fit(X, y, mean=norm[0], std=norm[1])
        self.base.mean, self.base.std, self.base.w, self.base.b = ro.mean, ro.std, ro.w, ro.b
        self.normalized = True
        self.refits += 1
        self.fit_thread = None

    def on_death(self, i):
        fitness = self.game.birds[i].frames
        self.deaths += 1
        self.history.append(fitness)
        self.best_ever = max(self.best_ever, fitness)
        self.elites = sorted(self.elites + [(self.genomes[i], fitness)], key=lambda e: -e[1])[:ELITES]
        if self.refits and self.rng.random() < FROM_RIDGE:
            self.genomes[i] = self.mutate((self.base.w, self.base.b))
        else:
            self.genomes[i] = self.mutate(self.elites[self.rng.integers(len(self.elites))][0])
        self.game.spawn(i)

    @property
    def fitting(self):
        return self.fit_thread is not None

    def save_best(self, path=MODEL):
        (w, b), _ = self.elites[0]
        Readout(self.base.feat_idx, w, b, self.base.mean, self.base.std).save(path)

    def close(self):
        for conn in self.conns:
            conn.send(None)
