import multiprocessing as mp
import os

import numpy as np

from .game import Flappy
from .readout import MODEL, Readout
from .sim import LIF
from .train import STEPS_PER_FRAME, select_features
from .vision import setup

N_BIRDS = max(2, min(8, os.cpu_count() or 2))
SIGMA = 0.05
ELITES = 3


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
    def __init__(self, n_birds=N_BIRDS, seed=0):
        self.brain, self.retina = setup()
        self.rng = np.random.default_rng(seed)
        base = Readout.load() if MODEL.exists() else Readout(select_features(self.brain.W, self.retina))
        self.base = base
        self.scale = np.abs(base.w).mean() or 1.0
        self.game = Flappy(seed, n_birds)
        self.genomes = [self.mutate((base.w, base.b)) for _ in range(n_birds)]
        self.elites = [((base.w, base.b), 0)]
        self.best_ever, self.deaths, self.history = 0, 0, []

        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n_birds):
            here, there = ctx.Pipe()
            p = ctx.Process(target=worker, args=(self.brain, self.retina, base.feat_idx, there), daemon=True)
            p.start()
            self.conns.append(here)
            self.procs.append(p)

    def mutate(self, genome):
        w, b = genome
        return (w + self.rng.normal(0, SIGMA * self.scale, w.shape).astype(np.float32),
                b + self.rng.normal(0, SIGMA * self.scale * 10))


    def step(self):
        for i, conn in enumerate(self.conns):
            conn.send(self.game.render(i))
        feats, self.counts = zip(*(conn.recv() for conn in self.conns))
        flaps = [bool(self.base.score(f, w, b) > 0) for f, (w, b) in zip(feats, self.genomes)]
        alive = self.game.step(flaps)
        for i, ok in enumerate(alive):
            if not ok:
                self.on_death(i)
        return alive

    def on_death(self, i):
        fitness = self.game.birds[i].frames
        self.deaths += 1
        self.history.append(fitness)
        self.best_ever = max(self.best_ever, fitness)
        self.elites.append((self.genomes[i], fitness))
        self.elites = sorted(self.elites, key=lambda e: -e[1])[:ELITES]
        parent = self.elites[self.rng.integers(len(self.elites))][0]
        self.genomes[i] = self.mutate(parent)
        self.game.spawn(i)

    def save_best(self, path=MODEL):
        (w, b), _ = self.elites[0]
        Readout(self.base.feat_idx, w, b, self.base.mean, self.base.std).save(path)

    def close(self):
        for conn in self.conns:
            conn.send(None)
