import numpy as np

from .brain import load

INPUT_TYPES = ("L1", "L2", "L3", "L5")
EYES = ("L", "R")
DRIVE_MV = 25.0


class Retina:
    """Each image channel is shown to one eye: channel 0 (the world) to the left, channel 1 (the bird) to the right."""

    def __init__(self, brain, n=16):
        self.n = n
        self.idx, self.chan, self.row, self.col = [], [], [], []
        for c, eye in enumerate(EYES):
            idx = np.concatenate([brain.neurons_of(t, side=eye) for t in INPUT_TYPES])
            hexes = brain.hex[idx]
            ok = np.isfinite(hexes).all(axis=1)
            idx, hexes = idx[ok], hexes[ok]
            lo, hi = hexes.min(axis=0), hexes.max(axis=0)
            grid = ((hexes - lo) / (hi - lo) * (n - 1)).round().astype(int)
            self.idx.append(idx)
            self.chan.append(np.full(len(idx), c))
            self.col.append(grid[:, 0])
            self.row.append(grid[:, 1])
        self.idx, self.chan, self.row, self.col = map(np.concatenate, (self.idx, self.chan, self.row, self.col))

    def currents(self, img):
        return DRIVE_MV * img[self.chan, self.row, self.col]

    def see(self, sim, img):
        sim.inject(self.idx, self.currents(img))


def setup(hops=2):
    """The brain pruned to what the retina can reach. ponytail: at GAIN 0.8 no neuron beyond 2 hops
    ever spikes (checked over 400 game frames); raise hops if the gain goes up."""
    brain = load()
    brain = brain.subgraph(brain.reachable(Retina(brain).idx, hops))
    return brain, Retina(brain)
