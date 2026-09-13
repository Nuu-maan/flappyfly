import numpy as np

INPUT_TYPES = ("L2", "L3", "L5")
EYE = "R"
DRIVE_MV = 25.0


class Retina:
    def __init__(self, brain, n=16):
        idx = np.concatenate([brain.neurons_of(t, side=EYE) for t in INPUT_TYPES])
        hexes = brain.hex[idx]
        ok = np.isfinite(hexes).all(axis=1)
        self.idx, hexes = idx[ok], hexes[ok]
        lo, hi = hexes.min(axis=0), hexes.max(axis=0)
        grid = ((hexes - lo) / (hi - lo) * (n - 1)).round().astype(int)
        self.col, self.row = grid[:, 0], grid[:, 1]
        self.n = n

    def currents(self, img):
        return DRIVE_MV * img[self.row, self.col]

    def see(self, sim, img):
        sim.inject(self.idx, self.currents(img))
