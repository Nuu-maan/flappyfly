from pathlib import Path

import numpy as np

MODEL = Path("data/readout.npz")
L2 = 3000.0


class Readout:
    def __init__(self, feat_idx, w=None, b=0.0, mean=None, std=None, fitness=0):
        self.feat_idx = feat_idx
        self.fitness = fitness
        d = 2 * len(feat_idx)
        self.w = np.zeros(d, dtype=np.float32) if w is None else w
        self.b = b
        self.mean = np.zeros(d, dtype=np.float32) if mean is None else mean
        self.std = np.ones(d, dtype=np.float32) if std is None else std

    def features(self, sim, counts):
        return np.concatenate([sim.v[self.feat_idx], counts[self.feat_idx]]).astype(np.float32)

    def score(self, X, w=None, b=None):
        w = self.w if w is None else w
        b = self.b if b is None else b
        return (X - self.mean) / self.std @ w + b

    def predict(self, X):
        return self.score(X) > 0

    def act(self, sim, counts):
        return bool(self.predict(self.features(sim, counts)))

    def fit(self, X, t, l2=L2, mean=None, std=None):
        self.mean = X.mean(axis=0) if mean is None else mean
        self.std = X.std(axis=0) + 1e-3 if std is None else std
        Z = (X - self.mean) / self.std
        A = Z.T @ Z + l2 * np.eye(Z.shape[1], dtype=np.float32)
        self.w = np.linalg.solve(A, Z.T @ (t - t.mean())).astype(np.float32)
        self.b = float(t.mean())
        return self

    def save(self, path=MODEL):
        np.savez(path, feat_idx=self.feat_idx, w=self.w, b=self.b, mean=self.mean, std=self.std, fitness=self.fitness)

    @classmethod
    def load(cls, path=MODEL):
        d = np.load(path)
        return cls(d["feat_idx"], d["w"], float(d["b"]), d["mean"], d["std"], int(d["fitness"]) if "fitness" in d else 0)

    @staticmethod
    def saved_fitness(path=MODEL):
        return int(np.load(path)["fitness"]) if path.exists() and "fitness" in np.load(path) else 0
