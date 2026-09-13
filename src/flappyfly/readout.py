from pathlib import Path

import numpy as np

MODEL = Path("data/readout.npz")
TRACE_DECAY = 0.7
L2 = 3000.0


class Readout:
    def __init__(self, feat_idx, w=None, b=0.0, mean=None, std=None):
        self.feat_idx = feat_idx
        self.w = np.zeros(len(feat_idx), dtype=np.float32) if w is None else w
        self.b = b
        self.mean = np.zeros(len(feat_idx), dtype=np.float32) if mean is None else mean
        self.std = np.ones(len(feat_idx), dtype=np.float32) if std is None else std
        self.trace = np.zeros(len(feat_idx), dtype=np.float32)

    def observe(self, counts):
        self.trace = self.trace * TRACE_DECAY + counts[self.feat_idx]
        return self.trace.copy()

    def predict(self, X):
        return (X - self.mean) / self.std @ self.w + self.b > 0

    def act(self, counts):
        return bool(self.predict(self.observe(counts)))

    def fit(self, X, t, l2=L2):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-3
        Z = (X - self.mean) / self.std
        A = Z.T @ Z + l2 * np.eye(Z.shape[1], dtype=np.float32)
        self.w = np.linalg.solve(A, Z.T @ (t - t.mean())).astype(np.float32)
        self.b = float(t.mean())
        return self

    def save(self, path=MODEL):
        np.savez(path, feat_idx=self.feat_idx, w=self.w, b=self.b, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path=MODEL):
        d = np.load(path)
        return cls(d["feat_idx"], d["w"], float(d["b"]), d["mean"], d["std"])
