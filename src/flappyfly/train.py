import time

import numpy as np

from .game import Flappy, bot, flap_margin
from .readout import Readout
from .sim import LIF
from .vision import setup

STEPS_PER_FRAME = 10
WARMUP_FRAMES = 100
MAX_FEATURES = 6000
EXPLORE = 0.05


def frame(sim, retina, game):
    retina.see(sim, game.render())
    return sim.run(STEPS_PER_FRAME)


def select_features(W, retina, frames=WARMUP_FRAMES, seed=0):
    """Neurons that respond to random images: every pixel gets exercised, unlike bot play."""
    sim, rng = LIF(W), np.random.default_rng(seed)
    total = np.zeros(W.shape[0], dtype=np.int64)
    for _ in range(frames):
        retina.see(sim, (rng.random((2, retina.n, retina.n)) < 0.2).astype(np.float32))
        total += sim.run(STEPS_PER_FRAME)
    top = np.argsort(-total)[:MAX_FEATURES]
    return np.sort(top[total[top] > 0])


def collect(W, retina, readout, n_frames, seed=0):
    rng = np.random.default_rng(seed)
    sim, game = LIF(W), Flappy(seed)
    X, y = [], []
    for _ in range(n_frames):
        counts = frame(sim, retina, game)
        margin = flap_margin(game.state())
        X.append(readout.features(sim, counts))
        y.append(margin)
        label = margin > 0
        action = not label if rng.random() < EXPLORE else label
        if not game.step(action):
            game.reset()
    return np.stack(X), np.array(y)


def evaluate(W, retina, readout, n_games=5, max_frames=3000, seed=100):
    scores = []
    for g in range(n_games):
        sim, game = LIF(W), Flappy(seed + g)
        for _ in range(max_frames):
            if not game.step(readout.act(sim, frame(sim, retina, game))):
                break
        scores.append(game.bird.score)
    return scores


def random_reservoir(W, seed=0):
    R = W.copy()
    R.indices = np.random.default_rng(seed).permutation(R.indices)
    R.sort_indices()
    return R


def train(n_frames=3000, control=True):
    brain, retina = setup()
    results = {}
    for name, W in [("fly", brain.W)] + ([("random", random_reservoir(brain.W))] if control else []):
        t = time.time()
        readout = Readout(select_features(W, retina))
        X, y = collect(W, retina, readout, n_frames)
        readout.fit(X, y)
        acc = (readout.predict(X) == (y > 0)).mean()
        scores = evaluate(W, retina, readout)
        results[name] = (readout, scores)
        print(f"{name}: {len(readout.feat_idx)} features, {n_frames} frames, "
              f"flap-rate {(y > 0).mean():.2f}, train acc {acc:.3f}, scores {scores}, {time.time() - t:.0f}s")
    return results
