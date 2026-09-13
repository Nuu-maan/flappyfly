import time

import numpy as np

from .game import Flappy, bot, flap_margin
from .readout import Readout
from .sim import LIF
from .vision import Retina

STEPS_PER_FRAME = 10
WARMUP_FRAMES = 100
EXPLORE = 0.05


def frame(sim, retina, game):
    retina.see(sim, game.render())
    return sim.run(STEPS_PER_FRAME)


def select_features(W, retina, frames=WARMUP_FRAMES, seed=0):
    sim, game = LIF(W), Flappy(seed)
    seen = np.zeros(W.shape[0], dtype=bool)
    for _ in range(frames):
        seen |= frame(sim, retina, game) > 0
        game.step(bot(game.state()))
    return np.flatnonzero(seen)


def collect(W, retina, readout, n_frames, seed=0):
    rng = np.random.default_rng(seed)
    sim, game = LIF(W), Flappy(seed)
    X, y = [], []
    for _ in range(n_frames):
        counts = frame(sim, retina, game)
        margin = flap_margin(game.state())
        X.append(readout.observe(counts))
        y.append(margin)
        label = margin > 0
        action = not label if rng.random() < EXPLORE else label
        _, alive = game.step(action)
        if not alive:
            game.reset()
    return np.stack(X), np.array(y)


def evaluate(W, retina, readout, n_games=5, max_frames=3000, seed=100):
    scores = []
    for g in range(n_games):
        sim, game = LIF(W), Flappy(seed + g)
        for _ in range(max_frames):
            _, alive = game.step(readout.act(frame(sim, retina, game)))
            if not alive:
                break
        scores.append(game.score)
    return scores


def random_reservoir(W, seed=0):
    R = W.copy()
    R.indices = np.random.default_rng(seed).permutation(R.indices)
    R.sort_indices()
    return R


def train(brain, n_frames=3000, control=True):
    retina = Retina(brain)
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
