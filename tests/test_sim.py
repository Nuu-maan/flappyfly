import numpy as np
import pytest

from flappyfly.brain import load
from flappyfly.sim import LIF
from flappyfly.vision import Retina, setup


@pytest.fixture(scope="module")
def brain():
    return load()


def test_silent_without_input(brain):
    assert LIF(brain.W).run(50).sum() == 0


def test_pruned_brain_keeps_retina_and_shrinks(brain):
    sub, retina = setup()
    assert len(retina.idx) == len(Retina(brain).idx)
    assert 20000 < sub.n < brain.n


def test_vision_drives_activity_and_is_input_specific(brain):
    retina = Retina(brain)
    left, right = np.zeros((2, 16, 16), dtype=np.float32), np.zeros((2, 16, 16), dtype=np.float32)
    left[:, :, :8], right[:, :, 8:] = 1.0, 1.0
    counts = []
    for img in (left, right):
        sim = LIF(brain.W)
        retina.see(sim, img)
        counts.append(sim.run(100))
    a, b = counts
    assert a.sum() > 1000 and b.sum() > 1000
    assert (a > 0).sum() > (retina.currents(left) > 0).sum()
    active = (a > 0) | (b > 0)
    assert np.corrcoef(a[active], b[active])[0, 1] < 0.5
