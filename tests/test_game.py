import numpy as np

from flappyfly.game import Flappy, bot


def test_bot_survives():
    g = Flappy()
    for _ in range(5000):
        assert g.step(bot(g.state())), f"bot died at frame {g.bird.frames}"
    assert g.bird.score > 50


def test_render():
    g = Flappy()
    for _ in range(40):
        g.step(bot(g.state()))
    img = g.render()
    assert img.shape == (2, 16, 16) and img[0].max() == 1.0 and img[1].max() == 1.0


def test_population_shares_pipes_and_respawns():
    g = Flappy(n_birds=3)
    alive = g.step([True, False, bot(g.state(2))])
    assert alive == [True, True, True]
    g.birds[1].y = -100
    assert g.step([False] * 3) == [True, False, True]
    assert g.step([False] * 3)[1] is False
    g.spawn(1)
    assert g.birds[1].alive and abs(g.birds[1].y - g.next_pipe()[1]) < 1
    assert np.array_equal(g.render(0), g.render(0))
