from flappyfly.game import Flappy, bot


def test_bot_survives():
    g = Flappy()
    for _ in range(5000):
        _, alive = g.step(bot(g.state()))
        assert alive, f"bot died at frame {g.frames}"
    assert g.score > 50


def test_render():
    img = Flappy().render()
    assert img.shape == (16, 16) and img.max() == 1.0
