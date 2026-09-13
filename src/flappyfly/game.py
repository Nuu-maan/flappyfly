import numpy as np

W, H = 288, 512
BIRD_X, BIRD_R = 60, 12
GRAVITY, FLAP, MAX_VY = 0.6, -8.0, 10.0
PIPE_W, GAP, PIPE_DX, SPEED = 52, 120, 160, 3


class Flappy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.y, self.vy = H / 2, 0.0
        self.pipes = [[W + i * PIPE_DX, self._gap_y()] for i in range(3)]
        self.score, self.frames, self.alive = 0, 0, True
        return self.state()

    def _gap_y(self):
        return float(self.rng.uniform(GAP, H - GAP))

    def next_pipe(self):
        return min((p for p in self.pipes if p[0] + PIPE_W > BIRD_X - BIRD_R), key=lambda p: p[0])

    def state(self):
        px, gy = self.next_pipe()
        return np.array([self.y, self.vy, px - BIRD_X, gy], dtype=np.float32)

    def step(self, flap):
        if not self.alive:
            return self.state(), False
        self.vy = FLAP if flap else min(self.vy + GRAVITY, MAX_VY)
        self.y += self.vy
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[0][0] + PIPE_W < 0:
            self.pipes.pop(0)
            self.pipes.append([self.pipes[-1][0] + PIPE_DX, self._gap_y()])
            self.score += 1
        px, gy = self.next_pipe()
        in_pipe = px < BIRD_X + BIRD_R and px + PIPE_W > BIRD_X - BIRD_R
        hit = in_pipe and (self.y - BIRD_R < gy - GAP / 2 or self.y + BIRD_R > gy + GAP / 2)
        self.alive = not (hit or self.y < 0 or self.y > H)
        self.frames += 1
        return self.state(), self.alive

    def render(self, n=16):
        img = np.zeros((n, n), dtype=np.float32)
        sx, sy = n / W, n / H
        for px, gy in self.pipes:
            x0, x1 = max(0, int(px * sx)), min(n, int((px + PIPE_W) * sx) + 1)
            img[: int((gy - GAP / 2) * sy), x0:x1] = 0.5
            img[int((gy + GAP / 2) * sy):, x0:x1] = 0.5
        img[min(n - 1, int(self.y * sy)), int(BIRD_X * sx)] = 1.0
        return img


def bot(state):
    y, vy, _, gy = state
    return y + vy * 3 > gy

