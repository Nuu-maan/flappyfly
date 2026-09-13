import numpy as np

W, H = 288, 512
BIRD_X, BIRD_R = 60, 12
GRAVITY, FLAP, MAX_VY = 1.0, -7.0, 10.0
PIPE_W, GAP, PIPE_DX, SPEED, MAX_GAP_JUMP = 52, 120, 200, 3, 150


class Bird:
    def __init__(self, y):
        self.y, self.vy = float(y), 0.0
        self.alive, self.frames, self.score = True, 0, 0


class Flappy:
    def __init__(self, seed=0, n_birds=1):
        self.rng = np.random.default_rng(seed)
        self.n_birds = n_birds
        self.reset()

    def reset(self):
        self.pipes = [[W, H / 2]]
        for i in range(2):
            self.pipes.append([W + (i + 1) * PIPE_DX, self._gap_y()])
        self.birds = [Bird(H / 2) for _ in range(self.n_birds)]
        self.frames = 0

    def spawn(self, i):
        self.birds[i] = Bird(self.next_pipe()[1])

    def _gap_y(self):
        prev = self.pipes[-1][1]
        return float(np.clip(prev + self.rng.uniform(-MAX_GAP_JUMP, MAX_GAP_JUMP), GAP, H - GAP))

    def next_pipe(self):
        return min((p for p in self.pipes if p[0] + PIPE_W > BIRD_X - BIRD_R), key=lambda p: p[0])

    @property
    def bird(self):
        return self.birds[0]

    def state(self, i=0):
        b = self.birds[i]
        px, gy = self.next_pipe()
        return np.array([b.y, b.vy, px - BIRD_X, gy], dtype=np.float32)

    def step(self, flaps):
        single = isinstance(flaps, (bool, np.bool_))
        flaps = [flaps] if single else flaps
        for p in self.pipes:
            p[0] -= SPEED
        passed = self.pipes[0][0] + PIPE_W < 0
        if passed:
            self.pipes.pop(0)
            self.pipes.append([self.pipes[-1][0] + PIPE_DX, self._gap_y()])
        px, gy = self.next_pipe()
        in_pipe = px < BIRD_X + BIRD_R and px + PIPE_W > BIRD_X - BIRD_R
        for b, flap in zip(self.birds, flaps):
            if not b.alive:
                continue
            b.vy = FLAP if flap else min(b.vy + GRAVITY, MAX_VY)
            b.y += b.vy
            b.score += passed
            b.frames += 1
            hit = in_pipe and (b.y - BIRD_R < gy - GAP / 2 or b.y + BIRD_R > gy + GAP / 2)
            b.alive = not (hit or b.y < 0 or b.y > H)
        self.frames += 1
        alive = [b.alive for b in self.birds]
        return alive[0] if single else alive

    def render(self, i=0, n=16):
        """Two channels: the next pipe and the bird itself, so they never occlude each other.
        ponytail: only the next pipe is shown (attention); a linear readout cannot gate between two visible gaps."""
        img = np.zeros((2, n, n), dtype=np.float32)
        sx, sy = n / W, n / H
        px, gy = self.next_pipe()
        x0, x1 = max(0, int(px * sx)), min(n, int((px + PIPE_W) * sx) + 1)
        img[0, : int((gy - GAP / 2) * sy), x0:x1] = 1.0
        img[0, int((gy + GAP / 2) * sy):, x0:x1] = 1.0
        b = self.birds[i]
        r0, r1 = int((b.y - BIRD_R) * sy), int((b.y + BIRD_R) * sy) + 1
        c0, c1 = int((BIRD_X - BIRD_R) * sx), int((BIRD_X + BIRD_R) * sx) + 1
        img[1, max(0, r0):r1, c0:c1] = 1.0
        return img


def flap_margin(state):
    y, vy, _, gy = state
    return y + vy - (gy + 12)


def bot(state):
    return flap_margin(state) > 0
