import numpy as np

DT = 1.0
TAU_M = 20.0
V_TH = 7.0
V_RESET = 0.0
REFRACTORY = 2.2
TAU_SYN = 5.0
DELAY = 2
TAU_ADAPT = 100.0
# ponytail: at gain 1.0 the visual drive tips KC/optic-lobe loops into self-sustained firing;
# 0.8 keeps the network input-driven. Replace with APL/homeostatic inhibition if the central brain matters.
GAIN = 0.8


class LIF:
    def __init__(self, W, gain=GAIN, adapt=0.0, noise_std=0.0, seed=0):
        self.W = W * gain
        self.adapt = np.float32(adapt)
        self.i_adapt = np.zeros(W.shape[0], dtype=np.float32)
        self.n = W.shape[0]
        self.v = np.zeros(self.n, dtype=np.float32)
        self.i_syn = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.float32)
        self.i_ext = np.zeros(self.n, dtype=np.float32)
        self.queue = [np.empty(0, dtype=np.int64) for _ in range(DELAY)]
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.syn_decay = np.float32(np.exp(-DT / TAU_SYN))
        self.adapt_decay = np.float32(np.exp(-DT / TAU_ADAPT))

    def inject(self, idx, current):
        self.i_ext[:] = 0
        self.i_ext[idx] = current

    def step(self):
        arriving = self.queue.pop(0)
        self.i_syn *= self.syn_decay
        if len(arriving):
            self.i_syn += np.asarray(self.W[arriving].sum(axis=0)).ravel()
        self.i_adapt *= self.adapt_decay
        drive = self.i_syn + self.i_ext - self.i_adapt
        if self.noise_std:
            drive = drive + self.rng.normal(0, self.noise_std, self.n).astype(np.float32)
        self.v += (DT / TAU_M) * (-self.v + drive)
        self.v[self.refrac > 0] = V_RESET
        self.refrac -= DT
        spiked = np.flatnonzero(self.v > V_TH)
        self.v[spiked] = V_RESET
        self.refrac[spiked] = REFRACTORY
        self.i_adapt[spiked] += self.adapt
        self.queue.append(spiked)
        return spiked

    def run(self, steps):
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(steps):
            counts[self.step()] += 1
        return counts
