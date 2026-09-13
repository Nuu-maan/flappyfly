import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data import CACHE, build


@dataclass
class Brain:
    bodyId: np.ndarray
    type: np.ndarray
    superclass: np.ndarray
    side: np.ndarray
    hex: np.ndarray
    soma: np.ndarray
    nt: np.ndarray
    sign: np.ndarray
    W: sp.csr_matrix

    @property
    def n(self):
        return len(self.bodyId)

    def neurons_of(self, type_regex=None, superclass=None, side=None):
        mask = np.ones(self.n, dtype=bool)
        if type_regex:
            pat = re.compile(type_regex)
            mask &= np.array([bool(pat.fullmatch(t)) for t in self.type])
        if superclass:
            mask &= self.superclass == superclass
        if side:
            mask &= self.side == side
        return np.flatnonzero(mask)


def load():
    if not CACHE.exists():
        build()
    d = np.load(CACHE)
    n = len(d["bodyId"])
    W = sp.csr_matrix((d["W_data"], d["W_indices"], d["W_indptr"]), shape=(n, n))
    return Brain(*(d[k] for k in ("bodyId", "type", "superclass", "side", "hex", "soma", "nt", "sign")), W)
