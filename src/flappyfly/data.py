import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
DATA_DIR = Path("data")
CACHE = DATA_DIR / "brain.npz"

MIN_WEIGHT = 3
MV_PER_SYNAPSE = 0.275
INHIBITORY = {"gaba", "glutamate", "histamine"}


def fetch(name):
    path = DATA_DIR / FILES[name]
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading {FILES[name]}")
        urllib.request.urlretrieve(BASE + FILES[name], path)
    return path


def build():
    ann = pf.read_table(fetch("annotations")).to_pandas()
    ann = ann[ann.type.notna()].sort_values("bodyId").reset_index(drop=True)
    nt = pf.read_table(fetch("nt"), columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.set_index("body").consensus_nt.reindex(ann.bodyId).fillna("unknown").to_numpy()
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)

    n = len(ann)
    ids = ann.bodyId.to_numpy()

    def lookup(bodies):
        idx = np.searchsorted(ids, bodies).clip(0, n - 1)
        return np.where(ids[idx] == bodies, idx, -1)

    rows, cols, vals = [], [], []
    for batch in pf.read_table(fetch("weights"), memory_map=True).to_batches():
        batch = batch.filter(pc.greater_equal(batch["weight"], MIN_WEIGHT))
        pre = lookup(batch["body_pre"].to_numpy())
        post = lookup(batch["body_post"].to_numpy())
        keep = (pre >= 0) & (post >= 0)
        rows.append(pre[keep])
        cols.append(post[keep])
        vals.append(batch["weight"].to_numpy()[keep].astype(np.float32))
    pre, post, w = map(np.concatenate, (rows, cols, vals))
    W = sp.csr_matrix((sign[pre] * w * MV_PER_SYNAPSE, (pre, post)), shape=(n, n))

    soma = np.array([xyz if xyz is not None else [np.nan] * 3 for xyz in ann.somaLocation], dtype=np.float32)
    np.savez(
        CACHE,
        bodyId=ann.bodyId.to_numpy(),
        type=ann.type.to_numpy().astype(str),
        superclass=ann.superclass.fillna("").to_numpy().astype(str),
        side=ann.somaSide.fillna("").to_numpy().astype(str),
        hex=ann[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=np.float32),
        soma=soma,
        nt=nt.astype(str),
        sign=sign,
        W_data=W.data,
        W_indices=W.indices,
        W_indptr=W.indptr,
    )
    print(f"cached {n} neurons, {W.nnz} edges -> {CACHE}")
