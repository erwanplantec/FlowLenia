from flowlenia.grn_flowlenia import GRNFlowLenia, Config, State
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flowlenia.utils import conn_from_matrix

seed = np.random.randint(0,10000)
key = jr.key(seed)
# --- 1. Set system config
M = np.array([[8]], dtype=int)
k = M.sum()
c0, c1 = conn_from_matrix(M)
# first k genes are used as per-kernel growth weights, so nb_genes must be >= k
cfg = Config(X=128, Y=128, C=M.shape[0], k=k, c0=c0, c1=c1, mace_temp=0.5, update_mode="mace",
             nb_genes=16, grn_dt=0.05, growth=False, c_growth=0.01)
# --- 2. initialize system
key, key_fl = jr.split(key)
fl = GRNFlowLenia(cfg, key=key_fl)
# --- 3. Initialize state
key, key_A, key_s = jr.split(key, 3)
s = fl.initialize(jr.key(1))
A = s.A.at[44:84, 44:84, :].set(jr.uniform(key_A, (40, 40, cfg.C)))
g = s.S.at[44:84, 44:84, :].set(jr.uniform(key_s, (40, 40, cfg.nb_genes)))
s = s.replace(A=A, S=g)
# --- 4. Simulate rollout
T = 500
s, ss = fl.rollout(s, jr.key(0), steps=T)
print("max gene expression: ", float(ss.S.max())) #type:ignore
# --- 5. Display (matter colored by first 3 genes, as RGB)
# average gene expression over occupied cells, per frame: (T, nb_genes)
occ = np.asarray(ss.A).sum(-1, keepdims=True) > cfg.matter_thr  # (T, X, Y, 1)
avg_S = (np.asarray(ss.S) * occ).sum((1, 2)) / np.maximum(occ.sum((1, 2)), 1)
ims = []
fig, ax = plt.subplots(2, 1, figsize=(6, 10), height_ratios=[0.7,0.3])
ax[1].set_xlim(0, T); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("step"); ax[1].set_ylabel("mean gene expression (occupied cells)")
for i in range(T):
    rgb = np.asarray(ss.S[i][..., :3])  # genes are already in [0, 1]
    img = np.clip(np.asarray(ss.A[i]).sum(-1, keepdims=True), 0., 1.) * rgb
    im = ax[0].imshow(img, animated=True)
    plots = ax[1].plot(np.arange(i+1), avg_S[:i+1], animated=True, color="k")
    ims.append([im, *plots])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=100)
plt.show()
