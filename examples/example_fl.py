from flowlenia.flowlenia import FlowLenia, Config, State
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flowlenia.utils import conn_from_matrix

seed = 1111
key = jr.key(seed)
# --- 1. Set system config
M = np.array([[3, 1, 0],
              [0, 3, 1],
              [1, 0, 3]], dtype=int)
k = M.sum()
c0, c1 = conn_from_matrix(M)
cfg = Config(X=128, Y=128, C=3, k=k, c0=c0, c1=c1)
# --- 2. initialize system
key, key_fl = jr.split(key)
fl = FlowLenia(cfg, key=jr.key(seed))
# --- 3. Initialize state
key, key_A = jr.split(key)
s = fl.initialize(jr.key(1))
A = s.A.at[44:84, 44:84, :].set(jr.uniform(key_A, (40, 40, cfg.C)))
s = s._replace(A=A)
# --- 4. Simulate rollout
s, ss = fl.rollout(s, steps=100)
# --- 5. Display
ims = []
fig, ax = plt.subplots()
for i in range(100):
    im = ax.imshow(ss.A[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()


