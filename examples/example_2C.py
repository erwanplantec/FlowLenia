from flowlenia import FlowLenia, Config, State
from flowlenia.utils import conn_from_matrix
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Set system config
C = 2
nb_k=20
c0, c1 = conn_from_matrix(
    np.array([[5, 5],
              [5, 5]])
)
config = Config(SX=256, SY=256, nb_k=nb_k, C=C, c0=c0, c1=c1, dt=.2)
# 2. initialize system
fl = FlowLenia(config)
# 3. Sample a set of parameters
params = fl.rule_space.sample(jax.random.PRNGKey(42))
# 4. Process parameters
c_params = fl.kernel_computer(params)
# 5. create init state
A = jnp.zeros((256, 256, C)).at[108:148, 108:148, :].set(
		jax.random.uniform(jax.random.PRNGKey(99), (40, 40, C))
	)
state = State(A=A)
# 6. Make a rollout
T = 500
(final_state, _), traj = fl.rollout_fn(c_params, state, T)
# 7. Display trajectory
fig, ax = plt.subplots()
As = np.array(traj.A)
ims = []
for i in range(T):
    A = np.dstack([As[i, :, :, 0], As[i, :, :, 0], As[i, :, :, 1]])
    im = ax.imshow(A, animated=True)
    if i == 0:
        ax.imshow(A)  # show an initial one first
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()