from flowlenia import FlowLenia, Config, State
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Set system config
nb_k=10
config = Config(SX=256, SY=256, nb_k=nb_k, C=1, c0=[0]*nb_k, 
	c1=[[i for i in range(nb_k)]], dt=.2)
# 2. initialize system
fl = FlowLenia(config)
# 3. Sample a set of parameters
params = fl.rule_space.sample(jax.random.PRNGKey(42))
# 4. Process parameters
c_params = fl.kernel_computer(params)
# 5. create init state
A = jnp.zeros((256, 256, 1)).at[108:148, 108:148, :].set(
		jax.random.uniform(jax.random.PRNGKey(42), (40, 40, 1))
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
    im = ax.imshow(As[i], animated=True)
    if i == 0:
        ax.imshow(As[i])  # show an initial one first
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()