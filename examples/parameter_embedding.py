from flowlenia import FlowLeniaParams, Config_P as Config, State_P as State
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Set system config
nb_k=10
config = Config(SX=256, SY=256, nb_k=nb_k, C=1, c0=[0]*nb_k, 
	c1=[[i for i in range(nb_k)]], dt=.2, mix='stoch')
# 2. initialize system
fl = FlowLeniaParams(config)
# 3. Sample a set of parameters
params = fl.rule_space.sample(jax.random.PRNGKey(42))
# 4. Process parameters
c_params = fl.kernel_computer(params)
# 5. create init state
A = jnp.zeros((256, 256, 1))
P = jnp.zeros((256, 256, nb_k))
centers = [(69, 69), (69, 197), (197, 69), (197, 197)]
key = jax.random.PRNGKey(42)
for x, y in centers :
    key, Akey, Pkey = jax.random.split(key, 3)
    A = A.at[x-20:x+20, y-20:y+20, :].set(
        jax.random.uniform(Akey, (40, 40, 1))
    )
    P = P.at[x-20:x+20, y-20:y+20, :].set(
        jnp.ones((40, 40, nb_k)) * jax.random.uniform(Pkey, (1, 1, nb_k))
    )
state = State(A=A, P=P)
# 6. Make a rollout
T = 500
(final_state, _), traj = fl.rollout_fn(c_params, state, T)
# 7. Display trajectory
fig, ax = plt.subplots()
As = np.array(traj.A)
Ps = np.array(traj.P)
ims = []
for i in range(T):
    A = As[i]
    P = Ps[i]
    img = A.sum(axis=-1, keepdims=True) * P[:, :, :3]
    im = ax.imshow(img, animated=True)
    if i == 0:
        ax.imshow(img)  # show an initial one first
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()