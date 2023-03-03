from flowlenia import FlowLenia, Config, State
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

# 1. Set system config
nb_k=10
config = Config(SX=256, SY=256, nb_k=10, C=1, c0=[0]*nb_k, 
	c1 = [[i for i in range(nb_k)]], dt=.2)
# 2. initialize system
fl = FlowLenia(config)
# 3. Sample a set of parameters
params = fl.rule_space.sample(jax.random.PRNGKey(42))
# 4. Process parameters
c_params = fl.kernel_computer(params)
# 5. create init state
A = jax.random.uniform(jax.random.PRNGKey(42), (256, 256, 1))
state = State(A=A)
# 6. Make a rollout
(final_state, _), traj = fl.rollout_fn(c_params, state, 300)

plt.imshow(final_state.A)
plt.show()