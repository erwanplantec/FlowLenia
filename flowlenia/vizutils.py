import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def display_fl(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A = states.A[i]
        C = A.shape[-1]
        if C==1:
            img = A
        if C==2:
            img=jnp.dstack([A[...,0], A[...,0], A[...,1]])
        else:
            img = A[...,:3]
        im = ax.imshow(img, animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

def display_flp(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A, P = states.A[i], states.P[i]
        im = ax.imshow(P[..., :3] * A.sum(-1, keepdims=True), animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()