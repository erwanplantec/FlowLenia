import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn

moore_deltas = jnp.asarray([[1, 1],
                            [1, 0],
                            [1, -1],
                            [0, 1],
                            [0, 0],
                            [0, -1],
                            [-1., 1],
                            [-1., 0],
                            [-1., -1]], dtype=jnp.int32)

@jax.jit
def gather_roll(X: jax.Array, deltas: jax.Array=moore_deltas):
    return jax.vmap(jnp.roll, in_axes=(None,0,None))(X, deltas, (0,1))


class MaceUpdate:
    # ------------------------------------------------------------------
    def __init__(self, temp: float=1.0, mix_mode: str="avg"):
        self.temp = temp
        self.mix_mode=mix_mode
    # ------------------------------------------------------------------
    def __call__(self, A: jax.Array, U: jax.Array, key: jax.Array, H:list[jax.Array]|None = None):
        if H is None:
            return self._without_hidden(A, U, key)
        else:
            return self._with_hidden(A, U, H, key)
    # ------------------------------------------------------------------
    def _without_hidden(self, A: jax.Array, U: jax.Array, key: jax.Array):
        
        projection_scores = gather_roll(U) 
        projection_parts = jnn.softmax(projection_scores*(1/self.temp), axis=0) # 9, W, H, C
        projection_qties = projection_parts * A[None]

        gathered_qties = jax.vmap(gather_roll, in_axes=(0,0))(projection_qties, -moore_deltas[:,None]).squeeze(1) #9,1,H,W,C

        new_A = gathered_qties.sum(0)

        return new_A
    # ------------------------------------------------------------------

    def _with_hidden(self, A: jax.Array, U: jax.Array,  H: list[jax.Array], key: jax.Array):

        projection_scores = gather_roll(U) 
        projection_parts = jnn.softmax(projection_scores*(1/self.temp), axis=0) # 9, W, H, C
        projection_qties = projection_parts * A[None]

        # compute incoming qties from each of the 9 neighbours
        gathered_qties = jax.vmap(gather_roll, in_axes=(0,0))(projection_qties, -moore_deltas[:,None]).squeeze(1) #9,1,H,W,C

        new_A = gathered_qties.sum(0)

        if self.mix_mode == "avg":
            gathered_hiddens = [gather_roll(h, -moore_deltas) for h in H]
            # safe denominator: the unused branch of jnp.where must not divide by 0, or gradients become NaN
            safe_A = jnp.where(new_A>1e-5, new_A, 1.0)
            new_H = [
                jnp.where(new_A>1e-5, jnp.sum(h*gathered_qties, axis=0) / safe_A, 0.0) for h in gathered_hiddens
            ]

        else:
            raise ValueError("no mix_mode " + self.mix_mode)

        return new_A, new_H



if __name__ == '__main__':
    H, W = 20, 20
    A = jnp.zeros((H,H,1)).at[2:4,:,:].set(1.)
    U = jnp.linspace(0., 1., H)[:,None].repeat(H, 1)[:,:,None]

    fn = MaceUpdate(temp=1.0)
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    ims = []
    fig, ax = plt.subplots(1, 2, sharey=True)
    for i in range(100):
        im1 = ax[0].imshow(A[...,0], animated=True)
        im2 = ax[1].imshow(U[...,0], animated=True)
        ims.append([im1, im2])
        A = fn(A, U, jr.key(0))
        print(A.sum())
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)
    plt.show()


