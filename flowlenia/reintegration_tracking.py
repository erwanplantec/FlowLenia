import jax
import jax.numpy as jnp
from functools import partial

class ReintegrationTracking:

    #-------------------------------------------------------------------

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, border="wall", has_hidden=False, 
                 mix="stoch"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix

    #-------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        
        if self.has_hidden:
            return self._apply_with_hidden(*args, **kwargs)
        else:
            return self._apply_without_hidden(*args, **kwargs)

    #-------------------------------------------------------------------

    def _apply_without_hidden(self, A: jax.Array, F: jax.Array)->jax.Array:

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def step(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)
            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)

        nA = step(A, mu, dxs, dys).sum(0)
        
        return nA

    #-------------------------------------------------------------------

    def _apply_with_hidden(self, A: jax.Array, H: jax.Array, F: jax.Array):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)
        
        @partial(jax.vmap, in_axes = (None, None, None, 0, 0))
        def step_flow(A, H, mu, dx, dy):
            """Summary
            """
            Ar = jnp.roll(A, (dx, dy), axis = (0, 1))
            Hr = jnp.roll(H, (dx, dy), axis = (0, 1)) #(x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis = (0, 1))

            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)

            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA, Hr

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
        nA, nH = step_flow(A, H, mu, dxs, dys)

        if self.mix == 'avg':
            nH = jnp.sum(nH * nA.sum(axis = -1, keepdims = True), axis = 0)  
            nA = jnp.sum(nH, axis = 0)
            nH = nH / (nA.sum(axis = -1, keepdims = True)+1e-10)

        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis = -1, keepdims = True)) - 1
            nA = jnp.sum(nA, axis = 0)
            nH = jnp.sum(nH * expnA, axis = 0) / (expnA.sum(axis = 0)+1e-10) #avg rule

        elif self.mix == "stoch":
            categorical=jax.random.categorical(
              jax.random.PRNGKey(42), 
              jnp.log(nA.sum(axis=-1, keepdims=True)), 
              axis=0)
            mask=jax.nn.one_hot(categorical,num_classes=(2*self.dd+1)**2,axis=-1)
            mask=jnp.transpose(mask,(3,0,1,2)) 
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)

        elif self.mix == "stoch_gene_wise":
            mask = jnp.concatenate(
              [jax.nn.one_hot(jax.random.categorical(
                                                    jax.random.PRNGKey(42), 
                                                    jnp.log(nA.sum(axis = -1, keepdims = True)), 
                                                    axis=0),
                              num_classes=(2*dd+1)**2,axis=-1)
              for _ in range(H.shape[-1])], 
              axis = 2)
            mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)
        
        return nA, nH

    #-------------------------------------------------------------------

