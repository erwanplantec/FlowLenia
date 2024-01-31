"""Summary
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Optional, Tuple
from jaxtyping import Float, Array

from flowlenia.utils import *
from flowlenia.reintegration_tracking import ReintegrationTracking



class Config(NamedTuple):
    X: int=128
    Y: int=128
    C: int=1
    c0: list[int]=[0]
    c1: list[list[int]]=[[0]]
    k: int=10
    dd: int=5
    dt: float=0.2
    sigma: float=.65
    border: str="wall"

class State(NamedTuple):
    A: Float[Array, "X Y C"]
    fK: Float[Array, "X Y k"]

class FlowLenia(eqx.Module):
    """
    """
    #-------------------------------------------------------------------
    # Parameters:
    R: Float
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    h: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]
    # Statics
    cfg: Config
    RT: ReintegrationTracking
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, key: jax.Array):

        # ---
        self.cfg = cfg
        # ---
        kR, kr, km, ks, kh, ka, kb, kw = jr.split(key, 8)
        self.R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        self.r = jr.uniform(kr, (cfg.k,  ), minval=0.200, maxval=1.00)
        self.m = jr.uniform(km, (cfg.k,  ), minval=0.050, maxval=0.50) 
        self.s = jr.uniform(ks, (cfg.k,  ), minval=0.001, maxval=0.18)
        self.h = jr.uniform(kh, (cfg.k,  ), minval=0.010, maxval=1.00)
        self.a = jr.uniform(ka, (cfg.k, 3), minval=0.000, maxval=1.00)
        self.b = jr.uniform(kb, (cfg.k, 3), minval=0.001, maxval=1.00)
        self.w = jr.uniform(kw, (cfg.k, 3), minval=0.010, maxval=0.50)
        # ---
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, 
                                        cfg.border, has_hidden=False)

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: Optional[jax.Array]=None)->State:
        
        # --- Lenia ---
        A = state.A

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.m, self.s) * self.h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- Flow ---

        nabla_U = sobel(U) #(x, y, 2, c)

        nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

        alpha = jnp.clip((A[:, :, None, :]/self.cfg.C)**2, .0, 1.)

        F = nabla_U * (1 - alpha) - nabla_A * alpha
        nA = self.RT(A, F) #type:ignore

        return state._replace(A=nA)
    
    #-------------------------------------------------------------------

    def rollout(self, state: State, key: Optional[jax.Array]=None, 
                steps: int=100)->Tuple[State, State]:

        def _step(s, x):
            return self.__call__(s), s
        return jax.lax.scan(_step, state, None, steps)

    #-------------------------------------------------------------------

    def rollout_(self, state: State, key: Optional[jax.Array]=None, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.__call__(s), state)

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->State:
        
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, self.R, self.r, 
                             self.a, self.w, self.b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        return State(A=A, fK=fK)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cfg = Config(X=64, Y=64, C=3, k=9)
    M = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]])
    c0, c1 = conn_from_matrix(M)
    cfg = cfg._replace(c0=c0, c1=c1)
    fl = FlowLenia(cfg, key=jr.key(101))
    s = fl.initialize(jr.key(2))
    locs = jnp.arange(20) + (cfg.X//2-10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 1)))
    s = s._replace(A=A)
    s = fl.rollout_(s, None, 100)
    plt.imshow(s.A); plt.show()








