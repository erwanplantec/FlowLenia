import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Optional

from jaxtyping import Array, Float

from flowlenia.reintegration_tracking import ReintegrationTracking
from flowlenia.utils import *

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
    mix_rule: str="stoch"

class State(NamedTuple):
    A: Float[Array, "X Y C"]
    P: Float[Array, "X Y K"]
    fK:jax.Array


class FlowLeniaParams(eqx.Module):
    
    """
    """
    #-------------------------------------------------------------------
    # Parameters:
    R: Float
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]
    # Statics:
    cfg: Config
    RT: ReintegrationTracking
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, *, key: jax.Array):

        # ---
        self.cfg = cfg
        # ---
        kR, kr, km, ks, ka, kb, kw = jr.split(key, 7)
        self.R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        self.r = jr.uniform(kr, (cfg.k,  ), minval=0.200, maxval=1.00)
        self.m = jr.uniform(km, (cfg.k,  ), minval=0.050, maxval=0.50) 
        self.s = jr.uniform(ks, (cfg.k,  ), minval=0.001, maxval=0.18)
        self.a = jr.uniform(ka, (cfg.k, 3), minval=0.000, maxval=1.00)
        self.b = jr.uniform(kb, (cfg.k, 3), minval=0.001, maxval=1.00)
        self.w = jr.uniform(kw, (cfg.k, 3), minval=0.010, maxval=0.50)
        # ---
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, 
                                        cfg.border, has_hidden=True, mix=cfg.mix_rule)

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: Optional[jax.Array]=None):
        
        A, P = state.A, state.P
            #---------------------------Original Lenia------------------------------------
        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.m, self.s) * P # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        #-------------------------------FLOW------------------------------------------

        F = sobel(U) #(x, y, 2, c) : Flow

        C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient

        alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)

        F = jnp.clip(F * (1 - alpha) - C_grad * alpha, 
                     -(self.cfg.dd-self.cfg.sigma), 
                     self.cfg.dd - self.cfg.sigma)

        nA, nP = self.RT(A, P, F) #type:ignore

        return state._replace(A=nA, P=nP)

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->State:

        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.k))
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, self.R, self.r, 
                             self.a, self.w, self.b)
        return State(A=A, P=P, fK=fK)


if __name__ == '__main__':
    cfg = Config()
    c0, c1 = conn_from_matrix(np.ones((1,1),dtype=int))
    cfg = cfg._replace(c0=c0, c1=c1)
    flp = FlowLeniaParams(cfg, key=jr.key(1))
    s = flp.initialize(jr.key(1))
    s = flp(s)
    print(s.A.shape)
    print(s.P.shape)
    print(s.fK.shape)
