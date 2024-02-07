from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, NamedTuple, Optional, Tuple
from jaxtyping import Array, Float

from flowlenia.reintegration_tracking import ReintegrationTracking
from flowlenia.vizutils import display_flp
from flowlenia.utils import *

class Config(NamedTuple):
    """
    """
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
    """
    """
    A: Float[Array, "X Y C"] #Cells activations
    P: Float[Array, "X Y K"] #Embedded parameters
    fK:jax.Array             #Kernels fft

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
    clbck: Optional[Callable]
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, callback: Optional[Callable]=None, *, key: jax.Array):

        # ---
        self.cfg = cfg
        # ---
        kR, kr, km, ks, ka, kb, kw = jr.split(key, 7)
        self.R = jr.uniform(kR, (        ), minval=2.000, maxval=25.0)
        self.r = jr.uniform(kr, (cfg.k,  ), minval=0.200, maxval=1.00)
        self.m = jr.uniform(km, (cfg.k,  ), minval=0.050, maxval=0.50) 
        self.s = jr.uniform(ks, (cfg.k,  ), minval=0.001, maxval=0.18)
        self.a = jr.uniform(ka, (cfg.k, 3), minval=0.000, maxval=1.00)
        self.b = jr.uniform(kb, (cfg.k, 3), minval=0.001, maxval=1.00)
        self.w = jr.uniform(kw, (cfg.k, 3), minval=0.010, maxval=0.50)
        # ---
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, 
                                        cfg.border, has_hidden=True, mix=cfg.mix_rule)
        # ---
        self.clbck = callback

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: Optional[jax.Array]=None):
        
        A, P = state.A, state.P
            # --- Original Lenia ---
        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.m, self.s) * P # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- FLOW ---

        F = sobel(U) #(x, y, 2, c) : Flow

        C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient

        alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)

        F = jnp.clip(F * (1 - alpha) - C_grad * alpha, 
                     -(self.cfg.dd-self.cfg.sigma), 
                     self.cfg.dd - self.cfg.sigma)

        nA, nP = self.RT(A, P, F) #type:ignore

        state = state._replace(A=nA, P=nP)

        # --- Callback ---

        if self.clbck is not None:
            state = self.clbck(state, key)
        
        # ---

        return state

    #-------------------------------------------------------------------

    def rollout(self, state: State, key: Optional[jax.Array]=None, 
                steps: int=100)->Tuple[State, State]:
        def _step(c, x):
            s, k = c
            k, k_ = jr.split(k)
            s = self.__call__(s, k_)
            return [s,k],s
        [s, _], S = jax.lax.scan(_step, [state,key], None, steps)
        return s, S

    #-------------------------------------------------------------------

    def rollout_(self, state: State, key: Optional[jax.Array]=None, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.__call__(s), state)

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->State:
        """Compute the kernels fft and put dummy arrays as placeholders for A and P"""
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.k))
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, self.R, self.r, 
                             self.a, self.w, self.b)
        return State(A=A, P=P, fK=fK)

#===========================================================================================
#====================================Simulaton utils========================================
#===========================================================================================


def beam_mutation(state: State, key: jax.Array, sz: int=20, p: float=0.01):
    kmut, kloc, kp = jr.split(key, 3)
    P = state.P
    k = P.shape[-1]
    mut = jnp.ones((sz,sz,k)) * jr.normal(kmut, (1,1,k))
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0]-sz).at[-1].set(0)
    dP = jax.lax.dynamic_update_slice(jnp.zeros_like(P), mut, loc)
    m = (jr.uniform(kp, ()) < p).astype(float)
    P = P + dP*m
    return state._replace(P=P)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cfg = Config(X=128, Y=128, C=3, k=9)
    M = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]])
    c0, c1 = conn_from_matrix(M)
    cfg = cfg._replace(c0=c0, c1=c1)
    flp = FlowLeniaParams(cfg, key=jr.key(1), callback=partial(beam_mutation, sz=20, p=0.1))
    s = flp.initialize(jr.key(1))
    locs = jnp.arange(20) + (cfg.X//2-10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 3)))
    P = s.P.at[jnp.ix_(locs, locs)].set(jnp.ones((20, 20, 9))*jr.uniform(jr.key(111), (1, 1, 9)))
    s = s._replace(A=A, P=P)
    s, S = flp.rollout(s, jr.key(1), 500)
    display_flp(S)


