import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn
import equinox as eqx; import equinox.nn as nn
from .flowlenia import (FlowLenia, Config as ConfigFL, State as StateFL, growth, sobel, 
                        UpdateFunction, ReintegrationTracking, MaceUpdate, get_kernels_fft)
from jaxtyping import Float, Array
from typing import Tuple

class GRN(eqx.Module):
    # ------------------------------------------------------------------
    W_in: jax.Array
    W_ex: jax.Array
    bias: jax.Array
    tau: jax.Array
    dt: float
    # ------------------------------------------------------------------
    def __init__(self,  nb_genes: int, dt: float=0.1, *, key: jax.Array):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.W_in = jr.normal(k1, (nb_genes, nb_genes))
        self.W_ex = jr.normal(k3, (nb_genes, nb_genes))
        self.bias = jr.uniform(k2, (nb_genes,))
        self.tau = jr.uniform(k4, (nb_genes,), minval=0.1, maxval=2.)
        self.dt = dt
    # ------------------------------------------------------------------
    def __call__(self, s: jax.Array, x: jax.Array, key: jax.Array):

        ds = (-s + jnn.tanh(self.W_in@s + self.W_ex@x + self.bias)) * (1/self.tau)
        s = jnp.clip(s + self.dt * ds, 0., 1.)
        return s

moore_kernel = jnp.ones((3, 3, 1)).at[1, 1, 0].set(0.) / 8.

def moore_mean(X: jax.Array, border: str="wall")->jax.Array:
    """Mean of X over the 8 Moore neighbours of each cell. X: (x, y, d)"""
    Xp = jnp.pad(X, ((1,1),(1,1),(0,0)), mode="wrap" if border=="torus" else "constant")
    return jax.scipy.signal.convolve(Xp, moore_kernel, mode="valid")

class Config(ConfigFL):
    nb_genes: int=16
    grn_dt: float=0.1
    matter_thr: float=1e-6 # cells with total matter below this have their gene state zeroed
    growth: bool=True
    c_growth: float=0.1
    growth_genes: int=1

class State(StateFL):
    S: jax.Array


class GRNFlowLenia(eqx.Module):
    # ------------------------------------------------------------------
    grn: GRN
    R: Float
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]
    W_sh: jax.Array
    # Statics
    cfg: Config
    update_mode: str
    update_func: UpdateFunction
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, key: jax.Array):

        # ---
        self.cfg = cfg
        # ---
        kR, kr, km, ks, kg, ka, kb, kw, ksh = jr.split(key, 9)
        self.R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        self.r = jr.uniform(kr, (cfg.k,  ), minval=0.200, maxval=1.00)
        self.m = jr.uniform(km, (cfg.k,  ), minval=0.050, maxval=0.50) 
        self.s = jr.uniform(ks, (cfg.k,  ), minval=0.001, maxval=0.18)
        self.a = jr.uniform(ka, (cfg.k, 3), minval=0.000, maxval=1.00)
        self.b = jr.uniform(kb, (cfg.k, 3), minval=0.001, maxval=1.00)
        self.w = jr.uniform(kw, (cfg.k, 3), minval=0.010, maxval=0.50)
        # ---
        self.grn = GRN(cfg.nb_genes, cfg.grn_dt, key=kg)
        self.W_sh = jr.normal(ksh, (cfg.k,))
        # ---
        self.update_mode = cfg.update_mode
        if cfg.update_mode == "rt":
            self.update_func = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, 
                                                     cfg.border, has_hidden=True)
        elif cfg.update_mode=="mace":
            self.update_func = MaceUpdate(cfg.mace_temp)
        else:
            raise ValueError(f"no update mode called {cfg.update_mode}")

    # ------------------------------------------------------------------

    def __call__(self, state: State, key: jax.Array) -> State: #type:ignore

        # --- Lenia ---
        A = state.A

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        # weght each kernel based on gene expression state
        k = U.shape[-1]
        h = state.S[...,:k] * self.W_sh[None,None]
        U = growth(U, self.m, self.s) * h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- Flow ---

        if self.update_mode=="rt":
            nabla_U = sobel(U) #(x, y, 2, c)
            nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)
            alpha = jnp.clip((A[:, :, None, :]/self.cfg.C)**2, .0, 1.)
            U = nabla_U * (1 - alpha) - nabla_A * alpha
        nA, [nS] = self.update_func(A, U, key, [state.S])

        # --- GRN ---
        
        x = moore_mean(nS*nA.sum(-1, keepdims=True), self.cfg.border) # (x,y,nb_genes) : external signal
        nS = jax.vmap(jax.vmap(self.grn, in_axes=(0,0,None)), in_axes=(0,0,None))(nS, x, key)
        nS = nS * (nA.sum(axis=-1, keepdims=True) > self.cfg.matter_thr)
        state = state.replace(A=nA, S=nS)

        # --- Growth ---
        if self.cfg.growth:
            growth_factor = (state.S[...,k:k+self.cfg.growth_genes].mean(-1)>0.5) * self.cfg.c_growth # H, W
            nA = state.A + growth_factor[...,None]
            state = state.replace(A=nA)

        return state

    # ------------------------------------------------------------------

    def rollout(self, state: State, key: jax.Array, 
                steps: int=100)->Tuple[State, State]:

        def _step(s, x):
            return self.__call__(s, key), s
        return jax.lax.scan(_step, state, None, steps)

    #-------------------------------------------------------------------

    def rollout_(self, state: State, key: jax.Array, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.__call__(s, key), state)

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->State:
        
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, self.R, self.r, 
                             self.a, self.w, self.b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        s0 = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.nb_genes))
        return State(A=A, fK=fK, S=s0)