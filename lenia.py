import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import typing as t
import chex
from utils import *

@chex.dataclass
class Params:

    """Summary
    """
    
    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    R: float
    T: float

@chex.dataclass
class CompiledParams:

    """Summary
    """
    
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    T: float

class Rule_space :

    """Summary
    
    Attributes:
        init_shape (TYPE): Description
        kernel_keys (TYPE): Description
        nb_k (TYPE): Description
        spaces (TYPE): Description
    """
    
    #-----------------------------------------------------------------------------
    def __init__(self, nb_k):
        """Summary
        
        Args:
            nb_k (TYPE): Description
        """
        self.nb_k = nb_k    
        self.kernel_keys = 'r b w a m s h'.split()
        self.spaces = {
            "r" : {'low' : .2, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            "b" : {'low' : .001, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "w" : {'low' : .01, 'high' : .5, 'mut_std' : .2, 'shape' : (3,)},
            "a" : {'low' : .0, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "m" : {'low' : .05, 'high' : .5, 'mut_std' : .2, 'shape' : None},
            "s" : {'low' : .001, 'high' : .18, 'mut_std' : .01, 'shape' : None},
            "h" : {'low' : .01, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            'R' : {'low' : 2., 'high' : 25., 'mut_std' : .2, 'shape' : None},
            'T' : {'low' : 10., 'high' : 50., 'mut_std' : .1, 'shape' : None}
        }
    #-----------------------------------------------------------------------------
    def sample(self)->Params:
        """Summary
        
        Returns:
            Params: Description
        """
        kernels = {}
        for k in 'rmsh':
          kernels[k] = np.random.uniform(
              self.spaces[k]['low'], self.spaces[k]['high'], self.nb_k
          )
        for k in "awb":
          kernels[k] = np.random.uniform(
              self.spaces[k]['low'], self.spaces[k]['high'], (self.nb_k, 3)
          )
        R = np.random.uniform(self.spaces['R']['low'], self.spaces['R']['high'])
        T = np.random.uniform(self.spaces['T']['low'], self.spaces['T']['high'])
        return Params(R=R, T=T, **kernels)



def compile_kernel_computer(SX: int, SY: int, nb_k: int)->t.Callable[Params, CompiledParams]:
    """return a jit compiled function taking as input lenia raw params and returning computed kernels (compiled params)
    
    Args:
        SX (int): Description
        SY (int): Description
        nb_k (int): Description
    
    Returns:
        t.Callable[Params, CompiledParams]: Description
    """
    mid = SX // 2
    def compute_kernels(params):
        """Compute kernels and return a dic containing fft kernels, T and R
        
        Args:
            params (TYPE): Description
        
        Returns:
            TYPE: Description
        """

        Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
              ((params.R+15) * params.r[k]) for k in range(nb_k) ]  # (x,y,k)
        K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params.a[k], params.w[k], params.b[k]) 
                        for k, D in zip(range(nb_k), Ds)])
        nK = K / jnp.sum(K, axis=(0,1), keepdims=True)
        fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))

        return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

    return jax.jit(compute_kernels)

@chex.dataclass
class L_State :
  A: jnp.ndarray

@chex.dataclass
class L_Config :
    SX: int
    SY: int
    nb_k: int
    C: int
    c0: t.Iterable
    c1: t.Iterable

class Lenia :

    #------------------------------------------------------------------------------

    def __init__(self, config: L_Config):

        self.config = config

        self.rule_space = None

        self.kernel_computer = None

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[t.Tuple[L_State, CompiledParams], L_State]:

        def step(state: L_State, params: CompiledParams)->L_State:
            """
            Main step
            A : state of the system (SX, SY, C)
            params : compiled paremeters (dict) must contain T, m, s, h and fK (computed kernels fft)
            """
            #---------------------------Original Lenia------------------------------------
            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([ U[:, :, c1[c]].sum(axis=-1) for c in range(C) ])  # (x,y,c)

            return jnp.clip(A + (1 / params.T) * U, 0., 1.)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self):

        def scan_step(carry: t.Tuple[L_State, CompiledParams], x)->t.Tuple[t.Tuple[L_State, CompiledParams], L_State]:
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: L_State, steps: int) -> t.Tuple[L_State, L_State]:
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout