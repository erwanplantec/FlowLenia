"""Summary
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import chex
from functools import partial
import typing as t

from flowlenia.utils import *
from flowlenia.reintegration_tracking import ReintegrationTracking

#==================================================================================================================
#==================================================PARAMETERS======================================================
#==================================================================================================================

@chex.dataclass
class Params:
    """Flow Lenia update rule parameters
    """
    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    R: float


@chex.dataclass
class CompiledParams:
    """Flow Lenia compiled parameters
    """
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray



class RuleSpace :

    """Rule space for Flow Lenia system
    
    Attributes:
        kernel_keys (TYPE): Description
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description
    """
    
    #-----------------------------------------------------------------------------
    def __init__(self, nb_k: int):
        """
        Args:
            nb_k (int): number of kernels in the update rule
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
        }
    #-----------------------------------------------------------------------------
    def sample(self, key: jnp.ndarray)->Params:
        """sample a random set of parameters
        
        Returns:
            Params: sampled parameters
        
        Args:
            key (jnp.ndarray): random generation key
        """
        kernels = {}
        for k in 'rmsh':
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'], 
              shape=(self.nb_k,)
            )
        for k in "awb":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'], 
              shape=(self.nb_k, 3)
            )
        R = jax.random.uniform(key=key, minval=self.spaces['R']['low'], maxval=self.spaces['R']['high'])
        return Params(R=R, **kernels)

class KernelComputer:

    """Summary
    
    Attributes:
        apply (Callable): main function transforming raw params (Params) in copmiled ones (CompiledParams)
        SX (int): X size
        SY (int): Y size
    """
    
    def __init__(self, SX: int, SY: int, nb_k: int):
        """Summary
        
        Args:
            SX (int): Description
            SY (int): Description
            nb_k (int): Description
        """
        self.SX = SX
        self.SY = SY

        mid = SX // 2
        def compute_kernels(params: Params)->CompiledParams:
            """Compute kernels and return a dic containing kernels fft
            
            Args:
                params (Params): raw params of the system
            
            Returns:
                CompiledParams: compiled params which can be used as update rule
            """

            Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
                  ((params.R+15) * params.r[k]) for k in range(nb_k) ]  # (x,y,k)
            K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params.a[k], params.w[k], params.b[k]) 
                            for k, D in zip(range(nb_k), Ds)])
            nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels 
            fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

            return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

        self.apply = jax.jit(compute_kernels)

    def __call__(self, params: Params):
        """callback to apply
        """
        return self.apply(params)


#==================================================================================================================
#==================================================FLOW LENIA======================================================
#==================================================================================================================

@chex.dataclass
class Config :

    """Configuration of Flow Lenia system
    """
    SX: int
    SY: int
    nb_k: int
    C: int
    c0: t.Iterable
    c1: t.Iterable
    dt: float 
    dd: int = 5
    sigma: float = .65
    n: int = 2
    theta_A : float = 1.
    border: str = 'wall'

@chex.dataclass
class State :

    """State of the system
    """
    A: jnp.ndarray

class FlowLenia :

    """class building the main functions of Flow Lenia
    
    Attributes:
        config (FL_Config): config of the system
        kernel_computer (KernelComputer): kernel computer
        rollout_fn (Callable): rollout function
        RT (ReintegrationTracking): Description
        rule_space (RuleSpace): Rule space of the system
        step_fn (Callable): system step function
    """
    
    #------------------------------------------------------------------------------

    def __init__(self, config: Config):
        """
        Args:
            config (Config): config of the system
        """
        self.config = config

        self.rule_space = RuleSpace(config.nb_k)

        self.kernel_computer = KernelComputer(self.config.SX, self.config.SY, self.config.nb_k)

        self.RT = ReintegrationTracking(self.config.SX, self.config.SY, self.config.dt, 
            self.config.dd, self.config.sigma, self.config.border)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def __call__(self, state: State, params: CompiledParams)->State:
        """callback to step function
        
        Args:
            state (State): Description
            params (CompiledParams): Description
        
        Returns:
            State: Description
        """
        return self.step_fn(state, params)

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[[State, CompiledParams], State]:
        """Build step function of the system according to config
        
        Returns:
            t.Callable[[State, CompiledParams], State]: step function which outputs next state 
            given a state and params
        """

        def step(state: State, params: CompiledParams)->State:
            """
            Main step
            
            Args:
                state (State): state of the system
                params (CompiledParams): params
            
            Returns:
                State: new state of the system
            
            """
            #---------------------------Original Lenia------------------------------------
            A = state.A

            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([ U[:, :, self.config.c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            #-------------------------------FLOW------------------------------------------

            nabla_U = sobel(U) #(x, y, 2, c)

            nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

            alpha = jnp.clip((A[:, :, None, :]/self.config.theta_A)**self.config.n, .0, 1.)

            F = nabla_U * (1 - alpha) - nabla_A * alpha

            nA = self.RT.apply(A, F)

            return State(A=nA)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self)->t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]:
        """build rollout function
        
        Returns:
            t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]: Description
        """
        def scan_step(carry: t.Tuple[State, CompiledParams], x)->t.Tuple[t.Tuple[State, CompiledParams], State]:
            """Summary
            
            Args:
                carry (t.Tuple[State, CompiledParams]): Description
                x (TYPE): Description
            
            Returns:
                t.Tuple[t.Tuple[State, CompiledParams], State]: Description
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: State, steps: int) -> t.Tuple[State, State]:
            """Summary
            
            Args:
                params (CompiledParams): Description
                init_state (State): Description
                steps (int): Description
            
            Returns:
                t.Tuple[State, State]: Description
            """
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout