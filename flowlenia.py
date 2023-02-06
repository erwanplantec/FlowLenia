"""Summary
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import chex
from functools import partial
import typing as t
from utils import *


#==================================================================================================================
#====================================================UTILS=========================================================
#==================================================================================================================

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


@chex.dataclass
class CompiledParams:

    """Summary
    """
    
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray



class Rule_space :

    """Rule space for Flow Lenia system
    
    Attributes:
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description
    
    Deleted Attributes:
        init_shape (TYPE): Description
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



def compile_kernel_computer(SX: int, SY: int, nb_k: int)->t.Callable[[Params], CompiledParams]:
    """return a jit compiled function taking as input lenia raw params and returning computed kernels (compiled params)
    
    Args:
        SX (int): size of world in X
        SY (int): size of world in Y
        nb_k (int): number of kernels
    
    Returns:
        t.Callable[Params, CompiledParams]: function to compile raw params
    """
    mid = SX // 2
    def compute_kernels(params: Params)->CompiledParams:
        """Compute kernels and return a dic containing fft kernels, T and R
        
        Args:
            params (Params): raw params of the system
        
        Returns:
            CompiledParams: compiled params which can be used in update rule
        """

        Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
              ((params.R+15) * params.r[k]) for k in range(nb_k) ]  # (x,y,k)
        K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params.a[k], params.w[k], params.b[k]) 
                        for k, D in zip(range(nb_k), Ds)])
        nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels 
        fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

        return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

    return jax.jit(compute_kernels)

#==================================================================================================================
#==================================================FLOW LENIA======================================================
#==================================================================================================================

@chex.dataclass
class FL_Config :

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
class FL_State :

    """State of the system
    """
    
    A: jnp.ndarray

class FlowLenia :

    """class building the main functions of Flow Lenia
    
    Attributes:
        config (FL_Config): config of the system
        kernel_computer (TYPE): kernel computer
        rollout_fn (TYPE): rollout function
        rule_space (TYPE): Rule space of the system
        step_fn (Callable): system step function
    """
    
    #------------------------------------------------------------------------------

    def __init__(self, config: FL_Config):
        """
        
        Args:
            config (FL_Config): config of the system
        """
        self.config = config

        self.rule_space = Rule_space(config.nb_k)

        self.kernel_computer = compile_kernel_computer(config.SX, config.SY, config.nb_k)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[[FL_State, CompiledParams], FL_State]:
        """Build step function of the system according to config
        
        Returns:
            t.Callable[t.Tuple[FL_State, CompiledParams], FL_State]: step function which outputs next state 
            given a state and params
        """
        x, y = jnp.arange(self.config.SX), jnp.arange(self.config.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)

        rolls = []
        rollxs = []
        rollys = []
        dd = self.config.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                rolls.append((dx, dy))
                rollxs.append(dx)
                rollys.append(dy)
        rollxs = jnp.array(rollxs)
        rollys = jnp.array(rollys)


        @partial(jax.vmap, in_axes = (0, 0, None, None))
        def step_flow(rollx: int, rolly: int, A: jnp.ndarray, mus: jnp.ndarray)->jnp.ndarray:
            """Computes quantity of matter arriving from neighbors at x + [rollx, rolly] for all
            xs in the system (all locations)
            
            Args:
                rollx (int): offset of neighbors in x direction
                rolly (int): offset of neighbors in y direction
                A (jnp.ndarray): state of the system (SX, SY, C)
                mus (jnp.ndarray): target locations of all cells (SX, SY, 2, C)
            
            Returns:
                jnp.ndarray: quantities of matter arriving to all cells from their respective neighbor (SX, SY, C)
            """
            rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
            rollmu = jnp.roll(mus, (rollx, rolly), axis = (0, 1))
            dpmu = jnp.min(jnp.stack(
                [jnp.absolute(pos[..., None] - (rollmu + jnp.array([di, dj])[None, None, :, None])) 
                for di in (-self.config.SX, 0, self.config.SX) for dj in (-self.config.SY, 0, self.config.SY)]
            ), axis = 0)
            sz = .5 - dpmu + self.config.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.config.sigma)) , axis = 2) / (4 * self.config.sigma**2)
            nA = rollA * area
            return nA

        def step(state: FL_State, params: CompiledParams)->FL_State:
            """
            Main step
            
            Args:
                state (FL_State): state of the system
                params (CompiledParams): params
            
            Returns:
                FL_State: new state of the system
            
            """
            #---------------------------Original Lenia------------------------------------
            A = state.A

            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([ U[:, :, self.config.c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            #-------------------------------FLOW------------------------------------------

            F = sobel(U) #(x, y, 2, c)

            C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

            alpha = jnp.clip((A[:, :, None, :]/self.config.theta_A)**self.config.n, .0, 1.)

            F = jnp.clip(F * (1 - alpha) - C_grad * alpha, 
                         -(self.config.dd-self.config.sigma), 
                         self.config.dd - self.config.sigma)

            ma = self.config.dd - self.config.sigma  # upper bound of the flow maggnitude
            mus = pos[..., None] + jnp.clip(self.config.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
            if self.config.border == "wall":
                mus = jnp.clip(mus, self.config.sigma, self.config.SX-self.config.sigma)
            nA = step_flow(rollxs, rollys, A, mus).sum(axis = 0)

            return FL_State(A=nA)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self)->t.Callable[[CompiledParams, FL_State, int], t.Tuple[FL_State, FL_State]]:
        """build rollout function
        
        Returns:
            t.Callable[t.Tuple[CompiledParams, FL_State, int], t.Tuple[FL_State, FL_State]]: Description
        """
        def scan_step(carry: t.Tuple[FL_State, CompiledParams], x)->t.Tuple[t.Tuple[FL_State, CompiledParams], FL_State]:
            """Summary
            
            Args:
                carry (t.Tuple[FL_State, CompiledParams]): Description
                x (TYPE): Description
            
            Returns:
                t.Tuple[t.Tuple[FL_State, CompiledParams], FL_State]: Description
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: FL_State, steps: int) -> t.Tuple[FL_State, FL_State]:
            """Summary
            
            Args:
                params (CompiledParams): Description
                init_state (FL_State): Description
                steps (int): Description
            
            Returns:
                t.Tuple[FL_State, FL_State]: Description
            """
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout


#==================================================================================================================
#============================================PARAMETER EMBEDDING===================================================
#==================================================================================================================

@chex.dataclass
class FLP_Config:

    """Summary
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
    theta_A: float = 1.
    border: str = 'wall'
    mix: str = 'stoch'

@chex.dataclass
class FLP_State:

    """state of system
    """
    
    A: jnp.ndarray
    P: jnp.ndarray

class FlowLeniaParams:

    """Flow Lenia system with parameters embedding
    
    Attributes:
        config (TYPE): config of the system
        kernel_computer (TYPE): -
        rollout_fn (TYPE): -
        rule_space (TYPE): -
        step_fn (TYPE): -
    """
    
    #------------------------------------------------------------------------------

    def __init__(self, config: FLP_Config):
        """_
        
        Args:
            config (FLP_Config): config of the system
        """
        
        self.config = config

        self.rule_space = Rule_space(config.nb_k)

        self.kernel_computer = compile_kernel_computer(config.SX, config.SY, config.nb_k)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[[FLP_State, CompiledParams], FLP_State]:
        """build step function
        
        Returns:
            t.Callable[t.Tuple[FLP_State, CompiledParams], FLP_State]: step function
        """
        
        x, y = jnp.arange(SX), jnp.arange(SY)
        X, Y = jnp.meshgrid(y, x)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)

        rolls = []
        rollxs = []
        rollys = []
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
              rolls.append((dx, dy))
              rollxs.append(dx)
              rollys.append(dy)
        rollxs = jnp.array(rollxs)
        rollys = jnp.array(rollys)

        SX, SY, dd, sigma, dt = (self.config.SX, self.config.SY, self.config.dd,
                             self.config.sigma, self.config.dt)

        @partial(jax.vmap, in_axes = (0, 0, None, None, None))
        def step_flow(rollx: int, rolly: int, A: jnp.ndarray, P: jnp.ndarray, 
                      mus: jnp.ndarray)->t.Tuple[jnp.ndarray, jnp.ndarray]:
            """Summary
            
            Args:
                rollx (int): offset on x axis
                rolly (int): offset on y axis
                A (jnp.ndarray): state of system
                P (jnp.ndarray): parameter map
                mus (jnp.ndarray): target locations of cells (x + dt * F(x))
            
            Returns:
                t.Tuple[jnp.ndarray, jnp.ndarray]: Description
            """
            rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
            rollP = jnp.roll(P, (rollx, rolly), axis = (0, 1)) #(x, y, k)
            rollmu = jnp.roll(mus, (rollx, rolly), axis = (0, 1))

            dpmu = jnp.min(jnp.stack(
                [jnp.absolute(pos[..., None] - (rollmu + jnp.array([di, dj])[None, None, :, None])) 
                for di in (-SX, 0, SX) for dj in (-SY, 0, SY)]
            ), axis = 0)

            #dpmu = jnp.absolute(pos[..., None] - rollmu)
            sz = .5 - dpmu + sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*sigma)) , axis = 2) / (4 * sigma**2)
            nA = rollA * area
            return nA, rollP

        def step(state: FLP_State, params: CompiledParams)->FLP_State:
            """
            Main step
            
            
            Args:
                state (FLP_State): state of the system where A are actications and P is the paramter map
                params (CompiledParams): compiled params of update rule
            
            Returns:
                FLP_State: new state of the systems
            """
            A, P = state.A, state.P
            #---------------------------Original Lenia------------------------------------
            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * P # (x,y,k)

            U = jnp.dstack([ U[:, :, c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            #-------------------------------FLOW------------------------------------------

            F = sobel(U) #(x, y, 2, c) : Flow

            C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient

            alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)

            F = jnp.clip(F * (1 - alpha) - C_grad * alpha, - (dd-sigma), dd - sigma)

            mus = pos[..., None] + dt * F #(x, y, 2, c) : target positions (distribution centers)

            nA, nP = step_flow(rollxs, rollys, A, P, mus) #vmapped

            if mix == 'avg':
                nP = jnp.sum(nP * nA.sum(axis = -1, keepdims = True), axis = 0)  
                nA = jnp.sum(nA, axis = 0)
                nP = nP / (nA.sum(axis = -1, keepdims = True)+1e-10)

            elif mix == "softmax":
                expnA = jnp.exp(nA.sum(axis = -1, keepdims = True)) - 1
                nA = jnp.sum(nA, axis = 0)
                nP = jnp.sum(nP * expnA, axis = 0) / (expnA.sum(axis = 0)+1e-10) #avg rule

            elif mix == "stoch":
                categorical=jax.random.categorical(
                  jax.random.PRNGKey(42), 
                  jnp.log(nA.sum(axis = -1, keepdims = True)), 
                  axis=0)
                mask=jax.nn.one_hot(categorical,num_classes=(2*dd+1)**2,axis=-1)
                mask=jnp.transpose(mask,(3,0,1,2)) 
                nP = jnp.sum(nP * mask, axis = 0)
                nA = jnp.sum(nA, axis = 0)

            elif mix == "stoch_gene_wise":
                mask = jnp.concatenate(
                  [jax.nn.one_hot(jax.random.categorical(
                                                        jax.random.PRNGKey(42), 
                                                        jnp.log(nA.sum(axis = -1, keepdims = True)), 
                                                        axis=0),
                                  num_classes=(2*dd+1)**2,axis=-1)
                  for _ in range(self.config.nb_k)], 
                  axis = 2)
                mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
                nP = jnp.sum(nP * mask, axis = 0)
                nA = jnp.sum(nA, axis = 0)

            return FLP_State(A=nA, P=nP)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self):
        """build a rollout function taking as input params, an initial state and a number of steps
        and returning the final state of the system and the stacked states
        
        Returns:
            Callable
        """
        def scan_step(carry: t.Tuple[FLP_State, CompiledParams], x)->t.Tuple[t.Tuple[FLP_State, CompiledParams], FLP_State]:
            """Summary
            
            Args:
                carry (t.Tuple[FLP_State, CompiledParams]): state of the system [state x params]
                x: None
            
            Returns:
                t.Tuple[t.Tuple[FLP_State, CompiledParams], FLP_State]: rollout function
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: FLP_State, steps: int) -> t.Tuple[FLP_State, FLP_State]:
            """Summary
            
            Args:
                params (CompiledParams): compiled params of the systems
                init_state (FLP_State): initial state of the system
                steps (int): number of steps to simulate
            
            Returns:
                t.Tuple[FLP_State, FLP_State]: returns the final state and the stacked states of the rollout
            """
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout