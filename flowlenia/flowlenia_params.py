import jax
import jax.numpy as jnp
import chex
from functools import partial
import typing as t

from flowlenia.flowlenia import (RuleSpace, KernelComputer, Config as FL_Config, State as FL_State,
	Params, CompiledParams)
from flowlenia.reintegration_tracking import ReintegrationTracking
from flowlenia.utils import *

@chex.dataclass
class Config(FL_Config):

    """Summary
    """
    mix: str = 'stoch'

@chex.dataclass
class State(FL_State):

    """state of system
    """
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

    def __init__(self, config: Config):
        """_
        
        Args:
            config (Config): config of the system
        """
        
        self.config = config

        self.rule_space = RuleSpace(config.nb_k)

        self.kernel_computer = KernelComputer(config.SX, config.SY, config.nb_k)

        self.RT = ReintegrationTracking(self.config.SX, self.config.SY, self.config.dt, 
            self.config.dd, self.config.sigma, self.config.border, has_hidden=True,
            hidden_dims=self.config.nb_k, mix=self.config.mix)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[[State, CompiledParams], State]:
        """build step function
        
        Returns:
            t.Callable[[State, CompiledParams], State]: step function
        """
        

        SX, SY, dd, sigma, dt = (self.config.SX, self.config.SY, self.config.dd,
                             self.config.sigma, self.config.dt)


        def step(state: State, params: CompiledParams)->State:
            """
            Main step
            
            
            Args:
                state (State): state of the system where A are actications and P is the paramter map
                params (CompiledParams): compiled params of update rule
            
            Returns:
                State: new state of the systems
            """
            A, P = state.A, state.P
            #---------------------------Original Lenia------------------------------------
            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * P # (x,y,k)

            U = jnp.dstack([ U[:, :, self.config.c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            #-------------------------------FLOW------------------------------------------

            F = sobel(U) #(x, y, 2, c) : Flow

            C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient

            alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)

            F = jnp.clip(F * (1 - alpha) - C_grad * alpha, - (dd-sigma), dd - sigma)

            nA, nP = self.RT.apply(A, P, F)

            return State(A=nA, P=nP)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self):
        """build a rollout function taking as input params, an initial state and a number of steps
        and returning the final state of the system and the stacked states
        
        Returns:
            Callable
        """
        def scan_step(carry: t.Tuple[State, CompiledParams], x)->t.Tuple[t.Tuple[State, CompiledParams], State]:
            """Summary
            
            Args:
                carry (t.Tuple[State, CompiledParams]): state of the system [state x params]
                x: None
            
            Returns:
                t.Tuple[t.Tuple[State, CompiledParams], State]: rollout function
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: State, steps: int) -> t.Tuple[State, State]:
            """Summary
            
            Args:
                params (CompiledParams): compiled params of the systems
                init_state (State): initial state of the system
                steps (int): number of steps to simulate
            
            Returns:
                t.Tuple[State, State]: returns the final state and the stacked states of the rollout
            """
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout