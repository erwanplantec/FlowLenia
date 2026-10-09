"""Wrapper adding a learnable initial pattern (seed) to a Flow Lenia model."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class Seeded(eqx.Module):
    """Model + learnable initial patch.

    A0 (p, p, C) is the initial matter patch, H0 (p, p, G) the initial hidden state patch (None for
    models without hidden state). Both are array leaves, so they are trained along with the model
    parameters. `initialize` places the patch at the grid center; matter is clipped to >= 0 and
    rescaled to total mass `mass`, hidden values are clipped to [0, 1].
    Stepping is delegated to the wrapped model.
    """
    model: eqx.Module
    A0: jax.Array
    H0: jax.Array | None
    hidden: str | None = eqx.field(static=True)
    mass: float = eqx.field(static=True)

    def __init__(self, model: eqx.Module, hidden: str | None, patch: int, mass: float, *, key: jax.Array):
        cfg = model.cfg
        p = min(patch, cfg.X, cfg.Y)
        kA, kH = jr.split(key)
        self.model, self.hidden, self.mass = model, hidden, float(mass)
        self.A0 = jr.uniform(kA, (p, p, cfg.C))
        if hidden is None:
            self.H0 = None
        else:
            G = getattr(model.initialize(kH), hidden).shape[-1]
            self.H0 = jr.uniform(kH, (p, p, G))

    def __call__(self, state, key: jax.Array):
        return self.model(state, key)

    @property
    def cfg(self):
        return self.model.cfg

    def seed_patch(self, key: jax.Array | None = None, noise: float = 0.):
        """Matter and hidden patches as they are placed in the grid (optionally perturbed by noise)."""
        A0 = jnp.clip(self.A0, 0.)
        H0 = None if self.H0 is None else self.H0
        if noise > 0. and key is not None:
            kA, kH = jr.split(key)
            A0 = jnp.clip(A0 * (1 + noise * jr.normal(kA, A0.shape)), 0.)
            if H0 is not None:
                H0 = H0 + noise * jr.normal(kH, H0.shape)
        A0 = A0 * self.mass / (A0.sum() + 1e-8)
        return A0, None if H0 is None else jnp.clip(H0, 0., 1.)

    def initialize(self, key: jax.Array, noise: float = 0.):
        """Initial state with the seed at the center. Calls model.initialize, so the kernels (fK)
        stay differentiable wrt the model parameters."""
        cfg = self.cfg
        s = self.model.initialize(key)
        A0, H0 = self.seed_patch(key, noise)
        p = A0.shape[0]
        x0, y0 = (cfg.X - p) // 2, (cfg.Y - p) // 2
        kw = dict(A=s.A.at[x0:x0 + p, y0:y0 + p].set(A0))
        if self.hidden is not None:
            kw[self.hidden] = getattr(s, self.hidden).at[x0:x0 + p, y0:y0 + p].set(H0)
        return s.replace(**kw)
