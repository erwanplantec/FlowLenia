"""Interactive web interface for the Flow Lenia models.

    uv run python -m webui.server            # then open http://127.0.0.1:8000
    uv run python -m webui.server --reload   # restart automatically when webui/ or flowlenia/ files change

One simulation per browser tab. The page sends JSON commands over a websocket
(build, play, brush, ...) and receives raw RGBA frames (binary messages) plus
JSON status/stats messages. All JAX work runs in a worker thread so the event
loop stays responsive.
"""
import asyncio
import dataclasses
import pathlib
import time
import traceback
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from flowlenia.flowlenia import FlowLenia, Config as ConfigFL
from flowlenia.flowlenia_params import FlowLeniaParams, Config as ConfigFLP
from flowlenia.grn_flowlenia import GRNFlowLenia, Config as ConfigGRN
from flowlenia.utils import conn_from_matrix

STATIC = pathlib.Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# hidden: name of the per-cell hidden state field transported with matter (None if absent)
MODELS: dict[str, dict[str, Any]] = {
    "FlowLenia":       dict(cls=FlowLenia,       cfg=ConfigFL,  hidden=None, hidden_label=None),
    "GRNFlowLenia":    dict(cls=GRNFlowLenia,    cfg=ConfigGRN, hidden="S",  hidden_label="gene"),
    "FlowLeniaParams": dict(cls=FlowLeniaParams, cfg=ConfigFLP, hidden="P",  hidden_label="param"),
}
# connectivity fields are derived from the matrix, not edited directly
DERIVED = {"C", "k", "c0", "c1"}
CHOICES = {
    "update_mode": ["rt", "mace"],
    "border": ["wall", "torus"],
    "mix_rule": ["stoch", "stoch_gene_wise", "avg", "softmax"],
}


def config_fields(cfg_cls) -> list[dict]:
    """Editable (scalar) fields of a Config class, for both dataclass and NamedTuple configs."""
    if dataclasses.is_dataclass(cfg_cls):
        items = []
        for f in dataclasses.fields(cfg_cls):
            if f.default is not dataclasses.MISSING:
                items.append((f.name, f.type, f.default))
    else:  # NamedTuple
        items = [(n, cfg_cls.__annotations__[n], cfg_cls._field_defaults[n]) for n in cfg_cls._fields]
    out = []
    for name, tp, default in items:
        if name in DERIVED or tp not in (int, float, str, bool):
            continue
        out.append(dict(name=name, type=tp.__name__, default=default, choices=CHOICES.get(name)))
    return out


def schema() -> dict:
    return {name: dict(fields=config_fields(m["cfg"]), hidden=m["hidden"], hidden_label=m["hidden_label"])
            for name, m in MODELS.items()}


def replace(state, **kw):
    """Works for both flax PyTreeNode (.replace) and NamedTuple (._replace) states."""
    return state._replace(**kw) if isinstance(state, tuple) else state.replace(**kw)


VIRIDIS = jnp.asarray((matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8))

# ---------------------------------------------------------------------------
# Jitted array functions (model independent)
# ---------------------------------------------------------------------------

def _to_rgba(rgb: jax.Array) -> jax.Array:
    rgb = (jnp.clip(rgb, 0., 1.) * 255).astype(jnp.uint8)
    return jnp.concatenate([rgb, jnp.full(rgb.shape[:2] + (1,), 255, jnp.uint8)], -1)


def _lut(x: jax.Array, lo: float, hi: float) -> jax.Array:
    idx = jnp.clip((x - lo) / (hi - lo + 1e-12) * 255, 0, 255).astype(jnp.int32)
    return VIRIDIS[idx].astype(jnp.float32) / 255.


@partial(jax.jit, static_argnames=("view",))
def render(A: jax.Array, H: jax.Array | None, view: str, ch: int, thr: float) -> jax.Array:
    mass = A.sum(-1)
    occ = (mass > thr)[..., None]
    if view == "matter":  # matter channels as RGB
        C = A.shape[-1]
        rgb = (jnp.repeat(A, 3, -1) if C == 1 else
               jnp.stack([A[..., 0], A[..., 0], A[..., 1]], -1) if C == 2 else A[..., :3])
    elif view == "mass":  # total matter, viridis
        rgb = _lut(mass, 0., jnp.maximum(mass.max(), 1e-6)) * occ
    elif view == "hidden_rgb":  # matter colored by first 3 hidden channels
        assert H is not None
        Hc = H[..., :3]
        Hc = jnp.pad(Hc, ((0, 0), (0, 0), (0, 3 - Hc.shape[-1])))
        rgb = jnp.clip(mass, 0., 1.)[..., None] * Hc
    elif view == "hidden_channel":  # single hidden channel, viridis, black where no matter
        assert H is not None
        h = H[..., ch]
        rgb = _lut(h, 0., jnp.maximum(1., h.max())) * occ
    else:
        raise ValueError(view)
    return _to_rgba(rgb)


def _disk(shape, cx, cy, r):
    ys, xs = jnp.mgrid[:shape[0], :shape[1]]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2)[..., None]  # (X, Y, 1)


@partial(jax.jit, static_argnames=("tool",))
def brush(A, H, key, tool: str, cx, cy, r, amount, channel, hidx, value, seed_hidden, thr):
    """Apply a brush stroke. channel < 0 means all matter channels; hidx < 0 means all hidden channels."""
    disk = _disk(A.shape, cx, cy, r)
    kA, kH = jr.split(key)
    if tool == "add":
        chm = jnp.where(channel < 0, 1., (jnp.arange(A.shape[-1]) == channel).astype(A.dtype))
        noise = jr.uniform(kA, A.shape, minval=0.5, maxval=1.)
        newA = A + disk * chm * amount * noise
        if H is not None:  # give hidden state to cells that were empty
            was_empty = (A.sum(-1, keepdims=True) <= thr) & disk
            H = jnp.where(was_empty & seed_hidden, jr.uniform(kH, H.shape), H)
        A = newA
    elif tool == "erase":
        A = A * (1 - disk)
        if H is not None:
            H = H * (1 - disk)
    elif tool in ("set_hidden", "random_hidden") and H is not None:
        occ = (A.sum(-1, keepdims=True) > thr) & disk
        hm = jnp.where(hidx < 0, True, jnp.arange(H.shape[-1]) == hidx)
        new = jnp.full_like(H, value) if tool == "set_hidden" else jr.uniform(kH, H.shape)
        H = jnp.where(occ & hm, new, H)
    return A, H


@jax.jit
def hold(mask, val, cx, cy, r, hidx, value):
    sel = _disk(mask.shape, cx, cy, r) & jnp.where(hidx < 0, True, jnp.arange(mask.shape[-1]) == hidx)
    return mask | sel, jnp.where(sel, value, val)


@jax.jit
def unhold(mask, cx, cy, r):
    return mask & ~_disk(mask.shape, cx, cy, r)


@jax.jit
def stats(A, H, thr):
    occ = A.sum(-1) > thr
    n = jnp.maximum(occ.sum(), 1)
    hmean = None if H is None else (H * occ[..., None]).sum((0, 1)) / n
    return A.sum(), occ.sum(), hmean


# ---------------------------------------------------------------------------
# Simulation session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self):
        self.model = None
        self.state = None
        self.key = jr.key(0)
        self.t = 0
        self.view, self.view_ch = "matter", 0

    # --- build -------------------------------------------------------------
    def build(self, name: str, values: dict, matrix: list[list[int]], param_seed: int, state_seed: int, patch: int):
        spec = MODELS[name]
        fields = {f["name"]: f for f in config_fields(spec["cfg"])}
        cast = {"int": int, "float": float, "str": str, "bool": lambda v: v in (True, "true", "on", 1)}
        kw = {k: cast[fields[k]["type"]](v) for k, v in values.items() if k in fields}
        M = np.asarray(matrix, dtype=int)
        assert M.ndim == 2 and M.shape[0] == M.shape[1] and M.sum() > 0, "connectivity matrix must be square and non-empty"
        c0, c1 = conn_from_matrix(M)
        cfg = spec["cfg"](C=M.shape[0], k=int(M.sum()), c0=c0, c1=c1, **kw)
        self.name, self.cfg, self.hidden = name, cfg, spec["hidden"]
        self.thr = float(getattr(cfg, "matter_thr", 1e-3))
        self.key = jr.key(state_seed)
        self.param_seed = param_seed
        self.model = spec["cls"](cfg, key=jr.key(param_seed))
        self._step_n = eqx.filter_jit(self._make_step(spec["hidden"], self.thr))
        self.reset(state_seed, patch)
        t0 = time.time()
        self.step(1)  # trigger compilation (and surface model errors now)
        self.frame()
        self.reset(state_seed, patch)
        return time.time() - t0

    def resample(self, param_seed: int, keep_state: bool, state_seed: int, patch: int):
        """Sample new model parameters with the same config. Optionally keep the current matter / hidden state."""
        old, old_H, t = self.state, self.H, self.t
        self.param_seed = param_seed
        self.model = MODELS[self.name]["cls"](self.cfg, key=jr.key(param_seed))
        t0 = time.time()
        self.reset(state_seed, patch)  # fresh state with the new kernels (fK)
        if keep_state:
            self._set(old.A, old_H)
            self.t = t
        self.step(0)  # compile for the new model before handing control back
        return time.time() - t0

    @staticmethod
    def _make_step(hidden: str | None, thr: float):
        def step_n(model, state, key, n, hold_mask, hold_val):
            def body(_, c):
                s, k = c
                k, k_ = jr.split(k)
                s = model(s, k_)
                if hidden is not None:  # re-impose held hidden values on occupied cells
                    occ = s.A.sum(-1, keepdims=True) > thr
                    s = replace(s, **{hidden: jnp.where(hold_mask & occ, hold_val, getattr(s, hidden))})
                return s, k
            return jax.lax.fori_loop(0, n, body, (state, key))
        return step_n

    # --- state ops ---------------------------------------------------------
    @property
    def H(self):
        return None if self.hidden is None else getattr(self.state, self.hidden)

    def _set(self, A, H):
        kw = dict(A=A)
        if self.hidden is not None:
            kw[self.hidden] = H
        self.state = replace(self.state, **kw)

    def reset(self, seed: int, patch: int):
        cfg = self.cfg
        self.state = self.model.initialize(jr.key(seed))
        k1, k2 = jr.split(jr.key(seed + 1))
        p = min(patch, cfg.X, cfg.Y)
        x0, y0 = (cfg.X - p) // 2, (cfg.Y - p) // 2
        A = self.state.A.at[x0:x0+p, y0:y0+p].set(jr.uniform(k1, (p, p, cfg.C)))
        H = self.H
        if H is not None:
            H = H.at[x0:x0+p, y0:y0+p].set(jr.uniform(k2, (p, p, H.shape[-1])))
        self._set(A, H)
        self.t = 0
        self.release()

    def release(self):
        """Drop all held hidden values."""
        shape = self.state.A.shape[:2] + ((self.H.shape[-1],) if self.H is not None else (1,))
        self.hold_mask, self.hold_val = jnp.zeros(shape, bool), jnp.zeros(shape)

    def clear(self):
        self._set(jnp.zeros_like(self.state.A), None if self.H is None else jnp.zeros_like(self.H))

    def step(self, n: int):
        self.state, self.key = self._step_n(self.model, self.state, self.key, jnp.int32(n),
                                            self.hold_mask, self.hold_val)
        jax.block_until_ready(self.state.A)
        self.t += n

    def brush(self, m: dict):
        self.key, k = jr.split(self.key)
        A, H = brush(self.state.A, self.H, k, m["tool"], m["x"], m["y"], m["radius"],
                     m.get("amount", 0.5), m.get("channel", -1), m.get("hidx", -1),
                     m.get("value", 0.5), bool(m.get("seed_hidden", True)), self.thr)
        self._set(A, H)
        if m["tool"] == "set_hidden" and self.H is not None and m.get("hold"):
            self.hold_mask, self.hold_val = hold(self.hold_mask, self.hold_val, m["x"], m["y"], m["radius"],
                                                 m.get("hidx", -1), m.get("value", 0.5))
        elif m["tool"] == "unhold":
            self.hold_mask = unhold(self.hold_mask, m["x"], m["y"], m["radius"])

    def frame(self) -> bytes:
        view = self.view if (self.H is not None or not self.view.startswith("hidden")) else "matter"
        ch = min(self.view_ch, self.H.shape[-1] - 1) if self.H is not None else 0
        return np.asarray(render(self.state.A, self.H, view, ch, self.thr)).tobytes()

    def stats(self) -> dict:
        mass, occ, hmean = stats(self.state.A, self.H, self.thr)
        return dict(type="stats", step=self.t, mass=float(mass), occupied=int(occ), held=int(self.hold_mask.any(-1).sum()),
                    hidden_mean=None if hmean is None else np.asarray(hmean).round(4).tolist())

    def info(self, compile_s: float) -> dict:
        return dict(type="built", model=self.name, param_seed=self.param_seed, X=self.cfg.X, Y=self.cfg.Y, C=self.cfg.C, k=self.cfg.k,
                    hidden=self.hidden, hidden_dim=None if self.H is None else int(self.H.shape[-1]),
                    compile_s=round(compile_s, 2))


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session()
    queue: asyncio.Queue = asyncio.Queue()
    running, spf, fps = False, 1, 30
    dirty = False
    await ws.send_json(dict(type="schema", models=schema()))

    async def receiver():
        try:
            while True:
                await queue.put(await ws.receive_json())
        except (WebSocketDisconnect, RuntimeError):
            await queue.put(None)

    recv_task = asyncio.create_task(receiver())

    async def error(e: Exception):
        nonlocal running
        running = False
        traceback.print_exc()
        await ws.send_json(dict(type="error", msg=f"{type(e).__name__}: {e}"))
        await ws.send_json(dict(type="running", value=False))

    async def handle(m: dict):
        nonlocal running, spf, fps, dirty
        t = m["type"]
        if t == "build":
            running = False
            await ws.send_json(dict(type="running", value=False))
            await ws.send_json(dict(type="status", msg="building + compiling…"))
            sess.model = None
            dt = await asyncio.to_thread(sess.build, m["model"], m["config"], m["matrix"],
                                         int(m.get("param_seed", 0)), int(m.get("state_seed", 0)),
                                         int(m.get("patch", 40)))
            await ws.send_json(sess.info(dt))
        elif t == "resample" and sess.model is not None:
            dt = await asyncio.to_thread(sess.resample, int(m["param_seed"]), bool(m.get("keep_state", True)),
                                         int(m.get("state_seed", 0)), int(m.get("patch", 40)))
            await ws.send_json(sess.info(dt))
        elif t == "set":
            spf, fps = max(1, int(m.get("spf", spf))), max(1, int(m.get("fps", fps)))
        elif t == "view":
            sess.view, sess.view_ch = m["view"], int(m.get("channel", 0))
        elif sess.model is None:
            return
        elif t in ("play", "pause"):
            running = t == "play"
            await ws.send_json(dict(type="running", value=running))
        elif t == "step":
            await asyncio.to_thread(sess.step, int(m.get("n", 1)))
        elif t == "reset":
            await asyncio.to_thread(sess.reset, int(m.get("state_seed", 0)), int(m.get("patch", 40)))
        elif t == "clear":
            await asyncio.to_thread(sess.clear)
        elif t == "release":
            await asyncio.to_thread(sess.release)
        elif t == "brush":
            await asyncio.to_thread(sess.brush, m)
        dirty = True

    last_stats, last_t, sps = 0., (time.time(), 0), 0.
    try:
        while True:
            t0 = time.time()
            # --- apply all pending commands
            while not queue.empty():
                m = queue.get_nowait()
                if m is None:
                    return
                try:
                    await handle(m)
                except Exception as e:
                    await error(e)
            # --- simulate
            if running and sess.model is not None:
                try:
                    await asyncio.to_thread(sess.step, spf)
                    dirty = True
                except Exception as e:
                    await error(e)
            # --- send frame + stats
            if dirty and sess.model is not None and sess.state is not None:
                dirty = False
                try:
                    await ws.send_bytes(await asyncio.to_thread(sess.frame))
                    if not running or time.time() - last_stats > 0.2:
                        now = time.time()
                        sps = (sess.t - last_t[1]) / max(now - last_t[0], 1e-6) if running else 0.
                        last_t, last_stats = (now, sess.t), now
                        await ws.send_json(sess.stats() | dict(sps=round(sps, 1)))
                except Exception as e:
                    await error(e)
            # --- pace: while running, wait out the rest of the frame (commands queue up meanwhile);
            #     while paused, block until the next command arrives
            if running:
                await asyncio.sleep(max(1 / fps - (time.time() - t0), 0.))
            else:
                m = await queue.get()
                if m is None:
                    return
                try:
                    await handle(m)
                except Exception as e:
                    await error(e)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    import argparse
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart when webui/ or flowlenia/ files change")
    a = p.parse_args()
    root = pathlib.Path(__file__).parent.parent
    uvicorn.run("webui.server:app", host=a.host, port=a.port, reload=a.reload,
                reload_dirs=[str(root / "webui"), str(root / "flowlenia")] if a.reload else None,
                reload_includes=["*.py", "*.html"] if a.reload else None)
