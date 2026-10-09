"""Train a Flow Lenia model with gradient descent (backprop through the rollout) to reproduce a target pattern.

    uv run python -m training.train training/configs/gecko.yml
    uv run python -m training.train training/configs/gecko.yml --set training.iterations=200 target.name=star
    uv run python -m training.train --list-targets

Every option has a default (DEFAULTS below); the YAML file only needs to override what changes.
Outputs go to `out_dir`: config.yml (resolved), model.eqx (checkpoint), losses.json, loss.png,
target.png, final.png (target vs result) and rollout.mp4 (or .gif if ffmpeg is missing).
Load a trained model with `load_run(out_dir)`.
"""
import argparse
import copy
import dataclasses
import json
import pathlib
import shutil
import subprocess
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

from flowlenia.flowlenia import FlowLenia, Config as ConfigFL
from flowlenia.grn_flowlenia import GRNFlowLenia, Config as ConfigGRN
from flowlenia.utils import conn_from_matrix
from training.seeded import Seeded
from training.targets import available, load_target

# model name -> (class, config class, hidden state field or None)
MODELS = {
    "FlowLenia": (FlowLenia, ConfigFL, None),
    "GRNFlowLenia": (GRNFlowLenia, ConfigGRN, "S"),
}

DEFAULTS = {
    "seed": 0,
    "out_dir": "runs/default",
    "target": {
        "name": "gecko",     # see training/targets.py or --list-targets
        "size": 40,          # target is fitted in a size x size box, centered in the grid
        "density": 1.0,      # target matter density where target alpha = 1
        "color": True,       # also fit the first 3 hidden channels to the target RGB (hidden-state models only)
    },
    "model": {
        "name": "GRNFlowLenia",
        "matrix": [[10]],    # kernel connectivity; sets C and k
        "param_seed": 0,     # initial parameters
        "config": {"X": 64, "Y": 64, "update_mode": "mace", "mace_temp": 0.5, "growth": False},
    },
    "init": {
        "patch": 24,         # size of the square initial patch (seed); its mass is rescaled to the target's total mass
        "learn": True,       # optimize the seed (matter + hidden state pixels) along with the model parameters
        "noise": 0.0,        # relative noise added to the seed for each training sample (robustness)
    },
    "training": {
        "iterations": 1000,
        "batch_size": 4,
        "steps": [60, 90],   # each sample is evaluated at a step drawn uniformly from this range
        "loss_window": 5,    # loss is averaged over the last loss_window steps before that step
        "lr": 1e-3,
        "lr_end": 1e-4,      # cosine decay from lr to lr_end
        "grad_clip": 1.0,    # global gradient norm clipping
        "color_weight": 1.0,
        "remat": True,       # recompute steps in the backward pass (much less memory, ~30% slower)
        "log_every": 10,
        "save_every": 100,
    },
    # model parameters are clipped back into these ranges after every update ("a.b" = nested field).
    # The seed is always kept in range: matter >= 0, hidden state in [0, 1].
    "clamp": {
        "R": [2.0, 25.0], "r": [0.05, 1.0], "m": [0.0, 1.0], "s": [0.001, 0.5], "h": [0.0, 1.0],
        "a": [0.0, 1.0], "b": [0.001, 1.0], "w": [0.01, 0.5], "grn.tau": [0.05, 10.0],
    },
    "eval": {"steps": 300, "fps": 30, "seed": 12345},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """--set a.b.c=value (value parsed as YAML)."""
    for s in sets:
        path, val = s.split("=", 1)
        d = cfg
        *keys, last = path.split(".")
        for k in keys:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


# ---------------------------------------------------------------------------
# Model / state
# ---------------------------------------------------------------------------

def build_model(mcfg: dict):
    cls, cfg_cls, hidden = MODELS[mcfg["name"]]
    M = np.asarray(mcfg["matrix"], dtype=int)
    c0, c1 = conn_from_matrix(M)
    known = {f.name for f in dataclasses.fields(cfg_cls)} if dataclasses.is_dataclass(cfg_cls) else set(cfg_cls._fields)
    kw = {k: v for k, v in mcfg["config"].items() if k in known}
    if ignored := sorted(set(mcfg["config"]) - known):
        print(f"warning: {mcfg['name']} config has no field(s) {ignored}, ignored")
    cfg = cfg_cls(C=M.shape[0], k=int(M.sum()), c0=c0, c1=c1, **kw)
    return cls(cfg, key=jr.key(mcfg["param_seed"])), cfg, hidden


def build_seeded(rc: dict, mass: float):
    """Model wrapped with a learnable seed (see training/seeded.py)."""
    model, cfg, hidden = build_model(rc["model"])
    return Seeded(model, hidden, rc["init"]["patch"], mass, key=jr.key(rc["seed"])), cfg, hidden


def make_clamp(spec: dict):
    """Clamp model parameters (paths relative to the wrapped model) and the seed into valid ranges."""
    spec = {f"model.{k}": v for k, v in spec.items()} | {"A0": [0., None], "H0": [0., 1.]}

    def clamp(diff):
        for name, (lo, hi) in spec.items():
            get = lambda m, name=name: _getpath(m, name)
            try:
                leaf = get(diff)
            except AttributeError:
                continue
            if leaf is None:
                continue
            diff = eqx.tree_at(get, diff, jnp.clip(leaf, lo, hi))
        return diff
    return clamp


def _getpath(obj, path: str):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


# ---------------------------------------------------------------------------
# Rendering / output helpers
# ---------------------------------------------------------------------------

def render(A: np.ndarray, H: np.ndarray | None, density: float, color: bool) -> np.ndarray:
    """uint8 RGB image: matter colored by the first 3 hidden channels (or grayscale matter)."""
    occ = np.clip(A.sum(-1) / density, 0., 1.)[..., None]
    rgb = occ * np.clip(H[..., :3], 0., 1.) if (H is not None and color) else np.repeat(occ, 3, -1)
    return (rgb * 255).astype(np.uint8)


def target_image(target: np.ndarray, color: bool) -> np.ndarray:
    a = target[..., 3:]
    rgb = a * target[..., :3] if color else np.repeat(a, 3, -1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def upscale(img: np.ndarray, f: int) -> np.ndarray:
    return img.repeat(f, 0).repeat(f, 1)


def save_video(frames: np.ndarray, path: pathlib.Path, fps: int) -> pathlib.Path:
    from PIL import Image
    h, w = frames.shape[1:3]
    frames = frames[:, :h // 2 * 2, :w // 2 * 2]
    if shutil.which("ffmpeg") is None:
        path = path.with_suffix(".gif")
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=1000 // fps, loop=0)
        return path
    h, w = frames.shape[1:3]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(path)], input=frames.tobytes(), check=True)
    return path


def load_run(run_dir: str | pathlib.Path):
    """Rebuild a trained model from a run directory. Returns (seeded model, cfg, hidden, run_config).
    `seeded.initialize(key)` gives the learned initial state, `seeded(state, key)` steps it,
    `seeded.model` is the bare Flow Lenia model."""
    run_dir = pathlib.Path(run_dir)
    rc = yaml.safe_load((run_dir / "config.yml").read_text())
    seeded, cfg, hidden = build_seeded(rc, rc["derived"]["mass"])
    seeded = eqx.tree_deserialise_leaves(run_dir / "model.eqx", seeded)
    return seeded, cfg, hidden, rc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="YAML config file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries, e.g. training.lr=3e-4")
    ap.add_argument("--list-targets", action="store_true")
    args = ap.parse_args()
    if args.list_targets:
        print("\n".join(available()))
        return

    rc = merge(DEFAULTS, yaml.safe_load(open(args.config)) if args.config else {})
    rc = apply_overrides(rc, args.set)
    tc, tr = rc["target"], rc["training"]
    out = pathlib.Path(rc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # --- target, model
    mcfg = rc["model"]["config"]
    X, Y = mcfg.get("X", 128), mcfg.get("Y", 128)
    target_np = load_target(tc["name"], tc["size"], X, Y)
    target = jnp.asarray(target_np)
    t_mass = target[..., 3] * tc["density"]                      # (X, Y) target matter
    t_rgb = target[..., 3:] * target[..., :3]                    # (X, Y, 3) premultiplied color
    mass = float(t_mass.sum())
    rc["derived"] = {"mass": mass}  # needed to rebuild the model in load_run
    (out / "config.yml").write_text(yaml.safe_dump(rc, sort_keys=False))
    model, cfg, hidden = build_seeded(rc, mass)
    color = bool(tc["color"]) and hidden is not None
    norm_mass, norm_rgb = float(jnp.mean(t_mass ** 2)), float(jnp.mean(t_rgb ** 2)) + 1e-8
    from PIL import Image
    Image.fromarray(upscale(target_image(target_np, color), 4)).save(out / "target.png")
    print(f"target {tc['name']!r}: {int((target_np[..., 3] > 0).sum())} cells, total mass {mass:.1f}, "
          f"model {rc['model']['name']} C={cfg.C} k={cfg.k}" + (f", fitting color of first 3 {hidden}" if color else ""))

    spec = jax.tree.map(eqx.is_inexact_array, model)
    if not rc["init"]["learn"]:  # freeze the seed
        spec = eqx.tree_at(lambda m: m.A0, spec, False)
        if model.H0 is not None:
            spec = eqx.tree_at(lambda m: m.H0, spec, False)
    diff, static = eqx.partition(model, spec)
    clamp = make_clamp(rc["clamp"])
    n_seed = model.A0.size + (0 if model.H0 is None else model.H0.size)
    n_params = sum(x.size for x in jax.tree.leaves(diff))
    print(f"trainable parameters: {n_params} ({n_seed} seed values {'learned' if rc['init']['learn'] else 'frozen'})")

    T_lo, T_hi = tr["steps"]
    W = tr["loss_window"]

    def step_loss(state):
        A = state.A.sum(-1)
        l_mass = jnp.mean((A - t_mass) ** 2) / norm_mass
        l_col = 0.
        if color:
            occ = jnp.clip(A / tc["density"], 0., 1.)[..., None]
            l_col = jnp.mean((occ * getattr(state, hidden)[..., :3] - t_rgb) ** 2) / norm_rgb
        return l_mass, l_col

    def sample_loss(diff, key, t_end):
        mdl = eqx.combine(diff, static)
        k_init, k_run = jr.split(key)
        s = mdl.initialize(k_init, rc["init"]["noise"])

        def body(s, xs):
            t, k = xs
            s = mdl(s, k)
            l_mass, l_col = step_loss(s)
            w = ((t + 1 > t_end - W) & (t + 1 <= t_end)).astype(jnp.float32) / W
            return s, (w * l_mass, w * l_col)

        if tr["remat"]:
            body = jax.checkpoint(body)
        _, (lm, lc) = jax.lax.scan(body, s, (jnp.arange(T_hi), jr.split(k_run, T_hi)))
        return lm.sum(), lc.sum()

    def loss_fn(diff, keys, t_ends):
        lm, lc = jax.vmap(sample_loss, in_axes=(None, 0, 0))(diff, keys, t_ends)
        lm, lc = lm.mean(), lc.mean()
        return lm + tr["color_weight"] * lc, (lm, lc)

    sched = optax.cosine_decay_schedule(tr["lr"], tr["iterations"], alpha=tr["lr_end"] / tr["lr"])
    opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(tr["grad_clip"]), optax.adam(sched)), 20)
    opt_state = opt.init(diff)

    @eqx.filter_jit
    def train_step(diff, opt_state, key):
        kk, kt = jr.split(key)
        keys = jr.split(kk, tr["batch_size"])
        t_ends = jr.randint(kt, (tr["batch_size"],), T_lo, T_hi + 1)
        (loss, (lm, lc)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(diff, keys, t_ends)
        updates, opt_state = opt.update(grads, opt_state, diff)
        diff = clamp(eqx.apply_updates(diff, updates))
        return diff, opt_state, loss, lm, lc, optax.global_norm(grads)

    def save(diff, history):
        eqx.tree_serialise_leaves(out / "model.eqx", eqx.combine(diff, static))
        (out / "losses.json").write_text(json.dumps(history))

    # --- loop
    key = jr.key(rc["seed"])
    history = {"iter": [], "loss": [], "mass": [], "color": []}
    print(f"compiling (rollouts of {T_hi} steps, batch {tr['batch_size']})…")
    t0, best = time.time(), float("inf")
    for it in range(1, tr["iterations"] + 1):
        key, k = jr.split(key)
        diff, opt_state, loss, lm, lc, gn = train_step(diff, opt_state, k)
        loss, lm, lc, gn = float(loss), float(lm), float(lc), float(gn)
        history["iter"].append(it); history["loss"].append(loss); history["mass"].append(lm); history["color"].append(lc)
        if it == 1:
            print(f"compiled + first step in {time.time() - t0:.1f}s"); t0 = time.time()
        if not np.isfinite(loss):
            print(f"[{it}] non-finite loss, update skipped")
        if it % tr["log_every"] == 0 or it == tr["iterations"]:
            dt = (time.time() - t0) / max(it - 1, 1)
            print(f"[{it:5d}/{tr['iterations']}] loss {loss:.4f} (mass {lm:.4f}" + (f", color {lc:.4f}" if color else "")
                  + f") |grad| {gn:.3g}  {dt:.2f}s/it")
        if it % tr["save_every"] == 0 or it == tr["iterations"]:
            save(diff, history)
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    # --- outputs: loss curve, final rollout vs target, video
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(history["iter"], history["loss"], label="total", color="k", lw=1)
    ax.semilogy(history["iter"], history["mass"], label="mass", lw=1)
    if color:
        ax.semilogy(history["iter"], history["color"], label="color", lw=1)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=120); plt.close(fig)

    mdl = eqx.combine(diff, static)
    s = mdl.initialize(jr.key(rc["eval"]["seed"]))

    @eqx.filter_jit
    def rollout(mdl, s, key):
        def body(s, k):
            s = mdl(s, k)
            return s, (s.A, getattr(s, hidden) if hidden else s.A)
        return jax.lax.scan(body, s, jr.split(key, rc["eval"]["steps"]))[1]

    As, Hs = map(np.asarray, rollout(mdl, s, jr.key(rc["eval"]["seed"] + 1)))
    Hs = Hs if hidden else None
    frames = np.stack([render(As[t], None if Hs is None else Hs[t], tc["density"], color) for t in range(len(As))])
    tgt = target_image(target_np, color)
    sep = np.full((cfg.X, 2, 3), 80, np.uint8)
    t_mid = min((T_lo + T_hi) // 2, len(frames)) - 1
    Image.fromarray(upscale(np.concatenate([tgt, sep, frames[t_mid], sep, frames[-1]], 1), 4)).save(out / "final.png")
    # learned seed: matter (left) and matter x first 3 hidden channels (right), at its place in the grid
    A0, H0 = map(lambda x: None if x is None else np.asarray(x), mdl.seed_patch())
    seed_imgs = [render(A0 / A0.max(), None, 1., False)] + ([render(A0 / A0.max(), H0, 1., True)] if color else [])
    sep_s = np.full((A0.shape[0], 2, 3), 80, np.uint8)
    Image.fromarray(upscale(np.concatenate(sum([[im, sep_s] for im in seed_imgs], [])[:-1], 1), 8)).save(out / "seed.png")
    video = save_video(np.stack([upscale(np.concatenate([tgt, sep, f], 1), 4) for f in frames]),
                       out / "rollout.mp4", rc["eval"]["fps"])
    print(f"saved to {out}/: model.eqx, config.yml, losses.json, loss.png, target.png, seed.png, "
          f"final.png (target | step {t_mid + 1} | step {len(frames)}), {video.name}")


if __name__ == "__main__":
    main()
