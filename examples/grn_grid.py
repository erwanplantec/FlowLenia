"""Render a grid of randomly sampled GRN Flow Lenia systems to a video.

Each tile is a different random model (kernels, growth and GRN weights) started
from a random patch of matter and genes. Matter is colored by the first 3 genes.

    PYTHONPATH=. uv run python examples/grn_grid.py --rows 4 --cols 4 --steps 500 --out grn_grid.mp4
"""
import argparse
import shutil
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from flowlenia.grn_flowlenia import GRNFlowLenia, Config, State
from flowlenia.utils import conn_from_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=4)
parser.add_argument("--cols", type=int, default=4)
parser.add_argument("--size", type=int, default=128, help="grid size of each system")
parser.add_argument("--patch", type=int, default=40, help="size of the initial random patch")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--every", type=int, default=2, help="record one frame every n steps")
parser.add_argument("--mode", type=str, default="mace", choices=["mace", "rt"])
parser.add_argument("--nb_genes", type=int, default=16)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", type=str, default="grn_grid.mp4", help=".mp4 (needs ffmpeg) or .gif")
args = parser.parse_args()

seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
print(f"seed: {seed}")
key = jr.key(seed)
N = args.rows * args.cols
# --- 1. Set system config
M = np.array([[3, 1, 0],
              [0, 3, 1],
              [1, 0, 3]], dtype=int)
k = int(M.sum())
assert args.nb_genes >= k, "first k genes are used as growth weights"
c0, c1 = conn_from_matrix(M)
cfg = Config(X=args.size, Y=args.size, C=3, k=k, c0=c0, c1=c1, mace_temp=0.3,
             update_mode=args.mode, nb_genes=args.nb_genes)
# --- 2. Sample N random systems (batched model)
key, key_fl = jr.split(key)
fls = eqx.filter_vmap(lambda k: GRNFlowLenia(cfg, key=k))(jr.split(key_fl, N))

# --- 3. Initialize states with a random centered patch
def init(fl: GRNFlowLenia, key: jax.Array) -> State:
    kA, ks = jr.split(key)
    s = fl.initialize(key)
    p, lo = args.patch, (args.size - args.patch) // 2
    A = s.A.at[lo:lo+p, lo:lo+p].set(jr.uniform(kA, (p, p, cfg.C)))
    g = s.s.at[lo:lo+p, lo:lo+p].set(jr.uniform(ks, (p, p, cfg.nb_genes)))
    return s.replace(A=A, s=g)

key, key_init = jr.split(key)
states = eqx.filter_vmap(init)(fls, jr.split(key_init, N))

# --- 4. Simulate, only keeping rendered uint8 frames to save memory
def render(s: State) -> jax.Array:
    img = jnp.clip(s.A * s.s[..., :3], 0., 1.)
    return (img * 255).astype(jnp.uint8)

@eqx.filter_jit
def simulate(fls, states, key):
    step = eqx.filter_vmap(lambda fl, s, k: fl(s, k))
    def _record(c, _):
        s, k = c
        def _step(c, _):
            s, k = c
            k, k_ = jr.split(k)
            return (step(fls, s, jr.split(k_, N)), k), None
        (s, k), _ = jax.lax.scan(_step, (s, k), None, args.every)
        return (s, k), jax.vmap(render)(s)
    _, frames = jax.lax.scan(_record, (states, key), None, args.steps // args.every)
    return frames  # (T, N, X, Y, 3)

frames = np.asarray(simulate(fls, states, key))
print(f"simulated {args.steps} steps of {N} systems -> {frames.shape[0]} frames")

# --- 5. Tile into a grid (1px separators)
T, _, H, W, _ = frames.shape
pad = 1
frames = np.pad(frames, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), constant_values=40)
frames = frames.reshape(T, args.rows, args.cols, H + 2*pad, W + 2*pad, 3)
frames = frames.transpose(0, 1, 3, 2, 4, 5).reshape(T, args.rows * (H + 2*pad), args.cols * (W + 2*pad), 3)
frames = frames[:, :frames.shape[1] // 2 * 2, :frames.shape[2] // 2 * 2]  # h264 needs even dims

# --- 6. Write video
if args.out.endswith(".gif") or shutil.which("ffmpeg") is None:
    from PIL import Image
    out = args.out if args.out.endswith(".gif") else args.out.rsplit(".", 1)[0] + ".gif"
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=1000 // args.fps, loop=0)
else:
    out = args.out
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
        stdin=subprocess.PIPE)
    proc.communicate(frames.tobytes())
print(f"saved to {out}")
