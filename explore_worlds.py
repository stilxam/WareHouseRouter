"""
Render a grid of warehouse worlds for different seeds so you can pick
sufficiently different environments for your sweep.

Usage examples:
  # Scan seeds 0-35 (default)
  python explore_worlds.py

  # Specific seeds
  python explore_worlds.py --seeds 42 123 777 999 1337

  # Range
  python explore_worlds.py --range 0 50

  # Control grid columns and output file
  python explore_worlds.py --range 0 24 --cols 5 --out worlds.png
"""

import argparse
import math
import sys

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from environment.warehouse import WarehouseRobotEnv


# ── rendering ────────────────────────────────────────────────────────────────

def render_world_image(world, params, px: int = 256) -> np.ndarray:
    """Return an (px, px, 3) uint8 RGB image of the world layout."""
    M       = params.M
    W_cell  = params.W_cell
    scale   = px / (M * W_cell)

    def to_pixel(x, y):
        return int(float(x) * scale), int(px - float(y) * scale)

    img = np.ones((px, px, 3), dtype=np.uint8) * 255

    blocked = np.asarray(world.blocked)
    for r in range(M):
        for c in range(M):
            if blocked[r, c]:
                pt1 = to_pixel(c * W_cell, (r + 1) * W_cell)
                pt2 = to_pixel((c + 1) * W_cell, r * W_cell)
                cv2.rectangle(img, pt1, pt2, (120, 120, 120), -1)

    # Start (green), Goal (red circle)
    cv2.circle(img, to_pixel(float(world.x_start), float(world.y_start)),
               max(3, int(0.15 * scale)), (0, 180, 0), -1)
    goal_pt = to_pixel(float(world.x_goal), float(world.y_goal))
    cv2.circle(img, goal_pt, max(3, int(params.r_goal * scale)), (200, 0, 0), 2)
    cv2.circle(img, goal_pt, 3, (200, 0, 0), -1)

    # Grid lines (faint)
    for i in range(M + 1):
        v = int(i * W_cell * scale)
        cv2.line(img, (v, 0), (v, px), (220, 220, 220), 1)
        cv2.line(img, (0, v), (px, v), (220, 220, 220), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def world_stats(world, params) -> dict:
    blocked    = np.asarray(world.blocked)
    M          = params.M
    n_blocked  = int(blocked.sum())
    n_free     = M * M - n_blocked

    # Euclidean start→goal in world units
    dx = float(world.x_goal) - float(world.x_start)
    dy = float(world.y_goal) - float(world.y_start)
    dist_eu = math.sqrt(dx * dx + dy * dy)

    # Manhattan distance in grid cells
    si, sj = int(world.start_idx[0]), int(world.start_idx[1])
    gi, gj = int(world.goal_idx[0]),  int(world.goal_idx[1])
    dist_mn = abs(gi - si) + abs(gj - sj)

    return {
        "blocked":  n_blocked,
        "free":     n_free,
        "dist_eu":  dist_eu,
        "dist_mn":  dist_mn,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browse warehouse world seeds")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seeds", type=int, nargs="+", metavar="S",
                       help="Explicit list of world seeds to render")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                       help="Render seeds START..END-1")
    parser.add_argument("--M",     type=int, default=8,  help="Warehouse grid size M (must match training; default 8)")
    parser.add_argument("--cols",  type=int, default=6,  help="Grid columns (default 6)")
    parser.add_argument("--px",    type=int, default=200, help="Pixels per world image (default 200)")
    parser.add_argument("--out",   type=str, default="world_grid.png",
                        help="Output PNG path (default world_grid.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive display, just save PNG")
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.range:
        seeds = list(range(args.range[0], args.range[1]))
    else:
        seeds = list(range(36))   # default: 0-35

    env    = WarehouseRobotEnv(M=args.M)
    params = env.default_params()

    print(f"Generating {len(seeds)} worlds…")
    images = []
    stats  = []
    for seed in seeds:
        key   = jax.random.PRNGKey(seed)
        world = env.generate_world(key, params)
        images.append(render_world_image(world, params, px=args.px))
        stats.append(world_stats(world, params))

    # ── print summary table ────────────────────────────────────────────────
    print(f"\n{'Seed':>6}  {'Blocked':>7}  {'Free':>5}  {'Dist(eu)':>8}  {'Dist(mn)':>8}")
    print("─" * 44)
    for seed, s in zip(seeds, stats):
        print(f"{seed:>6}  {s['blocked']:>7}  {s['free']:>5}  "
              f"{s['dist_eu']:>8.2f}  {s['dist_mn']:>8}")

    # ── matplotlib grid ───────────────────────────────────────────────────
    n    = len(seeds)
    cols = min(args.cols, n)
    rows = math.ceil(n / cols)

    fig_w = cols * (args.px / 100) + 0.5
    fig_h = rows * (args.px / 100 + 0.55)   # extra height for labels
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for ax, seed, img, s in zip(axes, seeds, images, stats):
        ax.imshow(img)
        ax.set_title(
            f"seed {seed}\nobs={s['blocked']}  d={s['dist_eu']:.1f}",
            fontsize=7, pad=2
        )
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
