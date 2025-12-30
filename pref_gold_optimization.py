"""Single-objective preference-based BO for gold-looking alloys.

Simulates a jeweler comparing alloys: a PairwiseGP surrogate learns a latent utility
surface from gold-likeness comparisons, searches the discrete dataset, and produces
publication-grade plots summarizing which alloys were sampled and how preferences evolved.
"""
import argparse
import colorsys
from itertools import combinations
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.settings import cholesky_jitter
from linear_operator.utils.errors import NotPSDError
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


TARGET_RGB = np.array([249 / 255, 224 / 255, 175 / 255])
TARGET_HSV = colorsys.rgb_to_hsv(*TARGET_RGB)


def gold_likeness(color_int, metric="hsv", w1=0.5, w2=0.5, sigma=20, tau=0.25):
    """Heuristic score for how visually gold-like / yellow-orange the alloy color appears."""
    red = (color_int >> 16) & 0xFF
    green = (color_int >> 8) & 0xFF
    blue = color_int & 0xFF
    r, g, b = red / 255, green / 255, blue / 255
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = hue * 360
    warm_score = np.exp(-((hue_deg - 45.0) ** 2) / (2 * sigma**2))

    # Yellow / orange distance computed in HSV space to emphasize hue & saturation alignment
    if metric == "hsv":
        color_feat = np.array([hue, saturation, value])
        target_feat = np.array(TARGET_HSV)
        weights = np.array([1.0, 1.0, 1.0])
        weighted_diff = (color_feat - target_feat) * weights
        dist = np.linalg.norm(weighted_diff)
    else:
        color_feat = np.array([r, g, b])
        dist = np.linalg.norm(color_feat - TARGET_RGB)
    harmony_score = np.exp(-(dist**2) / (2 * tau**2))
    return float(w1 * warm_score + w2 * harmony_score)


def build_preference_pairs(values):
    """Return tensor of (preferred, other) index pairs from scalar utilities."""
    pref_rows = []
    for i, j in combinations(range(len(values)), 2):
        if values[i] > values[j]:
            pref_rows.append([i, j])
        elif values[j] > values[i]:
            pref_rows.append([j, i])
        # ties add no data
    if not pref_rows:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(pref_rows, dtype=torch.long)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    compositions = torch.tensor(
        df[["E1", "E2", "E3", "E4", "E5"]].values, dtype=torch.double
    )
    keys = [
        "|".join(f"{val:.4f}" for val in row)
        for row in df[["E1", "E2", "E3", "E4", "E5"]].values
    ]
    colors = df["24-bit color"].astype(int).values
    color_lookup = dict(zip(keys, colors))
    return compositions, color_lookup


def color_from_composition(comp_tensor, color_map):
    key = "|".join(f"{val:.4f}" for val in comp_tensor.tolist())
    return color_map[key]


def color_int_to_rgb(color_int):
    """Convert 24-bit integer color to normalized RGB tuple."""
    return (
        ((color_int >> 16) & 0xFF) / 255,
        ((color_int >> 8) & 0xFF) / 255,
        (color_int & 0xFF) / 255,
    )


def plot_color_progress(records, output_path):
    """Publication-style visualization showing sampling order and ranking."""
    if not records:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    rgbs = np.array(
        [[(c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF] for c in [r["color"] for r in records]]
    ) / 255.0
    utilities = np.array([r["utility"] for r in records])
    iterations = np.array([r["iteration"] for r in records])
    order = np.argsort(-utilities)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(records) + 1)
    target_rgb = np.array([249 / 255, 224 / 255, 175 / 255])

    width = max(4, len(records) * 0.4)
    fig, (ax_iter, ax_rank) = plt.subplots(
        nrows=2, figsize=(width, 3.0), gridspec_kw={"height_ratios": [1, 1]}
    )

    for i, rgb in enumerate(rgbs):
        rect = Rectangle((i, 0), 1, 1, facecolor=rgb, edgecolor="black", linewidth=0.2)
        ax_iter.add_patch(rect)
        ax_iter.text(
            i + 0.5,
            0.5,
            f"RNK {int(ranks[i])}\nITR {iterations[i]}",
            ha="center",
            va="center",
            fontsize=5,
            color="black",
            weight="bold",
        )
    ax_iter.set_xlim(0, len(records))
    ax_iter.set_ylim(0, 1)
    ax_iter.axis("off")
    ax_iter.set_title("Color progression of evaluated alloys", pad=20)

    sorted_rgbs = rgbs[order]
    sorted_iters = iterations[order]
    ax_rank.set_xlim(0, len(records))
    ax_rank.set_ylim(0, 1)
    ax_rank.axis("off")
    ax_rank.set_title("Alloys ordered by preference rank", pad=10)
    for idx, rgb in enumerate(sorted_rgbs):
        rect = Rectangle((idx, 0), 1, 1, facecolor=rgb, edgecolor="black", linewidth=0.2)
        ax_rank.add_patch(rect)
        ax_rank.text(
            idx + 0.5,
            0.5,
            f"RNK {idx + 1}\nITR {sorted_iters[idx]}",
            ha="center",
            va="center",
            fontsize=5,
            color="black",
            weight="bold",
        )

    if len(sorted_rgbs) > 0:
        target_rect = Rectangle(
            (0.05, 0.65),
            0.25,
            0.3,
            facecolor=target_rgb,
            edgecolor="black",
            linewidth=0.6,
        )
        ax_rank.add_patch(target_rect)
        ax_rank.annotate(
            "",
            xy=(0.05, 0.95),
            xytext=(0.05, 1.18),
            arrowprops=dict(
                arrowstyle="-|>", color="black", linewidth=0.8, shrinkA=0, shrinkB=0
            ),
        )
        ax_rank.text(
            0.12,
            1.18,
            "Target color",
            ha="left",
            va="center",
            fontsize=7,
        )

    ax_iter.set_title("Color progression of evaluated alloys", pad=12)
    ax_rank.set_title("Alloys ordered by preference rank", pad=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=900)
    plt.close(fig)


def plot_tournament(matches, output_path):
    if not matches or not output_path:
        return
    width = max(6, len(matches) * 1.5)
    fig, ax = plt.subplots(figsize=(width, 3.2))
    prev_winner_pos = None
    for idx, match in enumerate(matches):
        x = idx * 1.5
        champ_rgb = color_int_to_rgb(match["champion_color"])
        chal_rgb = color_int_to_rgb(match["challenger_color"])
        champ_rect = Rectangle((x, 1.1), 1.2, 0.35, facecolor=champ_rgb, edgecolor="black")
        chal_rect = Rectangle((x, 0.3), 1.2, 0.35, facecolor=chal_rgb, edgecolor="black")
        ax.add_patch(champ_rect)
        ax.add_patch(chal_rect)
        ax.text(x + 0.6, 1.5, f"Iter {match['iteration']}", ha="center", va="bottom", fontsize=8)
        ax.text(x + 0.6, 1.25, "Champion", ha="center", va="bottom", fontsize=7)
        ax.text(x + 0.6, 0.48, "Challenger", ha="center", va="bottom", fontsize=7)
        # connect champion lineage
        champ_center = (x + 0.6, 1.275)
        if prev_winner_pos is not None:
            ax.plot(
                [prev_winner_pos[0], champ_center[0]],
                [prev_winner_pos[1], champ_center[1]],
                color="black",
                linewidth=1.2,
            )
        # link champion vs challenger
        ax.plot(
            [x + 0.6, x + 0.6],
            [0.65, 1.1],
            color="lightgray",
            linewidth=1.0,
            linestyle="--",
        )
        winner_y = 1.275 if match["winner"] == "champion" else 0.475
        ax.scatter(
            x + 0.6,
            winner_y,
            marker="*",
            color="gold",
            edgecolor="black",
            s=90,
            zorder=3,
        )
        prev_winner_pos = (x + 0.6, winner_y)
    ax.set_xlim(-0.2, max(6, len(matches) * 1.5))
    ax.set_ylim(0.2, 1.7)
    ax.axis("off")
    ax.set_title("Champion–Challenger Bracket")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_pairwise_model(train_x, pref_pairs, kernel_choice):
    ard_dims = train_x.shape[-1]
    if kernel_choice == "rbf":
        base_kernel = RBFKernel(ard_num_dims=ard_dims)
    elif kernel_choice == "matern32":
        base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_dims)
    elif kernel_choice == "matern52":
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_dims)
    else:
        raise ValueError(f"Unknown kernel choice '{kernel_choice}'")
    covar_module = ScaleKernel(base_kernel)
    return PairwiseGP(train_x, pref_pairs, covar_module=covar_module, jitter=1e-2)


def batched_posterior_mean(model, X, chunk_size=256):
    means = []
    X = X.to(model.train_inputs[0].dtype)
    with torch.no_grad():
        for chunk in X.split(chunk_size):
            if chunk.numel():
                offsets = (
                    torch.arange(chunk.size(0), dtype=chunk.dtype, device=chunk.device)
                    .unsqueeze(1)
                    .expand(-1, chunk.size(1))
                )
                chunk = chunk + 1e-6 * offsets
            post = None
            for extra_jitter in (None, 1e-2, 1e-1, 1.0):
                ctx = (
                    cholesky_jitter(extra_jitter)
                    if extra_jitter is not None
                    else nullcontext()
                )
                try:
                    with ctx:
                        post = model.posterior(chunk)
                    break
                except NotPSDError:
                    post = None
                    continue
            if post is None:
                raise NotPSDError(
                    "Unable to compute posterior; matrix not PSD even after jitter."
                )
            means.append(post.mean.squeeze(-1))
    return torch.cat(means, dim=0)


def preferential_bo(
    csv_path,
    iterations,
    init_points,
    seed,
    plot_path,
    kernel_choice,
    metric,
    tournament_plot,
):
    torch.manual_seed(seed)
    compositions, color_map = load_dataset(csv_path)

    # bootstrap training data with distinct random alloys
    init_idx = torch.randperm(len(compositions))[:init_points]
    train_x = compositions[init_idx].clone()
    selected = set(init_idx.tolist())
    evaluations = []
    matches = []
    util_list = []
    for comp in train_x:
        color_int = color_from_composition(comp, color_map)
        util = gold_likeness(color_int, metric=metric)
        util_list.append(util)
        evaluations.append(
            {
                "color": color_int,
                "composition": comp.tolist(),
                "utility": util,
                "iteration": len(evaluations) + 1,
            }
        )
    utilities = torch.tensor(util_list, dtype=torch.float32)
    pref_pairs = build_preference_pairs(utilities)

    history = []
    for itr in range(iterations):
        best_before_idx = torch.argmax(utilities).item()
        champion_comp = train_x[best_before_idx]
        champion_color = color_from_composition(champion_comp, color_map)
        champion_util = float(utilities[best_before_idx].item())
        if len(pref_pairs) == 0:
            raise RuntimeError("Need at least one strict preference to fit PairwiseGP.")
        model = build_pairwise_model(train_x, pref_pairs, kernel_choice)
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        sys.stdout.write(
            f"[Iter {itr + 1}] Fitting PairwiseGP (train size={len(train_x)})... "
        )
        sys.stdout.flush()
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            print("failed.")
            raise
        else:
            print("done.")

        posterior = batched_posterior_mean(model, compositions)
        mask = torch.zeros(len(compositions), dtype=torch.bool)
        mask[list(selected)] = True
        posterior = posterior.masked_fill(mask, float("-inf"))
        next_idx = torch.argmax(posterior).item()
        selected.add(next_idx)
        candidate = compositions[next_idx]
        candidate_color = color_from_composition(candidate, color_map)
        true_util = gold_likeness(candidate_color, metric=metric)

        train_x = torch.cat([train_x, candidate.unsqueeze(0)], dim=0)
        utilities = torch.cat(
            [utilities, torch.tensor([true_util], dtype=torch.float32)], dim=0
        )
        pref_pairs = build_preference_pairs(utilities)

        best_idx = torch.argmax(utilities).item()
        best_comp = train_x[best_idx]
        eval_record = {
            "color": candidate_color,
            "composition": candidate.tolist(),
            "utility": true_util,
            "iteration": len(evaluations) + 1,
        }
        evaluations.append(eval_record)
        matches.append(
            {
                "iteration": itr + 1,
                "champion_color": champion_color,
                "challenger_color": candidate_color,
                "winner": "challenger"
                if true_util > champion_util
                else "champion",
            }
        )
        history.append(
            {
                "iteration": itr + 1,
                "candidate_index": next_idx,
                "candidate_score": true_util,
                "best_score": utilities[best_idx].item(),
                "best_composition": best_comp.tolist(),
            }
        )
        print(
            f"[Iter {itr + 1}] evaluated idx {next_idx} | utility={true_util:.4f} | "
            f"best utility={utilities[best_idx].item():.4f}"
        )

    best_idx = torch.argmax(utilities).item()
    best_comp = train_x[best_idx]
    print("\nBest alloy found:")
    print(f"  Composition (Au, Ag, Cu, Pt, Al): {best_comp.tolist()}")
    print(f"  Gold-likeness score: {utilities[best_idx].item():.4f}")
    if plot_path:
        plot_color_progress(evaluations, plot_path)
    if tournament_plot:
        plot_tournament(matches, tournament_plot)
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Single-objective preferential BO focusing on gold-likeness."
    )
    parser.add_argument(
        "--csv",
        default="color_quinary_5%.csv",
        help="Dataset with alloy compositions and 24-bit colors.",
    )
    parser.add_argument("--iterations", type=int, default=10, help="BO iterations.")
    parser.add_argument(
        "--init-points", type=int, default=3, help="Random alloys to warm start with."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--color-plot",
        default="color_progress.png",
        help="Path to save the color progression plot (set empty to skip).",
    )
    parser.add_argument(
        "--kernel",
        choices=["rbf", "matern32", "matern52"],
        default="rbf",
        help="Kernel used inside the PairwiseGP surrogate.",
    )
    parser.add_argument(
        "--metric",
        choices=["hsv", "rgb"],
        default="hsv",
        help="Distance metric for the harmony score inside gold_likeness.",
    )
    parser.add_argument(
        "--tournament-plot",
        default="tournament_progress.png",
        help="Path to save a bracket-style visualization (set empty to skip).",
    )
    args = parser.parse_args()
    plot_path = args.color_plot if args.color_plot else None
    tournament_plot = args.tournament_plot if args.tournament_plot else None
    preferential_bo(
        args.csv,
        args.iterations,
        args.init_points,
        args.seed,
        plot_path,
        args.kernel,
        args.metric,
        tournament_plot,
    )


if __name__ == "__main__":
    main()
