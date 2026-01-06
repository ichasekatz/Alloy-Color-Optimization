"""
Multi-objective Bayesian optimization for alloy design.

This script mimics a preference-learning loop where a dummy user is repeatedly shown
two candidate alloys. The model assumes the user always prefers the option whose
color is closer to a target gold hue. Those pairwise comparisons fit a `PairwiseGP`,
which models an underlying utility surface from the observed win/loss outcomes.
In parallel, a `SingleTaskGP` captures the USD/g price surface: because we know the
price of every candidate alloy up front, the price GP is trained once on the full
dataset and never updated.

The two objectives optimized with analytic EHVI are:
1. Color preference (via pairwise GP, minimizing distance to the target hue).
2. Estimated material price per gram (trained once over all alloys).

At each iteration the script proposes new alloys believed to strike a balance between
preferred color and low price, effectively emulating what the system would recommend
to a user looking for both gold-like color and affordability.

Usage example::

    python multiobjective_ehvi.py --iterations 15 --init-points 5 --seed 0 --plot multiobj.png
"""

import argparse
import colorsys

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf_discrete
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt


TARGET_RGB = np.array([249 / 255, 224 / 255, 175 / 255])
TARGET_HSV = colorsys.rgb_to_hsv(*TARGET_RGB)
METAL_MOLAR_MASS = np.array(
    [196.96657, 107.8682, 63.546, 195.084, 26.9815385], dtype=np.float64
)
METAL_PRICE_PER_G = np.array([60.0, 0.8, 0.009, 35.0, 0.002], dtype=np.float64)


def color_distance(color_int, metric="hsv"):
    """Return distance from the target gold color in RGB or HSV space."""
    red = (color_int >> 16) & 0xFF
    green = (color_int >> 8) & 0xFF
    blue = color_int & 0xFF
    r, g, b = red / 255, green / 255, blue / 255
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
    if metric == "hsv":
        diff = np.array([hue, saturation, value]) - np.array(TARGET_HSV)
        dist = np.linalg.norm(diff)
    else:
        diff = np.array([r, g, b]) - TARGET_RGB
        dist = np.linalg.norm(diff)
    return float(dist)


def alloy_price(composition):
    """Estimate the price (USD/g) of an alloy by converting to mass fractions."""
    mol_frac = composition / METAL_MOLAR_MASS
    mol_frac /= mol_frac.sum()
    mass_frac = mol_frac * METAL_MOLAR_MASS / np.sum(mol_frac * METAL_MOLAR_MASS)
    return float(np.dot(mass_frac, METAL_PRICE_PER_G))


def load_dataset(csv_path):
    """Load obfuscated alloy compositions (E1-E5) and their colors."""
    df = pd.read_csv(csv_path)
    comps = torch.tensor(df[["E1", "E2", "E3", "E4", "E5"]].values, dtype=torch.double)
    colors = df["24-bit color"].astype(int).values
    return comps, colors


def evaluate_objectives(comp_tensor, color_int, metric):
    """Return [color distance, price] for a single alloy tensor."""
    comp_np = comp_tensor.numpy()
    return torch.tensor(
        [color_distance(color_int, metric), alloy_price(comp_np)], dtype=torch.double
    )


def init_data(compositions, colors, metric, n_init, seed):
    """Sample a random subset of alloys to bootstrap BO."""
    torch.manual_seed(seed)
    idx = torch.randperm(len(compositions))[:n_init]
    mask = torch.zeros(len(compositions), dtype=torch.bool)
    mask[idx] = True
    train_x = compositions[idx]
    raw = torch.stack(
        [evaluate_objectives(train_x[i], colors[idx[i]], metric) for i in range(n_init)],
        dim=0,
    )
    return mask, train_x, raw, idx.tolist()


def scale_color(color_vals, stats):
    """Normalize color distances to [0,1] using running min/max."""
    denom = max(stats["max"] - stats["min"], 1e-8)
    return (color_vals - stats["min"]) / denom


def build_preference_pairs(values):
    """Create pairwise preferences assuming smaller color distance is better."""
    prefs = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] == values[j]:
                continue
            if values[i] < values[j]:
                prefs.append([i, j])
            else:
                prefs.append([j, i])
    if not prefs:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(prefs, dtype=torch.long)


def plot_objectives(history, filename):
    """Plot color preference rank vs price for every queried alloy."""
    if not history:
        return
    color_dists = np.array([h["color_dist"] for h in history])
    prices = np.array([h["price"] for h in history])
    rgbs = np.array([h["color_rgb"] for h in history])
    order = np.argsort(color_dists)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(color_dists) + 1)
    pareto_tensor = torch.tensor(list(zip(color_dists, prices)), dtype=torch.double)
    pareto_mask = is_non_dominated(-pareto_tensor)
    pareto_prices = prices[pareto_mask]
    pareto_ranks = ranks[pareto_mask]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    plt.figure(figsize=(4, 3))
    plt.scatter(
        prices,
        ranks,
        c=rgbs,
        edgecolor="black",
        s=60,
        zorder=1.0,
        label="Queried alloys",
    )
    if pareto_prices.size > 0:
        unique_pairs = np.unique(np.column_stack([pareto_prices, pareto_ranks]), axis=0)
        order = np.argsort(unique_pairs[:, 0])
        ordered = unique_pairs[order]
        plt.plot(
            ordered[:, 0],
            ordered[:, 1],
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="Pareto Front",
            zorder=0.8,
        )

    ax = plt.gca()
    ax.set_ylim(bottom=0)
    yticks = [tick for tick in ax.get_yticks() if tick != 0]
    if 1 not in yticks:
        yticks.append(1)
    ax.set_yticks(sorted(set(yticks)))

    plt.xlabel("Price ($/g)", labelpad=6)
    plt.ylabel("Color Preference Ranking", labelpad=6)
    plt.title("Color vs. Price for Queried Alloys", pad=10)
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(filename, dpi=900)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective BO minimizing color distance and alloy price."
    )
    parser.add_argument("--csv", default="color_quinary_5%.csv")
    parser.add_argument("--metric", choices=["hsv", "rgb"], default="hsv")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--init-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="multiobj_objectives.png")
    args = parser.parse_args()

    compositions, colors = load_dataset(args.csv)
    selected_mask, train_x, train_raw, init_indices = init_data(
        compositions, colors, args.metric, args.init_points, args.seed
    )

    price_all = torch.tensor(
        [alloy_price(compositions[i].numpy()) for i in range(len(compositions))],
        dtype=torch.double,
    )
    price_min, price_max = price_all.min().item(), price_all.max().item()
    price_norm_all = (price_all - price_min) / max(price_max - price_min, 1e-8)
    price_gp = SingleTaskGP(compositions, price_norm_all.unsqueeze(-1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(price_gp.likelihood, price_gp))
    except ModelFittingError:
        print("Price GP fit failed; using unoptimized model.")

    color_stats = {
        "min": float(train_raw[:, 0].min().item()),
        "max": float(train_raw[:, 0].max().item()),
    }
    train_indices = init_indices[:]
    color_values = [float(val) for val in train_raw[:, 0]]
    pref_pairs = build_preference_pairs(color_values)
    scaled_color = scale_color(torch.tensor(color_values, dtype=torch.double), color_stats)
    train_obj = torch.stack([scaled_color, price_norm_all[train_indices]], dim=-1)
    train_y = -train_obj

    ref_point = torch.tensor([-1.1, -1.1], dtype=torch.double)

    history = []
    for i, idx in enumerate(init_indices):
        history.append(
            {
                "iteration": 0,
                "index": idx,
                "color_dist": float(train_raw[i, 0]),
                "price": float(train_raw[i, 1]),
                "hypervolume": None,
                "color_rgb": [
                    ((colors[idx] >> 16) & 0xFF) / 255,
                    ((colors[idx] >> 8) & 0xFF) / 255,
                    (colors[idx] & 0xFF) / 255,
                ],
            }
        )

    for itr in range(args.iterations):
        if pref_pairs.numel() == 0:
            raise RuntimeError("Pairwise GP requires at least one strict preference.")
        color_gp = PairwiseGP(train_x, pref_pairs, jitter=1e-2)
        color_mll = PairwiseLaplaceMarginalLogLikelihood(color_gp.likelihood, color_gp)
        try:
            fit_gpytorch_mll(color_mll)
        except ModelFittingError:
            print(f"[Iter {itr + 1}] Color GP fit failed; using unoptimized model.")
        model = ModelListGP(color_gp, price_gp)

        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
        acq = ExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, partitioning=partitioning
        )

        avail_idx = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
        choices = compositions[avail_idx]
        candidate, _ = optimize_acqf_discrete(acq, q=1, choices=choices)
        candidate = candidate.squeeze(0)
        match = torch.isclose(choices, candidate).all(dim=1)
        rel_idx = torch.nonzero(match, as_tuple=False).squeeze().item()
        global_idx = avail_idx[rel_idx].item()

        selected_mask[global_idx] = True
        new_raw = evaluate_objectives(
            compositions[global_idx], colors[global_idx], args.metric
        )
        train_x = torch.cat([train_x, compositions[global_idx].unsqueeze(0)], dim=0)
        train_raw = torch.cat([train_raw, new_raw.unsqueeze(0)], dim=0)
        color_stats["min"] = min(color_stats["min"], float(new_raw[0]))
        color_stats["max"] = max(color_stats["max"], float(new_raw[0]))
        color_values.append(float(new_raw[0]))
        pref_pairs = build_preference_pairs(color_values)
        train_indices.append(global_idx)
        scaled_color = scale_color(
            torch.tensor(color_values, dtype=torch.double), color_stats
        )
        train_obj = torch.stack(
            [scaled_color, price_norm_all[train_indices]], dim=-1
        )
        train_y = -train_obj

        hv = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y).compute_hypervolume()
        history.append(
            {
                "iteration": itr + 1,
                "index": global_idx,
                "color_dist": float(new_raw[0]),
                "price": float(new_raw[1]),
                "hypervolume": float(hv),
                "color_rgb": [
                    ((colors[global_idx] >> 16) & 0xFF) / 255,
                    ((colors[global_idx] >> 8) & 0xFF) / 255,
                    (colors[global_idx] & 0xFF) / 255,
                ],
            }
        )
        print(
            f"[Iter {itr + 1}] idx {global_idx} | color_dist={new_raw[0]:.4f} | price=${new_raw[1]:.2f}/g | HV={hv:.4f}"
        )

    pareto_mask = is_non_dominated(train_obj)
    print("\nPareto frontier (color distance, price $/g):")
    for obj in train_raw[pareto_mask]:
        print(f"  ({obj[0]:.4f}, ${obj[1]:.2f}/g)")

    if args.plot:
        plot_objectives(history, args.plot)


if __name__ == "__main__":
    main()
