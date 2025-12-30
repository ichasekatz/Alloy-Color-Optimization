import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "color_quinary_5%.csv"
OUTPUT_PATH = "quinary_affine_projection.png"


def load_data(path):
    df = pd.read_csv(path)
    comps = df[["E1", "E2", "E3", "E4", "E5"]].values
    colors = df["24-bit color"].astype(int).values
    return comps, colors


def color_int_to_rgb(color_int):
    return (
        ((color_int >> 16) & 0xFF) / 255,
        ((color_int >> 8) & 0xFF) / 255,
        (color_int & 0xFF) / 255,
    )


def affine_projection(compositions):
    angles = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2
    vertex = np.column_stack((np.cos(angles), np.sin(angles)))
    return compositions @ vertex, vertex


def main():
    comps, color_ints = load_data(DATA_PATH)
    coords, vertex = affine_projection(comps)
    rgbs = np.array([color_int_to_rgb(c) for c in color_ints])

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

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=rgbs,
        edgecolor="black",
        linewidth=0.2,
        s=18,
    )
    polygon = np.vstack([vertex, vertex[0]])
    ax.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=1.3)
    labels = ["E1", "E2", "E3", "E4", "E5"]
    for idx, (coord, label) in enumerate(zip(vertex, labels)):
        scale = 1.08 if label == "Au" else 1.15
        ax.text(
            coord[0] * scale,
            coord[1] * scale,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Optical Model Across Alloy Space", pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=900)
    plt.close(fig)


if __name__ == "__main__":
    main()
