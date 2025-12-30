# Jeweler-in-the-Loop Paper Figures

This repo reproduces three figures for the paper
"Jeweler-in-the-Loop: Personalized Alloy Color Optimization via Preference-Based BO"
(`ring_paper.tex`).

- `color_progress.png` (`pref_gold_optimization.py`): sequence of queried alloys,
  shown as color chips with iteration and final rank annotations. Used in
  `ring_paper.tex` as Figure `\\ref{fig:color_progress}`.
- `quinary_affine_projection.png` (`plot_quinary_projection.py`): 2D affine
  projection of the quinary composition simplex, colored by predicted appearance.
  Used in `ring_paper.tex` as Figure `\\ref{fig:quinary}`.
- `multiobj_objectives.png` (`multiobjective_ehvi.py`): color-preference rank vs.
  price for queried alloys, with the non-dominated (Pareto) front highlighted.
  Used in `ring_paper.tex` as Figure `\\ref{fig:cost_color_pareto}`.

## Paper context

- The manuscript uses `\\graphicspath{{./images/}}`, so place the generated PNGs in
  `images/` if you want LaTeX to pick them up without editing paths.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch botorch gpytorch linear_operator
```

## Data

The scripts expect `color_quinary_5%.csv` in the repo root (already included).
Element columns are obfuscated as `E1`-`E5`, and compositions include small
jitter to prevent direct mapping between compositions and colors.

## Scripts and outputs

- `pref_gold_optimization.py` runs single-objective preference-based BO over the
  obfuscated dataset and writes `color_progress.png` to visualize the sampling
  order and resulting preference ranking.
- `plot_quinary_projection.py` renders the quinary simplex as a pentagon and
  writes `quinary_affine_projection.png` with each point colored by its RGB
  appearance.
- `multiobjective_ehvi.py` runs multi-objective BO balancing color preference and
  price, then writes `multiobj_objectives.png` showing the trade-off and Pareto
  front.

## Generate figures

Create an `images/` directory if you want LaTeX to find the figures directly:

```bash
mkdir -p images
```

```bash
python pref_gold_optimization.py --csv color_quinary_5%.csv --color-plot images/color_progress.png
python plot_quinary_projection.py
mv quinary_affine_projection.png images/quinary_affine_projection.png
python multiobjective_ehvi.py --csv color_quinary_5%.csv --plot images/multiobj_objectives.png
```

Outputs are written with the filenames shown above.
