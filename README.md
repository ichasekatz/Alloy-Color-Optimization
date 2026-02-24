# Jeweler-in-the-Loop Paper Figures

This repo reproduces figures for the paper
"Jeweler-in-the-Loop: Personalized Alloy Color Optimization via Preference-Based BO"
(`ring_paper.tex`).

- `color_progress.png` (`pref_gold_optimization.py`): sequence of queried alloys,
  shown as color chips with iteration and final rank annotations. Used in
  `ring_paper.tex` as Figure `\\ref{fig:color_progress}`.
- `multiobj_objectives.png` (`multiobjective_ehvi.py`): color-preference rank vs.
  price for queried alloys, with the non-dominated (Pareto) front highlighted.
  Used in `ring_paper.tex` as Figure `\\ref{fig:cost_color_pareto}`.
- `tournament_progress.png` (`pref_gold_optimization.py`): optional bracket-style
  visualization of the champion-challenger comparisons.
- `rgb_projections.png` (`pref_gold_optimization.py`): projections of alloy gold-likeness
  as a function of the R, G, and B channel values.

## Paper context

- The manuscript uses `\\graphicspath{{./images/}}`, so place the generated PNGs in
  `images/` if you want LaTeX to pick them up without editing paths.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch botorch gpytorch linear_operator pillow
```

If you prefer conda:

```bash
conda create -y -n ringopt python=3.11
conda activate ringopt
python -m pip install numpy pandas matplotlib torch botorch gpytorch linear_operator pillow
```

CPU-only runs (avoid CUDA driver warnings):

```bash
CUDA_VISIBLE_DEVICES= python pref_gold_optimization.py --csv color_quinary_5%.csv --color-plot images/color_progress.png --tournament-plot images/tournament_progress.png --rgb-space-plot images/rgb_projections.png
CUDA_VISIBLE_DEVICES= python multiobjective_ehvi.py --csv color_quinary_5%.csv --plot images/multiobj_objectives.png
CUDA_VISIBLE_DEVICES= python color_dataset_generation.py --in compositions.csv --out compositions_with_color.csv
```

## Data

The scripts expect `color_quinary_5%.csv` in the repo root (already included).
Element columns are obfuscated as `E1`-`E5` to comply with Thermo-Calc terms of
service, and compositions include small jitter to prevent direct mapping between
compositions and colors.

## Scripts and outputs

- `pref_gold_optimization.py` runs single-objective preference-based BO over the
  obfuscated dataset and writes `color_progress.png` to visualize the sampling
  order and resulting preference ranking. It can also write `tournament_progress.png`
  and `rgb_projections.png`.
- `multiobjective_ehvi.py` runs multi-objective BO balancing color preference and
  price, then writes `multiobj_objectives.png` showing the trade-off and Pareto front.
- `color_dataset_generation.py` utilizes the alloy dataset `color_quinary_5%.csv` 
  by querying the Thermo-Calc Optical Properties model.

## Generate figures

Create an `images/` directory if you want LaTeX to find the figures directly:

```bash
mkdir -p images
```

```bash
python pref_gold_optimization.py --csv color_quinary_5%.csv --color-plot images/color_progress.png --tournament-plot images/tournament_progress.png
python multiobjective_ehvi.py --csv color_quinary_5%.csv --plot images/multiobj_objectives.png
python color_dataset_generation.py --in compositions.csv --out compositions_with_color.csv
```

Outputs are written with the filenames shown above.
