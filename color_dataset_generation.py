"""
Append Thermo-Calc Optical Properties color data to an existing quinary alloy dataset.

This script reads a CSV containing quinary E1–E2–E3–E4–E5 alloy compositions and uses
Thermo-Calc's Optical Properties - Noble property model to compute a 24-bit color value
for each alloy at a specified temperature (default: 298.15 K). The resulting dataset is
written to a new CSV file with an added column:

    "24-bit color"

Optionally, RGB values can also be derived and appended.

Key steps:
1. (Optional) Configure Thermo-Calc / Java / license environment variables.
2. Read compositions from an input CSV (must contain columns E1, E2, E3, E4, E5).
3. For each composition, run Thermo-Calc Optical Properties.
4. Extract the "24-bit color (Opt. Prop. - Noble)" property.
5. Write an output CSV containing the original compositions plus the color column.

Expected input CSV format:
--------------------------------------------------
E1,E2,E3,E4,E5
0.2,0.2,0.2,0.2,0.2
0.4,0.2,0.1,0.2,0.1
...

Usage example::

    python color_dataset_generation.py --in compositions.csv --out compositions_with_color.csv \
        --db TCNOBL3 --model "Optical Properties - Noble"

Notes:
- The Thermo-Calc Python API (TC-Python) requires a valid Thermo-Calc installation and license.
- Compositions must be mole fractions and should sum to 1.0.
- The "24-bit color" integer corresponds to a hexadecimal RGB value (0xRRGGBB).
"""


import argparse
import os
from typing import Sequence, Tuple

import pandas as pd
from PIL import ImageColor
from tc_python import CompositionUnit, TCPython


def thermo_calc_setup(use_placeholders: bool = False) -> None:
    """
    Set environment variables needed by Thermo-Calc / TC-Python.

    If use_placeholders=True, writes generic placeholder strings suitable for SI / docs.
    Otherwise, this function is a no-op by default (recommended) unless you want to
    hardcode local values.
    """
    if use_placeholders:
        os.environ["TC25A_HOME"] = "/path/to/Thermo-Calc-installation"
        os.environ["JAVA_HOME"] = "/path/to/java/home"
        os.environ["LSHOST"] = "license.server.hostname"
        os.environ["LSERVRC"] = "license.server.hostname"
        os.environ["TCLM_LICENSE_FILE"] = "port@license.server.hostname"
        return


def convert_color_to_rgb(color_value: int) -> Tuple[int, int, int]:
    """
    Convert a 24-bit integer color to an (R,G,B) tuple.

    Assumes color_value corresponds to 0xRRGGBB.
    """
    hex_code = hex(int(color_value))[2:].upper().zfill(6)
    return ImageColor.getcolor("#" + hex_code, "RGB")


def add_colors_from_csv(
    in_csv: str,
    out_csv: str,
    database: str,
    elements: Sequence[str],
    model_name: str,
    temperature_k: float,
    add_rgb: bool = False,
) -> None:
    """
    Read compositions from CSV, run Thermo-Calc Optical Properties, and append a
    "24-bit color" column (and optionally R,G,B) to the output CSV.

    Expected CSV columns: E1, E2, E3, E4, E5
    """
    df = pd.read_csv(in_csv)

    required = list(elements)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Ensure numeric and finite
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[required].isna().any(axis=1)
    if bad.any():
        bad_idx = df.index[bad].tolist()[:10]
        raise ValueError(
            f"Found non-numeric / missing composition values in rows (showing up to 10): {bad_idx}"
        )

    # (Optional but recommended) validate simplex-ish (tolerant)
    sums = df[required].sum(axis=1)
    if not ((sums - 1.0).abs() <= 1e-6).all():
        # Don't hard fail—Thermo-Calc may infer remainder depending on setup,
        # but warn loudly via exception so issues are caught early.
        bad_rows = df.index[((sums - 1.0).abs() > 1e-6)].tolist()[:10]
        raise ValueError(
            f"Some rows do not sum to 1.0 within tolerance. "
            f"Example bad rows (up to 10): {bad_rows}"
        )

    colors = []

    with TCPython() as session:
        system = (
            session.select_database_and_elements(database, list(elements))
            .get_system()
            .with_property_model_calculation(model_name)
            .set_temperature(temperature_k)
            .set_composition_unit(CompositionUnit.MOLE_FRACTION)
        )

        for _, row in df.iterrows():
            calc = (
                system.set_composition("E1", float(row["E1"]))
                .set_composition("E2", float(row["E2"]))
                .set_composition("E3", float(row["E3"]))
                .set_composition("E4", float(row["E4"]))
                .set_composition("E5", float(row["E5"]))
            )
            result = calc.calculate()
            color_value = result.get_value_of("24-bit color (Opt. Prop. - Noble)")
            colors.append(color_value)

    df["24-bit color"] = colors

    if add_rgb:
        rgb_tuples = df["24-bit color"].astype(int).apply(convert_color_to_rgb)
        df[["R", "G", "B"]] = pd.DataFrame(rgb_tuples.tolist(), index=df.index)

    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append Thermo-Calc Optical Properties 24-bit color to a CSV of quinary compositions."
    )
    parser.add_argument("--in", dest="in_csv", required=True, help="Input CSV with columns E1..E5")
    parser.add_argument("--out", dest="out_csv", required=True, help="Output CSV path")
    parser.add_argument("--db", default="TCNOBL3")
    parser.add_argument("--model", default="Optical Properties - Noble")
    parser.add_argument("--temp", type=float, default=298.15)
    parser.add_argument("--use-placeholders", action="store_true")
    parser.add_argument("--add-rgb", action="store_true", help="Also add R,G,B columns derived from 24-bit color")
    args = parser.parse_args()

    thermo_calc_setup(use_placeholders=args.use_placeholders)

    elements = ["E1", "E2", "E3", "E4", "E5"]

    add_colors_from_csv(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        database=args.db,
        elements=elements,
        model_name=args.model,
        temperature_k=args.temp,
        add_rgb=args.add_rgb,
    )


if __name__ == "__main__":
    main()