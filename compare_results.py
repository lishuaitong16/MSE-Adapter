"""
Compare test results across models (Qwen-1.8B / Llama2-7B / ChatGLM3-6B).
Usage: conda run -n MSE-Adapter python compare_results.py
"""

import os
import glob
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_DIRS = {
    "Qwen-1.8B":    ROOT / "MSE-Qwen-1.8B"   / "results" / "results",
    "Llama2-7B":    ROOT / "MSE-Llama2-7B"   / "results" / "results",
    "ChatGLM3-6B":  ROOT / "MSE-ChatGLM3-6B" / "results" / "results",
}

# Canonical column order per task (subset that all files share)
REGRESSION_COLS  = ["Mult_acc_2", "Mult_acc_2_weak", "F1_score", "MAE", "Corr", "Mult_acc_3", "Mult_acc_5", "R_squre"]
CLASSIFICATION_COLS = ["acc", "weight_F1"]


def load_latest(directory: Path, pattern: str) -> pd.DataFrame | None:
    """Load the first CSV matching pattern; return None if not found."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    df = pd.read_csv(matches[-1])
    df.attrs["source"] = matches[-1].name
    return df


def collect_dataset(dataset_glob: str, task: str) -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame} for all models that have this dataset."""
    results = {}
    for model, directory in MODEL_DIRS.items():
        df = load_latest(directory, dataset_glob)
        if df is not None:
            results[model] = df
    return results


def mean_row(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return mean of numeric columns present in df."""
    available = [c for c in cols if c in df.columns]
    return df[available].mean().rename("mean")


def print_table(dataset_name: str, task: str, model_frames: dict[str, pd.DataFrame]):
    cols = REGRESSION_COLS if task == "regression" else CLASSIFICATION_COLS

    rows = []
    for model, df in model_frames.items():
        available = [c for c in cols if c in df.columns]
        row = df[available].iloc[0].rename(model)  # single-seed: take first row
        rows.append(row)

    table = pd.DataFrame(rows).round(2)
    # reorder columns to canonical order
    ordered = [c for c in cols if c in table.columns]
    table = table[ordered]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Dataset : {dataset_name.upper()}  |  Task : {task}")
    print(sep)
    print(table.to_string())
    print()


def collect_all_tables() -> list[tuple[str, str, pd.DataFrame]]:
    """Return list of (dataset_name, task, table_df) for all available datasets."""
    datasets = [
        ("meld-classification-*.csv",  "MELD",   "classification"),
        ("simsv2-regression-*.csv",    "SIMSV2", "regression"),
        ("mosei-regression-*.csv",     "MOSEI",  "regression"),
        ("cherma-classification-*.csv","CHERMA", "classification"),
    ]
    tables = []
    for pattern, name, task in datasets:
        frames = collect_dataset(pattern, task)
        if frames:
            cols = REGRESSION_COLS if task == "regression" else CLASSIFICATION_COLS
            rows = []
            for model, df in frames.items():
                available = [c for c in cols if c in df.columns]
                row = df[available].iloc[0].rename(model)
                rows.append(row)
            table = pd.DataFrame(rows).round(2)
            ordered = [c for c in cols if c in table.columns]
            table = table[ordered]
            tables.append((name, task, table))
    return tables


def main():
    tables = collect_all_tables()

    if not tables:
        print("No results found.")
        return

    for name, task, table in tables:
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  Dataset : {name.upper()}  |  Task : {task}")
        print(sep)
        print(table.to_string())
        print()

    # Save one CSV per dataset
    for name, task, table in tables:
        t = table.copy()
        t.index.name = "Model"
        save_path = ROOT / f"comparison_{name.lower()}.csv"
        t.reset_index().to_csv(save_path, index=False)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
