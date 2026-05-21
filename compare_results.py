"""
Compare test results across models (Qwen-1.8B / Llama2-7B / ChatGLM3-6B).
Usage: conda run -n MSE-Adapter python compare_results.py
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_DIRS = {
    "Qwen-1.8B":   ROOT / "MSE-Qwen-1.8B"  / "results" / "results",
    "Llama2-7B":   ROOT / "MSE-Llama2-7B"  / "results" / "results",
    "ChatGLM3-6B": ROOT / "MSE-ChatGLM3-6B" / "results" / "results",
}

SKIP_COLS = {"Model", "Seed"}


def load_latest(directory: Path, pattern: str) -> pd.DataFrame | None:
    """Load the CSV matching pattern with the largest numeric suffix; return None if not found."""
    matches = sorted(directory.glob(pattern), key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    if not matches:
        return None
    return pd.read_csv(matches[-1])


def collect_dataset(pattern: str) -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame} for all models that have this dataset."""
    return {
        model: df
        for model, directory in MODEL_DIRS.items()
        if (df := load_latest(directory, pattern)) is not None
    }


def collect_all_tables() -> list[tuple[str, str, pd.DataFrame]]:
    """Return list of (dataset_name, task, table_df) for all available datasets."""
    datasets = [
        ("meld-classification-*.csv",   "MELD",   "classification"),
        ("simsv2-regression-*.csv",      "SIMSV2", "regression"),
        ("mosei-regression-*.csv",       "MOSEI",  "regression"),
        ("cherma-classification-*.csv",  "CHERMA", "classification"),
    ]
    tables = []
    for pattern, name, task in datasets:
        frames = collect_dataset(pattern)
        if not frames:
            continue
        rows = []
        for model, df in frames.items():
            metric_cols = [c for c in df.columns if c not in SKIP_COLS]
            row = df[metric_cols].iloc[0].rename(model)
            rows.append(row)
        # union of all columns; models missing a column get NaN
        table = pd.DataFrame(rows).round(2)
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
