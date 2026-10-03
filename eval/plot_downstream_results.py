"""
Plots for the downstream cloud-tracking evaluation. Reads real numbers
from outputs/downstream_results.json (written by run_downstream_eval.py)
instead of hardcoding them -- rerun run_downstream_eval.py first whenever
the model, sample count, or thresholds change, then rerun this script.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("outputs/downstream_results.json")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run eval/run_downstream_eval.py first."
        )

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    labels = results["methods"]  # {"average": "Average Baseline", ...}
    methods = list(labels.keys())
    method_titles = [labels[m].replace(" ", "\n", 1) for m in methods]
    stats = results["stats"]

    mean_errors = [stats[m]["mean_error_px"] for m in methods]
    median_errors = [stats[m]["median_error_px"] for m in methods]
    p90_errors = [stats[m]["p90_error_px"] for m in methods]
    wins = [results["win_counts"][m] for m in methods]

    n = results["valid_samples"]
    fig_width = max(7, 1.6 * len(methods))

    # 1. Mean vs Median Bar Chart
    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(fig_width, 5))
    plt.bar(x - width / 2, mean_errors, width, label="Mean Error")
    plt.bar(x + width / 2, median_errors, width, label="Median Error")
    plt.xticks(x, method_titles)
    plt.ylabel("Centroid Tracking Error (pixels)")
    plt.title(f"Downstream Cloud-Tracking Error (n={n})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "downstream_mean_median_error.png", dpi=200)
    plt.close()
    print("Saved:", OUT_DIR / "downstream_mean_median_error.png")

    # 2. 90th Percentile Bar Chart
    plt.figure(figsize=(fig_width, 5))
    plt.bar(method_titles, p90_errors)
    plt.ylabel("90th Percentile Error (pixels)")
    plt.title(f"High-Error Case Comparison (n={n})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "downstream_90th_percentile_error.png", dpi=200)
    plt.close()
    print("Saved:", OUT_DIR / "downstream_90th_percentile_error.png")

    # 3. Head-to-head Wins
    plt.figure(figsize=(fig_width, 5))
    plt.bar(method_titles, wins)
    plt.ylabel(f"Number of Samples Won (best of {len(methods)} methods)")
    plt.title(f"Head-to-Head Downstream Tracking Wins (n={n})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "downstream_head_to_head_wins.png", dpi=200)
    plt.close()
    print("Saved:", OUT_DIR / "downstream_head_to_head_wins.png")


if __name__ == "__main__":
    main()
