import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Written in part by gemini 3.1 Pro and Claude Opus 4.6
# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# The 9 metrics in the order defined/plotted in gridsearch.py
METRICS_INFO = [
    ("continuity",       "Continuity"),
    ("trustworthiness",  "Trustworthiness"),
    ("cluster_ordering", "Cluster Ordering"),
    ("pearson",          "Pearson Correlation"),
    ("spearman",         "Spearman Correlation"),
    ("silhouette",       "Silhouette Score"),
    ("average_metrics",  "Average Metrics"),
    ("wall_clock_time",  "CPU Process Time (s)"),
    ("dbscan_clusters",  "DBSCAN # Clusters"),
]

# Fixed vmin/vmax for score-like metrics; omitted keys use auto-scale
METRIC_RANGES = {
    "continuity":       (0.0,  1.0),
    "trustworthiness":  (0.0,  1.0),
    "cluster_ordering": (-1.0, 1.0),
    "pearson":          (-1.0, 1.0),
    "spearman":         (-1.0, 1.0),
    "silhouette":       (-1.0, 1.0),
    "average_metrics":  (-1.0, 1.0),
    # dbscan_clusters -> auto-scale
}

# Metrics that benefit from a diverging colormap centred at 0
DIVERGING_METRICS = {"pearson", "cluster_ordering", "spearman"}


def _cmap_for(metric_key):
    return "PRGn" if metric_key in DIVERGING_METRICS else "viridis"


def _fmt_for(metric_key):
    return ".0f" if metric_key == "dbscan_clusters" else ".3f"


def _str_from_npz(val):
    """Safely decode a scalar string stored in an .npz file."""
    if val is None:
        return None
    v = np.asarray(val)
    return str(v.item()) if v.ndim == 0 else str(val)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(file_path):
    """Load a single .npz file and return a run_data dict."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error: Failed to load '{file_path}': {e}")
        sys.exit(1)

    # Retrieve grid parameters
    output_dimensions = data.get("output_dimensions", data.get("outputDimensions", None))
    if output_dimensions is None:
        for k in data:
            if "dimension" in k.lower():
                output_dimensions = data[k]
                break

    epsilons = data.get("epsilons", None)

    if epsilons is None or output_dimensions is None:
        print(f"Error: Missing epsilons or output_dimensions in '{file_path}'. "
              f"Available keys: {list(data.keys())}")
        sys.exit(1)

    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    run_data = {
        "file_basename":        file_basename,
        "epsilons":             epsilons,
        "output_dimensions":    output_dimensions,
        "embedding_model":      _str_from_npz(data.get("embeddingModel",           None)),
        "primary_dim_reduct":   _str_from_npz(data.get("primaryDimReductType",     None)),
        "secondary_dim_reduct": _str_from_npz(data.get("secondaryDimReductType",   None)),
        "metrics": {}
    }

    for metric_key, _ in METRICS_INFO:
        metric_data = data.get(metric_key, None)
        if metric_data is None:
            if metric_key == "average_metrics":
                try:
                    metric_data = (
                        data["continuity"] + data["trustworthiness"]
                        + np.abs(data["cluster_ordering"])
                        + np.abs(data["pearson"])
                        + np.abs(data["spearman"])
                        + data["silhouette"]
                    ) / 6.0
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Warning: Could not compute 'average_metrics' on the fly for '{file_path}'. Error: {e}")
                    metric_data = np.zeros((len(epsilons), len(output_dimensions)))
            else:
                print(f"Warning: Metric '{metric_key}' not found in '{file_path}'. Using zeros.")
                metric_data = np.zeros((len(epsilons), len(output_dimensions)))
        run_data["metrics"][metric_key] = metric_data

    return run_data


def load_run_group(files):
    """Load and cross-validate a list of .npz files belonging to one group."""
    loaded_runs = []
    for file_path in files:
        run_data = load_run(file_path)
        if loaded_runs:
            first = loaded_runs[0]
            if not np.array_equal(run_data["epsilons"], first["epsilons"]):
                print(f"Error: Epsilon parameters in '{file_path}' do not match '{files[0]}'.")
                sys.exit(1)
            if not np.array_equal(run_data["output_dimensions"], first["output_dimensions"]):
                print(f"Error: Output dimensions in '{file_path}' do not match '{files[0]}'.")
                sys.exit(1)
        loaded_runs.append(run_data)
    return loaded_runs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _suptitle(run):
    em = run.get("embedding_model")      or "Unknown"
    pr = run.get("primary_dim_reduct")   or "Unknown"
    sr = run.get("secondary_dim_reduct") or "Unknown"
    return (f"Embedding: {em}  |  Primary Dim Reduction: {pr}  "
            f"|  Secondary Dim Reduction: {sr}")


def plot_comparison(loaded_runs, output_path):
    """Plot Nx9 comparison heatmap grid and save to output_path."""
    N = len(loaded_runs)
    num_metrics = len(METRICS_INFO)
    print(f"  Generating {N}x{num_metrics} comparison plot -> {output_path}")

    fig, axes = plt.subplots(N, num_metrics,
                             figsize=(6 * num_metrics, 5 * N), squeeze=False)
    fig.suptitle(_suptitle(loaded_runs[0]), fontsize=14, fontweight="bold", y=1.002)

    for i, run_data in enumerate(loaded_runs):
        for j, (metric_key, metric_title) in enumerate(METRICS_INFO):
            ax = axes[i, j]
            vmin, vmax = METRIC_RANGES.get(metric_key, (None, None))
            sns.heatmap(
                run_data["metrics"][metric_key],
                xticklabels=run_data["output_dimensions"],
                yticklabels=run_data["epsilons"],
                annot=True,
                fmt=_fmt_for(metric_key),
                cmap=_cmap_for(metric_key),
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            ax.set_title(f"{metric_title}\n({run_data['file_basename']})")
            ax.set_xlabel("Output Dimension")
            ax.set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_average(loaded_runs, output_path):
    """Plot 1x9 average heatmap grid and save to output_path."""
    N = len(loaded_runs)
    num_metrics = len(METRICS_INFO)
    print(f"  Generating 1x{num_metrics} average plot -> {output_path}")

    first = loaded_runs[0]
    fig_avg, axes_avg = plt.subplots(1, num_metrics,
                                     figsize=(6 * num_metrics, 5), squeeze=False)
    fig_avg.suptitle(_suptitle(first), fontsize=14, fontweight="bold", y=1.05)

    for j, (metric_key, metric_title) in enumerate(METRICS_INFO):
        avg_data = np.mean([r["metrics"][metric_key] for r in loaded_runs], axis=0)
        ax = axes_avg[0, j]
        vmin, vmax = METRIC_RANGES.get(metric_key, (None, None))
        sns.heatmap(
            avg_data,
            xticklabels=first["output_dimensions"],
            yticklabels=first["epsilons"],
            annot=True,
            fmt=_fmt_for(metric_key),
            cmap=_cmap_for(metric_key),
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(f"Average {metric_title}\n(N={N} runs)")
        ax.set_xlabel("Output Dimension")
        ax.set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_avg)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Directory crawling
# ---------------------------------------------------------------------------

def crawl_and_plot(root_dir):
    """
    Walk root_dir recursively. For every leaf directory that contains .npz
    files (excluding embeddingdata* files), generate a comparison plot and an
    average plot saved inside that same directory.

    Expected layout::

        root_dir/
          <embedding_model>/
            <primary_secondary_dimreduct>/
              data1.npz
              data2.npz
    """
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    # Collect directories that contain .npz data files
    groups = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()  # deterministic traversal order
        npz_files = sorted(
            os.path.join(dirpath, f)
            for f in filenames
            if f.endswith(".npz") and not f.startswith("embeddingdata")
        )
        if npz_files:
            groups[dirpath] = npz_files

    if not groups:
        print(f"No .npz data files found under '{root_dir}'.")
        sys.exit(1)

    total = len(groups)
    print(f"Found {total} group(s) of .npz files under '{root_dir}'.\n")

    for idx, (dirpath, files) in enumerate(groups.items(), start=1):
        rel = os.path.relpath(dirpath, root_dir)
        print(f"[{idx}/{total}] Processing group: {rel}  ({len(files)} file(s))")

        loaded_runs = load_run_group(files)

        out_comparison = os.path.join(dirpath, "gridsearch_comparison.png")
        out_average    = os.path.join(dirpath, "gridsearch_comparison_average.png")

        plot_comparison(loaded_runs, out_comparison)
        plot_average(loaded_runs, out_average)
        print()

    print(f"Done. Generated plots for {total} group(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot gridsearch results comparison as an Nx9 grid of heatmaps."
    )
    parser.add_argument(
        "-f", "--files", nargs="*",
        help="Paths to gridsearch .npz result files (optional if -d or -c is used)"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Directory containing gridsearch .npz result files to load (flat, one group)"
    )
    parser.add_argument(
        "-c", "--crawl",
        help=(
            "Root directory to crawl recursively. Every sub-directory that "
            "contains .npz files is treated as one run-group and gets its own "
            "pair of plots saved inside that sub-directory. "
            "Expected layout: <root>/<embedding>/<primary_secondary_dimreduct>/*.npz"
        )
    )
    parser.add_argument(
        "-o", "--output", default="gridsearch_comparison.png",
        help="Output path for comparison plot (used with -d or positional files)"
    )
    parser.add_argument(
        "-ao", "--average-output", default="gridsearch_comparison_average.png",
        help="Output path for average plot (used with -d or positional files)"
    )
    args = parser.parse_args()

    # ---- Mode 1: crawl an entire runs/ tree --------------------------------
    if args.crawl:
        crawl_and_plot(args.crawl)
        return

    # ---- Mode 2: single flat directory or explicit file list ---------------
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory '{args.directory}' does not exist.")
            sys.exit(1)
        dir_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.endswith(".npz")]
        # Filter out raw embeddingdata files
        files = sorted(f for f in dir_files if not os.path.basename(f).startswith("embeddingdata"))
        if not files:
            print(f"Error: No grid search .npz files found in directory '{args.directory}'.")
            sys.exit(1)
    else:
        files = args.files
        if not files:
            print(
                "Error: You must specify either a list of .npz files, "
                "a directory (-d/--directory), or a crawl root (-c/--crawl)."
            )
            parser.print_help()
            sys.exit(1)

    loaded_runs = load_run_group(files)
    plot_comparison(loaded_runs, args.output)
    plot_average(loaded_runs, args.average_output)


if __name__ == "__main__":
    main()
