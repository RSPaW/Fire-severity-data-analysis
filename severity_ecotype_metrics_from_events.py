#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to read pickeld data extracted from R data files (RDS) and make some
initial plots."""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_eco_statistics(eco_value, event_data):
    """Plot relationships between sev, solb, and yslt parameters by eco type.

    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame containing the event data with columns: eco, sev, solb, yslt
    stub : str, optional
        Filename prefix for saved plots (default: "output")

    Returns
    -------
    dict
        Statistics dictionary for this eco type
    """
    eco_data = event_data[event_data["eco"] == eco_value]
    corr = eco_data[["sev", "solb", "yslt"]].corr()

    return {
        "eco": eco_value,
        "n": len(eco_data),
        "sev_mean": eco_data["sev"].mean(),
        "sev_std": eco_data["sev"].std(),
        "sev_median": eco_data["sev"].median(),
        "solb_mean": eco_data["solb"].mean(),
        "solb_std": eco_data["solb"].std(),
        "solb_median": eco_data["solb"].median(),
        "yslt_mean": eco_data["yslt"].mean(),
        "yslt_std": eco_data["yslt"].std(),
        "yslt_median": eco_data["yslt"].median(),
        "corr_sev_solb": corr.loc["sev", "solb"],
        "corr_sev_yslt": corr.loc["sev", "yslt"],
        "corr_solb_yslt": corr.loc["solb", "yslt"],
    }


def plot_parameter_relationships(
    event_data, stub="output", use_parallel=True, plot_type="hexbin"
):
    """Plot relationships between sev, solb, and yslt parameters by eco type.

    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame containing the event data with columns: eco, sev, solb, yslt
    stub : str, optional
        Filename prefix for saved plots (default: "output")
    use_parallel : bool, optional
        If True, use parallel processing for statistics computation.
        Does not affect plotting as matplotlib is not thread-safe.
        (default: True).
    plot_type : str, optional
        Type of plot to generate:
        - "hexbin": Hexagonal binning (fast, shows density) - default
        - "scatter": Scatter plot (slow for large datasets)
        - "violin": Violin plots (shows distributions by bins)
        (default: "hexbin")
    """
    eco_types = sorted(event_data["eco"].unique())
    n_ecos = len(eco_types)
    print(f"\nGenerating plots for {n_ecos} eco types:")
    for eco in eco_types:
        print(f"  > {eco}")
    print(f"Total records: {len(event_data)}")

    # Filter out rows with NaNs in the columns being plotted
    keep_cols = ["solb", "sev", "yslt", "eco"]
    event_data_clean = event_data[keep_cols].dropna()
    print(event_data_clean.describe())
    n_removed = len(event_data) - len(event_data_clean)
    if n_removed > 0:
        print(
            f"Removed {n_removed} rows with NaN values "
            f"({100 * n_removed / len(event_data):.1f}%)"
        )

    # =====================================================================
    # Plot 1: Severity vs Severity of Last Burn,
    # by Eco Type (coloured by Years Since Last Treatment)
    # =====================================================================
    print(f"\n[1/4] Creating Severity vs SOLB plot ({plot_type})...")
    fig, axes = plt.subplots(n_ecos, 1, figsize=(12, 3 * n_ecos), sharex=True)
    if n_ecos == 1:
        axes = [axes]

    for idx, eco_value in enumerate(eco_types):
        eco_data = event_data_clean[
            event_data_clean["eco"] == eco_value
        ].copy()
        print(f"  > Eco Type: {eco_value} (n={len(eco_data)})")

        ax = axes[idx]

        if plot_type == "hexbin":
            # Hexagonal binning - much faster for large datasets
            hexbin = ax.hexbin(
                eco_data["solb"],
                eco_data["sev"],
                C=eco_data["yslt"],
                gridsize=50,
                cmap="viridis",
                mincnt=1,
                reduce_C_function=np.median,
            )
            ax.set_xlabel("Severity of Last Burn (solb)")
            ax.set_ylabel("Severity (sev)")
            ax.set_title(
                f"Eco Type: {eco_value} - Severity vs SOLB "
                "(coloured by median YSLT)"
            )
            cbar = plt.colorbar(hexbin, ax=ax)
            cbar.set_label("Median YSLT")
        else:  # scatter
            scatter = ax.scatter(
                eco_data["solb"],
                eco_data["sev"],
                c=eco_data["yslt"],
                cmap="viridis",
                alpha=0.3,
                s=10,
                rasterized=True,
            )
            ax.set_xlabel("Severity of Last Burn (solb)")
            ax.set_ylabel("Severity (sev)")
            ax.set_title(
                f"Eco Type: {eco_value} - Severity vs SOLB (coloured by YSLT)"
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("YSLT")

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{stub}_sev_vs_solb_by_eco_{plot_type}.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {stub}_sev_vs_solb_by_eco_{plot_type}.png")

    # =====================================================================
    # Plot 2: Severity vs YSLT by Eco Type (coloured by SOB)
    # =====================================================================
    print(f"[2/4] Creating Severity vs YSLT plot ({plot_type})...")
    fig, axes = plt.subplots(n_ecos, 1, figsize=(12, 3 * n_ecos), sharex=True)
    if n_ecos == 1:
        axes = [axes]

    for idx, eco_value in enumerate(eco_types):
        eco_data = event_data_clean[
            event_data_clean["eco"] == eco_value
        ].copy()
        print(f"  > Eco Type: {eco_value} (n={len(eco_data)})")

        ax = axes[idx]

        if plot_type == "hexbin":
            # Hexagonal binning - much faster for large datasets
            hexbin = ax.hexbin(
                eco_data["yslt"],
                eco_data["sev"],
                C=eco_data["solb"],
                gridsize=50,
                cmap="coolwarm",
                mincnt=1,
                reduce_C_function=np.median,
            )
            ax.set_xlabel("Years Since Last Treatment (yslt)")
            ax.set_ylabel("Severity (sev)")
            ax.set_title(
                f"Eco Type: {eco_value} - Severity vs YSLT "
                "(coloured by median SOLB)"
            )
            cbar = plt.colorbar(hexbin, ax=ax)
            cbar.set_label("Median SOLB")
        else:  # scatter
            scatter = ax.scatter(
                eco_data["yslt"],
                eco_data["sev"],
                c=eco_data["solb"],
                cmap="coolwarm",
                alpha=0.3,
                s=10,
                rasterized=True,
            )
            ax.set_xlabel("Years Since Last Treatment (yslt)")
            ax.set_ylabel("Severity (sev)")
            ax.set_title(
                f"Eco Type: {eco_value} - Severity vs YSLT (coloured by SOLB)"
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("SOLB")

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{stub}_sev_vs_yslt_by_eco_{plot_type}.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {stub}_sev_vs_yslt_by_eco_{plot_type}.png")

    # =====================================================================
    # Correlation matrices by Eco Type
    # =====================================================================
    print("[3/4] Creating correlation matrices...")
    fig, axes = plt.subplots(1, n_ecos, figsize=(5 * n_ecos, 4))
    if n_ecos == 1:
        axes = [axes]

    for idx, eco_value in enumerate(eco_types):
        eco_data = event_data[event_data["eco"] == eco_value]
        print(f"  > Eco Type: {eco_value} (n={len(eco_data)})")
        corr_matrix = eco_data[["sev", "solb", "yslt"]].corr()

        ax = axes[idx]
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Set ticks and labels
        ticks = np.arange(len(["sev", "solb", "yslt"]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(["sev", "solb", "yslt"])
        ax.set_yticklabels(["sev", "solb", "yslt"])

        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_title(f"{eco_value}")
        plt.colorbar(im, ax=ax, label="Correlation")

    plt.savefig(f"{stub}_corr_by_eco.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {stub}_corr_by_eco.png")

    # =====================================================================
    # Print Summary Statistics by Eco Type
    # =====================================================================
    print("[4/4] Computing summary statistics...")

    # Compute statistics in parallel if requested
    if use_parallel:
        n_workers = min(len(eco_types), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            stats_list = list(
                executor.map(
                    lambda eco: compute_eco_statistics(eco, event_data_clean),
                    eco_types,
                )
            )
    else:
        stats_list = [
            compute_eco_statistics(eco, event_data_clean) for eco in eco_types
        ]

    # Print statistics
    for s in stats_list:
        print(f"\nEco Type: {s['eco']} (n={s['n']})")
        for param in ["sev", "solb", "yslt"]:
            print(
                f"  {param}: mean={s[f'{param}_mean']:.3f}, "
                f"std={s[f'{param}_std']:.3f}, "
                f"median={s[f'{param}_median']:.3f}"
            )
        # print(
        #     f" sev: mean={s['sev_mean']:.3f}, "
        #     f"std={s['sev_std']:.3f}, "
        #     f"median={s['sev_median']:.3f}"
        # )
        # print(
        #     f"solb: mean={s['solb_mean']:.3f}, std={s['solb_std']:.3f}, median={s['solb_median']:.3f}"
        # )
        # print(
        #     f"yslt: mean={s['yslt_mean']:.3f}, std={s['yslt_std']:.3f}, median={s['yslt_median']:.3f}"
        # )
        print("  Correlations:")
        print(f"    sev-solb: {s['corr_sev_solb']:.3f}")
        print(f"    sev-yslt: {s['corr_sev_yslt']:.3f}")
        print(f"    solb-yslt: {s['corr_solb_yslt']:.3f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    data_file = Path(sys.argv[1])
    plot_type = sys.argv[2] if len(sys.argv) > 2 else "hexbin"

    if plot_type not in ["hexbin", "scatter"]:
        print(
            f"Invalid plot type '{plot_type}'. Must be one of: hexbin, scatter"
        )
        print("NOTE: scatter plots can be very slow for large datasets.\n")
        print("Usage: python extract_and_plot.py <data_file.pkl> [plot_type]")
        sys.exit(1)

    print(f"Loading data from {data_file}...")
    event_data = pd.read_pickle(data_file)
    print("Data loaded successfully!")

    print("\nData summary:")
    print(event_data.describe())

    # Generate plots
    plot_parameter_relationships(event_data, "total", plot_type=plot_type)
