import os
from types import new_class
import pandas as pd
import matplotlib.gridspec as gridspec
import mplcatppuccin
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import textwrap
import numpy as np
from matplotlib.patches import FancyBboxPatch
from math import ceil

script_dir = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(script_dir, "export")
plots_dir = os.path.join(script_dir, "plots")
files = sorted(os.listdir(directory))
csv_files = {
    "total_metrics_by_subject": os.path.join(directory, "total_metrics_by_subject.csv"),
    "intervention_ratio_by_subject": os.path.join(
        directory, "intervention_ratio_by_subject.csv"
    ),
    "average_metrics_by_subject": os.path.join(
        directory, "average_metrics_by_subject.csv"
    ),
    "subject_query": os.path.join(directory, "subject.csv"),
    "type_query": os.path.join(directory, "type.csv"),
    "failing_query": os.path.join(directory, "failing.csv"),
    "total_query": os.path.join(directory, "total.csv"),
    "rating_query": os.path.join(directory, "rating.csv"),
}

mpl.style.use("mocha")


def plot_intervention_ratio_by_subject():
    df = pd.read_csv(csv_files["intervention_ratio_by_subject"])
    df = df.set_index("subject")
    ax = df.plot.bar(rot=0)
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0.0140,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)
    plt.title("Ratio of manual intervention to Copilot changes by subject")
    plt.xlabel("Subject")
    plt.ylabel("Manual Intervention Ratio")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(plots_dir, "intervention_ratio_by_subject.png"), dpi=300)


def plot_average_metrics_by_subject():
    df = pd.read_csv(csv_files["average_metrics_by_subject"])
    df = df.set_index("subject")
    ax = df.plot.bar(rot=0)
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0.0140,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)
    plt.title("Average Metrics By Subject")
    plt.xlabel("Subject")
    plt.ylabel("Average")
    plt.savefig(os.path.join(plots_dir, "average_metrics_by_subject.png"), dpi="figure")


def plot_grid_on_bars(ax, chunk_size):
    # Get the x-coordinates of the bars
    bar_x_coords = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]

    # Generate additional x-coordinates that are located halfway between each pair of bars
    additional_x_coords = [
        (bar_x_coords[i] + bar_x_coords[i + 1]) / 2
        for i in range(len(bar_x_coords) - 1)
    ]
    additional_x_coords.append(
        bar_x_coords[-1] + (bar_x_coords[-1] - bar_x_coords[-2]) / 2
    )

    # Combine the original and additional x-coordinates
    all_x_coords = sorted(bar_x_coords + additional_x_coords)

    if chunk_size % 2 == 0:
        # Create a custom grid of dots
        x = np.linspace(min(all_x_coords), max(all_x_coords), 2 * chunk_size)
        y = np.linspace(*ax.get_ylim(), 2 * chunk_size)
    else:
        x = np.linspace(min(all_x_coords), max(all_x_coords), 2 * chunk_size + 1)
        y = np.linspace(*ax.get_ylim(), 2 * chunk_size + 1)
    x, y = np.meshgrid(x, y)

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    color = mpl.rcParams["axes.edgecolor"]

    # Plot the grid of dots using scatter
    ax.scatter(
        x, y, s=4, color=color, alpha=0.9, zorder=0, edgecolors=color
    )  # Adjust size (s), color, and alpha as needed
    ax.axhline(y=0, color=color, linewidth=0.5)


def plot_subject_query():
    df = pd.read_csv(csv_files["subject_query"])
    df = df.set_index("subject")

    num_plots = 3
    chunk_size = ceil(len(df) / num_plots)

    for i in range(num_plots):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]

        # Increase the width of the figure
        plt.figure(figsize=(30, 50))
        ax = chunk_df.plot.bar(rot=0, width=0.3)

        plt.title("Average Rating By Subject")
        plt.xlabel("Subject")
        plt.ylabel("Average Rating")

        plot_grid_on_bars(ax, chunk_size)

        new_patches = []
        for patch in reversed(ax.patches):
            bb = patch.get_bbox()
            color = patch.get_facecolor()
            p_bbox = FancyBboxPatch(
                (bb.xmin, bb.ymin),
                abs(bb.width),
                abs(bb.height),
                boxstyle="round,pad=-0,rounding_size=0.015",
                ec="none",
                fc=color,
            )
            patch.remove()
            new_patches.append(p_bbox)
        for patch in new_patches:
            ax.add_patch(patch)

        plt.xticks(fontsize=8, rotation=45)
        wrapper = textwrap.TextWrapper(
            width=10, break_long_words=False, break_on_hyphens=False
        )
        labels = [wrapper.fill(label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, f"subject_query_{i + 1}.png"), dpi=300)
        plt.clf()  # Clear the current figure for the next plot


def plot_type_query():
    df = pd.read_csv(csv_files["type_query"])
    df = df.set_index("type")

    ax = plt.gca()
    bars = ax.bar(df.index, df["average_rating"], width=0.2)

    plt.title("Average Rating By Type")
    plt.xlabel("Type")
    plt.ylabel("Average Rating")

    plt.tight_layout()
    plot_grid_on_bars(ax, len(df))

    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)

    plt.savefig(os.path.join(plots_dir, "type.png"), dpi=300)


def plot_rating_query():
    df = pd.read_csv(csv_files["rating_query"])
    df = df.set_index("rating")

    ax = plt.gca()
    bars = ax.bar(df.index, df["percentage"], width=0.2)

    plt.title("Percentage of Each Rating")
    plt.xlabel("Rating")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plot_grid_on_bars(ax, len(df))

    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)

    plt.savefig(os.path.join(plots_dir, "rating_perc.png"), dpi=300)

    plt.clf()

    ax = plt.gca()
    bars = ax.bar(df.index, df["Times Rated"], width=0.2)

    plt.title("Number of Times Each Rating Was Given")
    plt.xlabel("Rating")
    plt.ylabel("Times Rated")

    plt.tight_layout()
    plot_grid_on_bars(ax, len(df))

    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)

    plt.savefig(os.path.join(plots_dir, "rating_times.png"), dpi=300)


def plot_total_metrics_by_subject():
    df = pd.read_csv(csv_files["total_metrics_by_subject"])
    df = df.set_index("subject")
    ax = df.plot.bar(rot=0)
    new_patches = []

    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0.0140,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)
    plt.title("Total Metrics By Subject")
    fig = plt.gcf()
    fig.set_figwidth(10)
    plt.xlabel("")
    plt.ylabel("Instances")
    plt.savefig(os.path.join(plots_dir, "total_metrics_by_subject.png"), dpi=300)


plot_rating_query()
plot_intervention_ratio_by_subject()
plot_average_metrics_by_subject()
plot_total_metrics_by_subject()
plot_subject_query()
plot_type_query()
