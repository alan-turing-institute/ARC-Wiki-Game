from __future__ import annotations

import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

fig_path = os.path.join(os.path.dirname(__file__), "fig1_graph.pdf")
metadata_path = os.path.join(os.path.dirname(__file__), "dataset_metadata.csv")

hist_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "graph-prep/data/metadata",
)


## Get Data ##

metadata_df = pd.read_csv(metadata_path)

topic_name = [
    ("enwiki", "English"),
    ("frwiki", "French"),
    ("ruwiki", "Russian"),
    ("zhwiki", "Chinese"),
    ("bnwiki", "Bengali"),
    ("oa_physics", "Physics"),
    ("oa_biochemistry", "Biology"),
    ("oa_medicine", "Medicine"),
    ("oa_social_sciences", "Social Sci"),
    ("oa_arts", "Arts"),
]

histograms = {}

for dataset, topic in topic_name:
    hist_df = pd.read_csv(os.path.join(hist_path, f"{dataset}_histogram.csv"))
    steps, counts = hist_df.values.T
    counts = counts / np.sum(counts)
    counts[0] = 0
    counts = 100 * np.cumsum(counts)
    histograms[topic] = (steps, counts)


##  Set up Figure ##

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

ax1 = axes[0]
ax2 = axes[1]

ax1.text(
    -0.1,
    1.0,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax2.text(
    -0.1,
    1.0,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)

## Axis 1 - Article Count and Links per Article ##

x_labels = [
    f"{topic}"
    for dataset, topic in zip(
        metadata_df["Dataset"], metadata_df["Topic"], strict=False
    )
]
x_vals = np.arange(len(metadata_df))  # the label locations

# bar chart for article count

bars = ax1.bar(x_vals, metadata_df["Article"], color="blue", label="Article")
ax1.set_xlabel("Wikipedia" + " " * 30 + "Open Alex")
ax1.set_ylabel("Article count in millions", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(x_vals)
ax1.set_xticklabels(x_labels, rotation=60, ha="right")

y_top = 21
y_ticks = np.arange(0, y_top, 2)
ax1.set_ylim(0, y_top)
ax1.set_yticks(y_ticks)

ax1.grid(axis="y", linestyle="--", alpha=0.7)

# line chart for links per article

ax1t = ax1.twinx()
ax1t.plot(
    x_vals,
    metadata_df["LinkPage"],
    color="red",
    ls="",
    marker="o",
    ms=8,
    label="Links per Article",
)
ax1t.set_ylabel("Mean links per article", color="red")
ax1t.tick_params(axis="y", labelcolor="red")
ax1t.set_ylim(0, y_top * 2.5)
ax1t.set_yticks(y_ticks * 2.5)

## Axis 2 - Cumulative Percentage of Links ##

topic_points = dict(
    zip(
        [t[1] for t in topic_name],
        [
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
            (6, 0, 0),
            (7, 0, 0),
            (4, 1, 45),
            (5, 1, 45),
            (6, 1, 45),
            (7, 1, 45),
            (8, 1, 45),
        ],
        strict=False,
    )
)

sorted_histograms = dict(sorted(histograms.items(), key=lambda x: -x[1][1][-1]))


for n, (topic, hist) in enumerate(sorted_histograms.items()):
    ax2.plot(
        hist[0],
        hist[1],
        label=topic,
        color=cc.glasbey_dark[n],
        marker=topic_points[topic],
        linestyle="--" if topic_points[topic][1] else "-",
    )


ax2.set_xlim(0, 10)
ax2.set_ylim(0, 100)
ax2.set_xlabel("Steps through link graph")
ax2.set_ylabel("Percentage of reachable articles")

ax2.grid(alpha=0.7)
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which="both", direction="in")

ax2.legend(fontsize="small", loc="upper left", framealpha=1.0)


fig.tight_layout()  # Adjust layout to prevent overlap

plt.savefig(fig_path)
