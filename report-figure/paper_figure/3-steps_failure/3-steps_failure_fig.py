from __future__ import annotations

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig3_steps_failure.pdf"

results_dir = current_dir.parents[2] / "agents" / "result" / "table"
embed_path = results_dir / "embed_model_wiki_summary_stats.csv"
prompt_path = results_dir / "prompt_wiki_summary_stats.csv"
tfidf_path = results_dir / "tfidf_wiki_summary_stats.csv"
reasoning_path = results_dir / "reasoning_wiki_summary_stats.csv"
human_path = results_dir / "human_wiki_summary_stats.csv"

embed_count_path = results_dir / "embed_model_wiki_summary_game_count.csv"
prompt_count_path = results_dir / "prompt_wiki_summary_game_count.csv"
tfidf_count_path = results_dir / "tfidf_wiki_summary_game_count.csv"
reasoning_count_path = results_dir / "reasoning_wiki_summary_game_count.csv"

# Load Data

color_hue = {
    "TFIDF": 298 / 360,
    "Embedding": 28 / 360,
    "LLM": 185 / 360,
    "Reasoning LLM": 112 / 360,
    "Human": 355 / 360,
}

# Varying the brightness as well as the hue keeps the classes apart in greyscale.
color_val = {
    "TFIDF": 0.8,
    "Embedding": 0.85,
    "LLM": 0.95,
    "Reasoning LLM": 0.9,
    "Human": 0.85,
}

embed_df = pd.read_csv(embed_path, sep="\t")
embed_df["Colour_name"] = "Embedding"

prompt_df = pd.read_csv(prompt_path, sep="\t")
prompt_df["Colour_name"] = "LLM"

tfidf_df = pd.read_csv(tfidf_path, sep="\t")
tfidf_df["Colour_name"] = "TFIDF"

reasoning_df = pd.read_csv(reasoning_path, sep="\t")
# Distinguishes the models run through both agents, e.g. Llama3.1:405B.
reasoning_df["Model"] = reasoning_df["Model"] + " (R)"
reasoning_df["Colour_name"] = "Reasoning LLM"

human_df = pd.read_csv(human_path, sep="\t")
human_df["Colour_name"] = "Human"


all_df = pd.concat([prompt_df, reasoning_df, embed_df, tfidf_df, human_df])

all_df = all_df[(all_df["Language"] == "English") & (all_df["Topic"] == "Wikipedia")]

# Closed source models are marked with a hatch rather than a colour of their own.
all_df["Hatch"] = all_df["Model"].str.contains("GPT")


rename_dict = {
    "All-MPNet-Base v2": "All-MPNet-Base",
    "Intfloat Multilingual-E5-Small": "E5-Small",
    "Average Word Embeddings Glove 6B300d": "Glove Word",
    "Roberta-Large v1": "Roberta-Large",
    "GPT 4o": "GPT-4o",
    "GPT 4o mini": "GPT-4o mini",
    "GPT-o3-mini (R)": "GPT-o3 mini (R)",
    "DeepSeek-R1:8B (R)": "DS-R1:8B (R)",
    "DeepSeek-R1:32B (R)": "DS-R1:32B (R)",
    "DeepSeek-R1:70B (R)": "DS-R1:70B (R)",
    "TFIDF Self Corpus": "TFIDF",
}

all_df["ModelName"] = all_df["Model"]
all_df["ModelName"] = all_df["ModelName"].replace(rename_dict)


all_df = all_df.sort_values(by="Success_rate", ascending=True)

table_list = [
    "GPT 4o",
    "GPT 4o mini",
    "Gemma2:27B",
    "Llama3.1:8B",
    "Llama3.2:1B",
    "Llama3.2:3B",
    "Llama3.1:70B",
    "Llama3.1:405B",
    "GPT-o3-mini (R)",
    "DeepSeek-R1:8B (R)",
    "DeepSeek-R1:32B (R)",
    "DeepSeek-R1:70B (R)",
    "QwQ:32B (R)",
    "Gemma2:27B (R)",
    "Llama3.1:8B (R)",
    "Llama3.1:405B (R)",
    "Intfloat Multilingual-E5-Small",
    "MiniLM-L6 v2",
    "All-MPNet-Base v2",
    "Average Word Embeddings Glove 6B300d",
    "Roberta-Large v1",
    "TFIDF Self Corpus",
    "Human",
]

table_df = all_df[all_df["Model"].isin(table_list)]

# Game outcome counts for the same set of models

embed_count_df = pd.read_csv(embed_count_path, sep="\t")
prompt_count_df = pd.read_csv(prompt_count_path, sep="\t")
tfidf_count_df = pd.read_csv(tfidf_count_path, sep="\t")

reasoning_count_df = pd.read_csv(reasoning_count_path, sep="\t")
reasoning_count_df["Model"] = reasoning_count_df["Model"] + " (R)"

count_df = pd.concat(
    [prompt_count_df, reasoning_count_df, embed_count_df, tfidf_count_df]
)

count_df = count_df[
    (count_df["Language"] == "English") & (count_df["Topic"] == "Wikipedia")
]

# List of columns representing game outcomes
outcomes = [
    "Success",
    "Step_limit",
    "Dead_ends",
    "Hallucination",
    "Bad_requests",
    "No_response",
    "No_answer",
]

hatch_patterns = ["//", "\\\\", "||", "--", "OO", "..", "xx"]

# Only the reasoning table has a No_answer column.
count_df[outcomes] = count_df[outcomes].fillna(0)

# Convert absolute counts to fractions of Total_games
for outcome in outcomes:
    count_df[outcome] = count_df[outcome] / count_df["Total_games"]

# The human games were not recorded in enough detail to say why they were lost, so
# only the successful fraction of the row is filled in.
human_count_df = pd.DataFrame({"Model": ["Human"]})
human_count_df[outcomes] = 0.0
human_count_df["Success"] = human_df["Success_rate"].to_numpy()

count_df = pd.concat([count_df, human_count_df])

# Matching the order of the two panels so one set of labels serves both.
count_df = count_df.set_index("Model").reindex(table_df["Model"]).reset_index()


# Setting up the figure
fig, axes = plt.subplots(1, 2, figsize=(8, 6.5))

ax1 = axes[0]
ax2 = axes[1]

ax1.text(
    -0.08,
    1.0,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax2.text(
    1.08,
    1.0,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="left",
)

# Axis 1 - Horizontal bar chart for mean steps
y_pos = np.arange(len(table_df))
bars = ax1.barh(
    y_pos,
    table_df["Mean"],
    xerr=table_df["Mean_std"],
    color=[
        hsv_to_rgb((color_hue[cn], 0.50, color_val[cn]))
        for cn in table_df["Colour_name"]
    ],
    edgecolor="black",
    capsize=6,
    error_kw={"elinewidth": 2, "capthick": 2},
)
for bar, hatched in zip(bars, table_df["Hatch"], strict=True):
    if hatched:
        bar.set_hatch("//")

ax1.set_yticks(y_pos)
# The model names sit between the two panels, so they serve both.
ax1.tick_params(axis="y", left=False, right=True, labelleft=False)
ax1.set_xlabel("Mean Steps in Successful Games")
ax1.set_xlim(0, 72)
ax1.set_ylim(-0.6, len(table_df) - 0.4)

ax1.grid(axis="x", linestyle="--", alpha=0.7)
ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
ax1.tick_params(which="both", axis="x", direction="in")


# Adding a legend for model types
legend_labels = [*color_hue, "Closed source"]
# Wrapping the two long labels keeps the legend narrow enough to clear the bars.
legend_labels = [label.replace(" ", "\n") for label in legend_labels]
handles = [
    plt.Line2D(
        [0], [0], color=hsv_to_rgb((color_hue[label], 0.65, color_val[label])), lw=4
    )
    for label in color_hue
]
handles.append(
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//")
)

ax1.legend(handles, legend_labels, loc="upper right", fontsize="small")


# Axis 2 - Stacked bar chart for game outcomes
left_value = None
for n, outcome in enumerate(outcomes):
    ax2.barh(
        y_pos,
        count_df[outcome],
        left=left_value,
        label=outcome.replace("_", " "),
        color=cc.glasbey_category10[n + 2],
        hatch=hatch_patterns[n],
        edgecolor="black",  # Without an opaque edge the PDF sets a stroke alpha=0
        linewidth=0,
    )
    # Update the left value to stack the bars
    left_value = (
        count_df[outcome] if left_value is None else left_value + count_df[outcome]
    )

ax2.set_yticks(y_pos)
ax2.set_yticklabels(table_df["ModelName"])
ax2.set_xlabel("Fraction of Game Outcomes")
ax2.set_xlim(0, 1)
ax2.set_ylim(ax1.get_ylim())

ax2.grid(axis="x", linestyle="--", alpha=0.7)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(which="both", axis="x", direction="in")

ax2.legend(loc="upper left", fontsize="small", framealpha=1.0)


fig.tight_layout()  # Adjust layout to prevent overlap

# Centre the names in the gap the layout left between the two panels.
gutter = (ax2.get_position().x0 - ax1.get_position().x1) * fig.get_figwidth() * 72
ax2.tick_params(axis="y", pad=gutter / 2 - plt.rcParams["ytick.major.size"])
for label in ax2.get_yticklabels():
    label.set_horizontalalignment("center")

fig.savefig(fig_path)
