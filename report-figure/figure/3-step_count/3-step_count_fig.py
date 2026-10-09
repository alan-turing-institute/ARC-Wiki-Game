from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from scipy.stats import bootstrap, nbinom

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig3_step_count.pdf"
human_results_path = current_dir / "human_results.csv"

results_dir = current_dir.parents[2] / "agents" / "result" / "table"
embed_path = results_dir / "embed_model_wiki_summary_stats.csv"
prompt_path = results_dir / "prompt_wiki_summary_stats.csv"
tfidf_path = results_dir / "tfidf_wiki_summary_stats.csv"
reasoning_path = results_dir / "reasoning_wiki_summary_stats.csv"

data_output_dir = current_dir.parents[2] / "agents" / "data" / "output_data" / "en_wiki"
gpt_games_path = (
    data_output_dir
    / "step_list_prompt_gpt4o"
    / "en_wiki_step_list_wikigame-gpt-4o_0.csv"
)

e5_games_path_start = (
    data_output_dir / "step_list" / "en_wiki_step_list_intfloat_multilingual-e5-small_"
)

r1_games_path_start = (
    data_output_dir / "reasoning_medium" / "en_wiki_step_list_deepseek-r1:70b_"
)

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

human_games_df = pd.read_csv(human_results_path, sep=",")

human_steps = human_games_df[human_games_df["Success?"] == "Yes"]["Steps"].values

human_df = pd.DataFrame(
    {
        "Model": ["Human"],
        "Success_rate": [0.688],
        "Success_rate_std": [0.044],
        "Mean": human_steps.mean(),
        "Mean_std": bootstrap(human_steps[None, :], np.mean).standard_error,
        "Colour_name": ["Human"],
        "Language": ["English"],
        "Topic": ["Wikipedia"],
    },
    index=[0],
)


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

# Get histogram data

x_p = np.arange(0, 101, 1)
bins = np.arange(0, 101, 1) + 0.5

gpt_games_df = pd.read_csv(gpt_games_path, sep="\t")

gpt_steps = gpt_games_df["num_steps"].values - gpt_games_df["optimal_steps"].values
gpt_steps = gpt_steps - (~gpt_games_df["found_target"]).values * 1000

gpt_n = all_df.loc[all_df["Model"] == "GPT 4o", "n"].values[0]
gpt_p = all_df.loc[all_df["Model"] == "GPT 4o", "p"].values[0]

e5_list = pd.concat(
    [
        pd.read_csv(str(e5_games_path_start) + str(i) + ".csv", sep="\t")
        for i in range(0, 9001, 1000)
    ]
)

e5_steps = e5_list["num_steps"].values - e5_list["optimal_steps"].values
e5_steps = e5_steps - (~e5_list["found_target"]).values * 1000

e5_n = all_df.loc[all_df["Model"] == "Intfloat Multilingual-E5-Small", "n"].values[0]
e5_p = all_df.loc[all_df["Model"] == "Intfloat Multilingual-E5-Small", "p"].values[0]

r1_list = pd.concat(
    [
        pd.read_csv(str(r1_games_path_start) + str(i) + ".csv", sep="\t")
        for i in range(0, 200, 10)
    ]
)

r1_steps = r1_list["num_steps"].values - r1_list["optimal_steps"].values
r1_steps = r1_steps - (~r1_list["found_target"]).values * 1000

r1_n = all_df.loc[all_df["Model"] == "DeepSeek-R1:70B (R)", "n"].values[0]
r1_p = all_df.loc[all_df["Model"] == "DeepSeek-R1:70B (R)", "p"].values[0]

# Setting up the figure
fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:, 1])

# Axes Labels
ax1.text(
    -0.07,
    1.08,
    "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax2.text(
    -0.07,
    1.08,
    "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax3.text(
    -0.07,
    1.08,
    "C",
    transform=ax3.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)
ax4.text(
    -0.35,
    0.94,
    "D",
    transform=ax4.transAxes,
    fontsize=20,
    fontweight="bold",
    va="top",
    ha="right",
)

# Axis 1 - Histogram of GPT steps

ax1.hist(
    gpt_steps,
    bins=bins,
    density=True,
    color=hsv_to_rgb((color_hue["LLM"], 0.55, color_val["LLM"])),
)

ax1.plot(
    x_p,
    nbinom.pmf(x_p, gpt_n, gpt_p),
    "k--",
)

ax1.set_ylabel("Game Frequency")
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 0.095)

ax1.grid(alpha=0.7)
ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax1.tick_params(which="both", axis="both", direction="in")
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(AutoMinorLocator(4))

ax1.text(
    0.95,
    0.95,
    "GPT-4o",
    transform=ax1.transAxes,
    fontsize=12,
    va="top",
    ha="right",
)


# Axis 2 - Histogram of DeepSeek-R1:70B steps

ax2.hist(
    r1_steps,
    bins=bins,
    density=True,
    color=hsv_to_rgb((color_hue["Reasoning LLM"], 0.55, color_val["Reasoning LLM"])),
)

ax2.plot(
    x_p,
    nbinom.pmf(x_p, r1_n, r1_p),
    "k--",
)

ax2.set_ylabel(ax1.get_ylabel())
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())

ax2.grid(alpha=0.7)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax2.tick_params(which="both", axis="both", direction="in")
ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
ax2.yaxis.set_minor_locator(AutoMinorLocator(4))

ax2.text(
    0.95,
    0.95,
    "DeepSeek(DS)-R1:70B (R)",
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="right",
)


# Axis 3 - Histogram of E5 steps

ax3.hist(
    e5_steps,
    bins=bins,
    density=True,
    color=hsv_to_rgb((color_hue["Embedding"], 0.55, color_val["Embedding"])),
)

ax3.plot(
    x_p,
    nbinom.pmf(x_p, e5_n, e5_p),
    "k--",
)

# Only the bottom panel carries the shared x label.
ax3.set_xlabel("Number of steps above optimal")
ax3.set_ylabel(ax1.get_ylabel())
ax3.set_xlim(ax1.get_xlim())
ax3.set_ylim(ax1.get_ylim())

ax3.grid(alpha=0.7)
ax3.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax3.tick_params(which="both", axis="both", direction="in")
ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
ax3.yaxis.set_minor_locator(AutoMinorLocator(4))

ax3.text(
    0.95,
    0.95,
    "E5-Small",
    transform=ax3.transAxes,
    fontsize=12,
    va="top",
    ha="right",
)


# Axis 4 - Horizontal bar chart for success rate
y_pos = np.arange(len(table_df))
bars = ax4.barh(
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

ax4.set_yticks(y_pos)
ax4.set_yticklabels(table_df["ModelName"])
ax4.set_xlabel("Mean Steps in Successful Games")
ax4.set_xlim(0, 72)
ax4.set_ylim(-0.6, len(table_df) - 0.4)

ax4.grid(axis="x", linestyle="--", alpha=0.7)
ax4.xaxis.set_minor_locator(AutoMinorLocator(10))
ax4.tick_params(which="both", axis="x", direction="in")


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

ax4.legend(handles, legend_labels, loc="upper right")


fig.tight_layout()  # Adjust layout to prevent overlap
# Applied after tight_layout, which otherwise recomputes the gap between the columns.
# Setting wspace on the gridspec instead would disable tight_layout entirely.
fig.subplots_adjust(wspace=0.55, hspace=0.15)

fig.savefig(fig_path)
