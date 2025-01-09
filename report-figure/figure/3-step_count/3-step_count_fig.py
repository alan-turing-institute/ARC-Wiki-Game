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

data_output_dir = current_dir.parents[2] / "agents" / "data" / "output_data" / "en_wiki"
gpt_games_path = (
    data_output_dir
    / "step_list_prompt_gpt4o"
    / "en_wiki_step_list_wikigame-gpt-4o_0.csv"
)

e5_games_path_start = (
    data_output_dir / "step_list" / "en_wiki_step_list_intfloat_multilingual-e5-small_"
)

# Load Data

color_names = {
    "Open LLM": 185 / 360,
    "Closed LLM": 112 / 360,
    "Embedding": 28 / 360,
    "TFIDF": 298 / 360,
    "Human": 355 / 360,
}

embed_df = pd.read_csv(embed_path, sep="\t")
embed_df["Colour"] = color_names["Embedding"]

prompt_df = pd.read_csv(prompt_path, sep="\t")
prompt_df["Colour"] = prompt_df["Model"].map(
    lambda m: (color_names["Closed LLM"] if "GPT" in m else color_names["Open LLM"])
)

tfidf_df = pd.read_csv(tfidf_path, sep="\t")
tfidf_df["Colour"] = color_names["TFIDF"]

human_games_df = pd.read_csv(human_results_path, sep=",")

human_steps = human_games_df[human_games_df["Success?"] == "Yes"]["Steps"].values

human_df = pd.DataFrame(
    {
        "Model": ["Human"],
        "Success_rate": [0.688],
        "Success_rate_std": [0.044],
        "Mean": human_steps.mean(),
        "Mean_std": bootstrap(human_steps[None, :], np.mean).standard_error,
        "Colour": [color_names["Human"]],
        "Language": ["English"],
        "Topic": ["Wikipedia"],
    },
    index=[0],
)


all_df = pd.concat([prompt_df, embed_df, tfidf_df, human_df])

all_df = all_df[(all_df["Language"] == "English") & (all_df["Topic"] == "Wikipedia")]


rename_dict = {
    "All-MPNet-Base v2": "All-MPNet-Base",
    "Intfloat Multilingual-E5-Small": "E5-Small",
    "Average Word Embeddings Glove 6B300d": "Glove Word",
    "Roberta-Large v1": "Roberta-Large",
    "GPT 4o": "GPT-4o",
    "GPT 4o mini": "GPT-4o mini",
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

# Setting up the figure
fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

# Axes Labels
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
ax3.text(
    -0.25,
    1.0,
    "C",
    transform=ax3.transAxes,
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
    color=hsv_to_rgb((color_names["Closed LLM"], 0.55, 0.95)),
)

ax1.plot(
    x_p,
    nbinom.pmf(x_p, gpt_n, gpt_p),
    "k--",
)

ax1.set_xlabel("Number of steps above optimal")
ax1.set_ylabel("Game Frequency")
ax1.set_xlim(0, 100)

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
    fontsize=16,
    va="top",
    ha="right",
)


# Axis 2 - Histogram of E5 steps

ax2.hist(
    e5_steps,
    bins=bins,
    density=True,
    color=hsv_to_rgb((color_names["Embedding"], 0.55, 0.95)),
)

ax2.plot(
    x_p,
    nbinom.pmf(x_p, e5_n, e5_p),
    "k--",
)

ax2.set_xlabel(ax1.get_xlabel())
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
    "E5-Small",
    transform=ax2.transAxes,
    fontsize=16,
    va="top",
    ha="right",
)


# Axis 3 - Horizontal bar chart for success rate
y_pos = np.arange(len(table_df))
bars = ax3.barh(
    y_pos,
    table_df["Mean"],
    xerr=table_df["Mean_std"],
    color=[hsv_to_rgb((c, 0.50, 0.95)) for c in table_df["Colour"]],
    edgecolor="black",
    capsize=6,
    error_kw={"elinewidth": 2, "capthick": 2},
)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(table_df["ModelName"])
ax3.set_xlabel("Mean Steps in Successful Games")
ax3.set_xlim(0, 72)
ax3.set_ylim(-0.6, len(table_df) - 0.4)

ax3.grid(axis="x", linestyle="--", alpha=0.7)
ax3.xaxis.set_minor_locator(AutoMinorLocator(10))
ax3.tick_params(which="both", axis="x", direction="in")


# Adding a legend for model types
legend_labels = list(color_names.keys())
handles = [
    plt.Line2D([0], [0], color=hsv_to_rgb((color_names[label], 0.65, 0.95)), lw=4)
    for label in legend_labels
]

ax3.legend(handles, legend_labels, loc="upper right")


fig.tight_layout()  # Adjust layout to prevent overlap

fig.savefig(fig_path)
