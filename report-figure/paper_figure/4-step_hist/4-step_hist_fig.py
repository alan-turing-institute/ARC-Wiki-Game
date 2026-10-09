from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from scipy.stats import nbinom

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig4_step_hist.pdf"

results_dir = current_dir.parents[2] / "agents" / "result" / "table"
embed_path = results_dir / "embed_model_wiki_summary_stats.csv"
prompt_path = results_dir / "prompt_wiki_summary_stats.csv"
reasoning_path = results_dir / "reasoning_wiki_summary_stats.csv"
human_path = results_dir / "human_wiki_summary_stats.csv"

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

human_dir = current_dir.parents[2] / "agents" / "data" / "human"
human_games_path = human_dir / "en_wiki_step_list_human_0.csv"

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
prompt_df = pd.read_csv(prompt_path, sep="\t")

reasoning_df = pd.read_csv(reasoning_path, sep="\t")
# Distinguishes the models run through both agents, e.g. Llama3.1:405B.
reasoning_df["Model"] = reasoning_df["Model"] + " (R)"

human_df = pd.read_csv(human_path, sep="\t")

all_df = pd.concat([prompt_df, reasoning_df, embed_df, human_df])

all_df = all_df[(all_df["Language"] == "English") & (all_df["Topic"] == "Wikipedia")]

# Get histogram data

x_p = np.arange(0, 101, 1)
bins = np.arange(0, 101, 1) + 0.5


def get_steps(games_df):
    """Steps above optimal, with lost games pushed off the left of the axis."""
    steps = games_df["num_steps"].values - games_df["optimal_steps"].values
    return steps - (~games_df["found_target"]).values * 1000


gpt_games_df = pd.read_csv(gpt_games_path, sep="\t")
gpt_steps = get_steps(gpt_games_df)

gpt_n = all_df.loc[all_df["Model"] == "GPT 4o", "n"].values[0]
gpt_p = all_df.loc[all_df["Model"] == "GPT 4o", "p"].values[0]

e5_list = pd.concat(
    [
        pd.read_csv(str(e5_games_path_start) + str(i) + ".csv", sep="\t")
        for i in range(0, 9001, 1000)
    ]
)

e5_steps = get_steps(e5_list)

e5_n = all_df.loc[all_df["Model"] == "Intfloat Multilingual-E5-Small", "n"].values[0]
e5_p = all_df.loc[all_df["Model"] == "Intfloat Multilingual-E5-Small", "p"].values[0]

r1_list = pd.concat(
    [
        pd.read_csv(str(r1_games_path_start) + str(i) + ".csv", sep="\t")
        for i in range(0, 200, 10)
    ]
)

r1_steps = get_steps(r1_list)

r1_n = all_df.loc[all_df["Model"] == "DeepSeek-R1:70B (R)", "n"].values[0]
r1_p = all_df.loc[all_df["Model"] == "DeepSeek-R1:70B (R)", "p"].values[0]

human_games_df = pd.read_csv(human_games_path, sep="\t")
human_steps = get_steps(human_games_df)

human_n = all_df.loc[all_df["Model"] == "Human", "n"].values[0]
human_p = all_df.loc[all_df["Model"] == "Human", "p"].values[0]

# Setting up the figure
fig, axes = plt.subplots(2, 2, figsize=(8, 4.2))

ax1, ax2, ax3, ax4 = axes.flatten()

# Set up plotting function


def plot_step_hist(axis, steps, fit_n, fit_p, colour_name, label, panel, y_max=0.145):
    axis.hist(
        steps,
        bins=bins,
        density=True,
        color=hsv_to_rgb((color_hue[colour_name], 0.55, color_val[colour_name])),
    )

    axis.plot(
        x_p,
        nbinom.pmf(x_p, fit_n, fit_p),
        "k--",
    )

    axis.set_xlabel("Number of steps above optimal")
    axis.set_ylabel("Game Frequency")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, y_max)

    axis.grid(alpha=0.7)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.tick_params(which="both", axis="both", direction="in")
    axis.xaxis.set_minor_locator(AutoMinorLocator(4))
    axis.yaxis.set_minor_locator(AutoMinorLocator(4))

    axis.text(
        0.95,
        0.95,
        label,
        transform=axis.transAxes,
        fontsize=13.5,
        va="top",
        ha="right",
    )

    axis.text(
        -0.1,
        1.12,
        panel,
        transform=axis.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="right",
    )


plot_step_hist(ax1, gpt_steps, gpt_n, gpt_p, "LLM", "GPT-4o", "A")
plot_step_hist(
    ax2, r1_steps, r1_n, r1_p, "Reasoning LLM", "DeepSeek(DS)-R1:70B (R)", "B"
)
plot_step_hist(ax3, e5_steps, e5_n, e5_p, "Embedding", "E5-Small", "C")
plot_step_hist(ax4, human_steps, human_n, human_p, "Human", "Human", "D", y_max=0.24)

fig.tight_layout()  # Adjust layout to prevent overlap

fig.savefig(fig_path)
