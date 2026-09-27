from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator, LogLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig2_performance.pdf"
results_dir = current_dir.parents[2] / "agents" / "result" / "table"

embed_path = results_dir / "embed_model_wiki_summary_stats.csv"
prompt_path = results_dir / "prompt_wiki_summary_stats.csv"
tfidf_path = results_dir / "tfidf_wiki_summary_stats.csv"
reasoning_path = results_dir / "reasoning_wiki_summary_stats.csv"

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

human_df = pd.DataFrame(
    {
        "Model": ["Human"],
        "Success_rate": [0.688],
        "Success_rate_std": [0.044],
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


# Data Preparation
param_dict = {
    "GPT 4o": 1800,
    "GPT 4o mini": 100,
    "Gemma2:27B": 27,
    "Llama3.1:8B": 8,
    "Llama3.2:1B": 1,
    "Llama3.2:3B": 3,
    "Llama3.1:70B": 70,
    "Llama3.1:405B": 405,
    "GPT-o3-mini (R)": 100,
    "DeepSeek-R1:32B (R)": 32,
    "DeepSeek-R1:70B (R)": 70,
    "QwQ:32B (R)": 32,
    "Gemma2:27B (R)": 27,
    "Llama3.1:8B (R)": 8,
    "Llama3.1:405B (R)": 405,
    "Intfloat Multilingual-E5-Small": 0.118,
    "MiniLM-L6 v2": 0.023,
    "Mistral-Nemo": 12,
    "Roberta-Large v1": 0.335,
    "All-MPNet-Base v2": 0.109,
    "Average Word Embeddings Glove 6B300d": 0.12,
}
param_err = {
    "GPT 4o": 800,
    "GPT 4o mini": 90,
    "GPT-o3-mini (R)": 90,
}

all_df["Parameters"] = all_df["Model"].map(param_dict)
all_df["Parameters_std"] = all_df["Model"].map(param_err)


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


scale_df = all_df[all_df["Parameters"] > 0]

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


# Setting up the figure
fig, axes = plt.subplots(1, 2, figsize=(8, 5))

ax1 = axes[0]
ax2 = axes[1]

ax1.text(
    -0.45,
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

# Axis 1 - Horizontal bar chart for success rate
y_pos = np.arange(len(table_df))
bars = ax1.barh(
    y_pos,
    table_df["Success_rate"],
    xerr=table_df["Success_rate_std"],
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
ax1.set_yticklabels(table_df["ModelName"])
ax1.set_xlabel("Success rate")
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.6, len(table_df) - 0.4)

ax1.grid(axis="x", linestyle="--", alpha=0.7)
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(which="both", axis="x", direction="in")


# Adding a legend for model types
legend_labels = [*color_hue, "Closed source"]
handles = [
    plt.Line2D(
        [0], [0], color=hsv_to_rgb((color_hue[label], 0.65, color_val[label])), lw=4
    )
    for label in color_hue
]
handles.append(
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//")
)

ax1.legend(handles, legend_labels, loc="lower right")

# Axis 2 - model parameters vs success rate
for col_n, col_h in color_hue.items():
    col_scale_df = scale_df[scale_df["Colour_name"] == col_n]
    ax2.errorbar(
        col_scale_df["Parameters"] * 1e9,
        col_scale_df["Success_rate"],
        xerr=col_scale_df["Parameters_std"] * 1e9,
        yerr=col_scale_df["Success_rate_std"],
        color=hsv_to_rgb((col_h, 0.95, color_val[col_n])),
        capsize=4,
        elinewidth=2.5,
        capthick=2.5,
        linestyle="None",
    )
ax2.set_xscale("log")
ax2.set_xlabel("Parameters (log scale)")
ax2.set_ylabel("Success rate")
ax2.set_xlim(1.5e7, 3e12)
ax2.set_ylim(0, 0.88)

ax2.plot(
    ax2.get_xlim(),
    (human_df["Success_rate"].iloc[0] - human_df["Success_rate_std"].iloc[0])
    * np.array([1, 1]),
    "--",
    color=hsv_to_rgb((color_hue["Human"], 0.95, color_val["Human"])),
)
ax2.plot(
    ax2.get_xlim(),
    (human_df["Success_rate"].iloc[0] + human_df["Success_rate_std"].iloc[0])
    * np.array([1, 1]),
    "--",
    color=hsv_to_rgb((color_hue["Human"], 0.95, color_val["Human"])),
)

ax2.grid(axis="both", alpha=0.7)
ax2.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=5))
ax2.tick_params(which="both", direction="in")


fig.tight_layout()  # Adjust layout to prevent overlap

fig.savefig(fig_path)
