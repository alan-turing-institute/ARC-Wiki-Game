from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig5_topics.pdf"

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

rename_dict = {
    "All-MPNet-Base v2": "All-MPNet-Base",
    "Intfloat Multilingual-E5-Small": "E5-Small",
    "GPT 4o": "GPT-4o",
    "GPT 4o mini": "GPT-4o mini",
    "DeepSeek-R1:70B (R)": "DS-R1:70B (R)",
    "TFIDF Self Corpus": "TFIDF",
}

all_df["ModelName"] = all_df["Model"]
all_df["ModelName"] = all_df["ModelName"].replace(rename_dict)


all_df = all_df.sort_values(by="Success_rate", ascending=True)

show_list = [
    "Human",
    "TFIDF Self Corpus",
    "GPT 4o",
    "GPT 4o mini",
    "Llama3.1:405B",
    "Gemma2:27B",
    "DeepSeek-R1:70B (R)",
    "Intfloat Multilingual-E5-Small",
    "All-MPNet-Base v2",
]

all_df["Topic"] = all_df["Topic"].str.replace("OpenAlex ", "")

all_df = all_df[all_df["Model"].isin(show_list)]

topic_names = {
    ("English", "Wikipedia"): "English Wiki",
    ("French", "Wikipedia"): "French Wiki",
    ("Russian", "Wikipedia"): "Russian Wiki",
    ("Chinese", "Wikipedia"): "Chinese Wiki",
    ("Bengali", "Wikipedia"): "Bengali Wiki",
    ("English", "Physics"): "Physics OA",
    ("English", "Biochemistry"): "Biology OA",
    ("English", "Medicine"): "Medicine OA",
    ("English", "Social Sciences"): "Social Sci OA",
    ("English", "Arts and Humanities"): "Arts OA",
}
topic_order = {topic: i for i, topic in enumerate(topic_names.keys())}

all_df["Order"] = all_df[["Language", "Topic"]].apply(tuple, axis=1).map(topic_order)

all_df = all_df[all_df["Order"].notnull()]
all_df = all_df.sort_values(by="Order")


# Setting up the Figure
fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
ax1, ax2 = axes

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

marker_choice = dict(
    zip(show_list, ["o", "+", "s", "D", "<", "^", ">", "P", "X"], strict=False)
)
line_choice = dict(
    zip(show_list, ["", "--", "--", ":", "--", ":", "-.", "--", ":"], strict=False)
)

# Models sharing a colour class are stepped apart in saturation and brightness.
class_models: dict[str, list[str]] = {}
for model in show_list:
    col_n = all_df.loc[all_df["Model"] == model, "Colour_name"].values[0]
    class_models.setdefault(col_n, []).append(model)

color_choice = {
    model: hsv_to_rgb((color_hue[col_n], 0.95 - 0.11 * i, color_val[col_n] - 0.16 * i))
    for col_n, models in class_models.items()
    for i, model in enumerate(models)
}


# Plotting the Figure


def plot_axis(axis, to_show_list):
    legend_handles = []
    legend_labels = []

    for model in to_show_list:
        model_df = all_df[all_df["Model"] == model]
        axis.errorbar(
            model_df["Order"].values,
            model_df["Success_rate"].values,
            yerr=model_df["Success_rate_std"].values,
            marker=marker_choice[model],
            markersize=8,
            capsize=8,
            linestyle=line_choice[model],
            linewidth=2,
            color=color_choice[model],
            label=model_df["ModelName"].values[0],
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker_choice[model],
                color=color_choice[model],
                linestyle=line_choice[model],
                linewidth=2,
            )
        )
        legend_labels.append(model_df["ModelName"].values[0])

    axis.legend(
        labels=legend_labels,
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        columnspacing=1.0,
        framealpha=1.0,
        fontsize="small",
    )

    # Formatting the Figure

    axis.grid(axis="y", linestyle="-", alpha=0.7)
    axis.set_ylim(0, 0.88)

    axis.tick_params(axis="y", which="both", direction="in")
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))


plot_axis(ax1, [show_list[x] for x in [0, 7, 8, 1]])
plot_axis(ax2, [show_list[x] for x in [0, 2, 6, 3, 4, 5]])

ax1.set_ylabel("Success Rate")
ax2.set_ylabel("Success Rate")

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

ax1.set_xticks(np.arange(len(topic_order)))
ax1.set_xticklabels(topic_names.values(), rotation=60, ha="right")
ax2.set_xticks(np.arange(len(topic_order) - 2))
ax2.set_xticklabels(list(topic_names.values())[:-2], rotation=60, ha="right")


fig.tight_layout()  # Adjust layout to prevent overlap

fig.subplots_adjust(wspace=0.025)  # Reduce the space between subplots

fig.savefig(fig_path)
