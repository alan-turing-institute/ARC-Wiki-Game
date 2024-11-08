from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import AutoMinorLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "fig4_topics.pdf"

results_dir = current_dir.parents[2] / "agents" / "result" / "table"
embed_path = results_dir / "embed_model_wiki_summary_stats.csv"
prompt_path = results_dir / "prompt_wiki_summary_stats.csv"
tfidf_path = results_dir / "tfidf_wiki_summary_stats.csv"


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

human_df = pd.DataFrame(
    {
        "Model": ["Human"],
        "Success_rate": [0.688],
        "Success_rate_std": [0.044],
        "Colour": [color_names["Human"]],
        "Language": ["English"],
        "Topic": ["Wikipedia"],
    },
    index=[0],
)


all_df = pd.concat([prompt_df, embed_df, tfidf_df, human_df])

rename_dict = {
    "All-MPNet-Base v2": "All-MPNet-Base",
    "Intfloat Multilingual-E5-Small": "E5-Small",
    "GPT 4o": "GPT-4o",
    "GPT 4o mini": "GPT-4o mini",
}

all_df["ModelName"] = all_df["Model"]
all_df["ModelName"] = all_df["ModelName"].replace(rename_dict)


all_df = all_df.sort_values(by="Success_rate", ascending=True)

show_list = [
    "Human",
    "GPT 4o",
    "GPT 4o mini",
    "Llama3.1:405B",
    "Gemma2:27B",
    "Intfloat Multilingual-E5-Small",
    "All-MPNet-Base v2",
]

all_df = all_df[all_df["Model"].isin(show_list)]

topic_names = {
    ("English", "Wikipedia"): "English Wiki",
    ("French", "Wikipedia"): "French Wiki",
    ("Russian", "Wikipedia"): "Russian Wiki",
    ("Chinese", "Wikipedia"): "Chinese Wiki",
    ("Bengali", "Wikipedia"): "Bengali Wiki",
    ("English", "Physics"): "Physics OA",
    ("English", "Biochemistry"): "Biochemistry OA",
    ("English", "Medicine"): "Medicine OA",
    ("English", "Social Sciences"): "Social Sci OA",
    ("English", "Arts and Humanities"): "Arts OA",
}
topic_order = {topic: i for i, topic in enumerate(topic_names.keys())}

all_df["Order"] = all_df[["Language", "Topic"]].apply(tuple, axis=1).map(topic_order)

all_df = all_df[all_df["Order"].notnull()]
all_df = all_df.sort_values(by="Order")


# Setting up the Figure
fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

marker_choice = dict(zip(show_list, ["o", "s", "D", "<", "^", ">", "v"], strict=False))
line_choice = dict(zip(show_list, ["", "--", ":", "--", ":", "--", ":"], strict=False))


# Plotting the Figure

legend_handles = []
legend_labels = []

for model in show_list:
    model_df = all_df[all_df["Model"] == model]
    ax1.errorbar(
        model_df["Order"].values,
        model_df["Success_rate"].values,
        yerr=model_df["Success_rate_std"].values,
        marker=marker_choice[model],
        markersize=8,
        capsize=8,
        linestyle=line_choice[model],
        color=hsv_to_rgb((model_df["Colour"].values[0], 0.95, 0.85)),
        label=model_df["ModelName"].values[0],
    )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker=marker_choice[model],
            color=hsv_to_rgb((model_df["Colour"].values[0], 0.95, 0.85)),
            linestyle=line_choice[model],
        )
    )
    legend_labels.append(model_df["ModelName"].values[0])

legend_handles.insert(1, plt.Line2D([0], [0], marker="", linestyle=""))
legend_labels.insert(1, " ")

ax1.legend(
    labels=legend_labels,
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    ncol=4,
    framealpha=1.0,
)

# Formatting the Figure

ax1.set_xticks(np.arange(len(topic_order)))
ax1.set_xticklabels(topic_names.values(), rotation=45, ha="right")

ax1.set_ylabel("Success Rate")

ax1.grid(axis="y", linestyle="-", alpha=0.7)
ax1.set_ylim(0, 0.85)

ax1.tick_params(axis="y", which="both", direction="in")
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))


fig.tight_layout()  # Adjust layout to prevent overlap

fig.savefig(fig_path)

print(all_df["Order"].values)
