from __future__ import annotations

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path_1 = current_dir / "afig_failure_by_topic_llm.pdf"
fig_path_2 = current_dir / "afig_failure_by_topic_embed.pdf"
results_dir = current_dir.parents[2] / "agents" / "result" / "table"

embed_path = results_dir / "embed_model_wiki_summary_game_count.csv"
prompt_path = results_dir / "prompt_wiki_summary_game_count.csv"

# Load Data

embed_df = pd.read_csv(embed_path, sep="\t")
prompt_df = pd.read_csv(prompt_path, sep="\t")

all_df = pd.concat([prompt_df, embed_df])

# Models short names

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

# Filter data for model comparison

# List of columns representing game outcomes
outcomes = [
    "Success",
    "Step_limit",
    "Dead_ends",
    "Hallucination",
    "Bad_requests",
    "No_response",
]

# Convert absolute counts to fractions of Total_games
for outcome in outcomes:
    all_df[outcome] = all_df[outcome] / all_df["Total_games"]

# Set of Topics

all_df["Topic"] = all_df["Topic"].str.replace("OpenAlex ", "")

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

all_df["Topic_name"] = (
    all_df[["Language", "Topic"]].apply(tuple, axis=1).map(topic_names)
)
all_df["Order"] = all_df[["Language", "Topic"]].apply(tuple, axis=1).map(topic_order)

all_df = all_df[all_df["Order"].notnull()]
all_df = all_df.sort_values(by="Order")

# Make model dataframes

gpt4o_df = all_df[all_df["ModelName"] == "GPT-4o"]
llama405_df = all_df[all_df["ModelName"] == "Llama3.1:405B"]
gemma27_df = all_df[all_df["ModelName"] == "Gemma2:27B"]
llama3b_df = all_df[all_df["ModelName"] == "Llama3.2:3B"]

e5small_df = all_df[all_df["ModelName"] == "E5-Small"]
mpnet_df = all_df[all_df["ModelName"] == "All-MPNet-Base"]
glove_df = all_df[all_df["ModelName"] == "Glove Word"]

# Set up plotting function


def plot_topic_bar(axis, data_df, title):
    # Plot each outcome as a stacked bar
    bottom_value = None
    for n, outcome in enumerate(outcomes):
        axis.bar(
            data_df["Topic_name"],
            data_df[outcome],
            bottom=bottom_value,
            label=outcome.replace("_", " "),
            color=cc.glasbey_category10[n + 2],
        )
        # Update the bottom value to stack the bars
        bottom_value = (
            data_df[outcome]
            if bottom_value is None
            else bottom_value + data_df[outcome]
        )

    # Rotate x-axis tick labels
    axis.set_xticklabels(axis.get_xticklabels(), rotation=60, ha="right")

    # Add title
    axis.set_title(title)

    # Add labels
    axis.grid(axis="y", linestyle="--", alpha=0.7)
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))


# Setting up the figure 1
fig1, axes = plt.subplots(1, 4, figsize=(8, 5), width_ratios=[0.5, 1, 1, 1])

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
ax4 = axes[3]

plot_topic_bar(ax1, gpt4o_df, "GPT-4o")
plot_topic_bar(ax2, llama405_df, "Llama3.1:405B")
plot_topic_bar(ax3, gemma27_df, "Gemma2:27B")
plot_topic_bar(ax4, llama3b_df, "Llama3.2:3B")

ax1.set_ylabel("Fraction of Game Outcomes")
ax4.set_ylabel("Fraction of Game Outcomes")
ax4.yaxis.set_label_position("right")
ax1.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=False,
    labelleft=True,
)
ax2.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=False,
    labelleft=False,
)
ax3.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=False,
    labelleft=False,
)
ax4.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=True,
    labelleft=False,
)

fig1.legend(
    handles=ax1.get_legend_handles_labels()[0],
    labels=ax1.get_legend_handles_labels()[1],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
)

fig1.tight_layout()  # Adjust layout to prevent overlap

fig1.subplots_adjust(wspace=0.025, bottom=0.3)  # Reduce the space between subplots

fig1.savefig(fig_path_1)

# Setting up the figure 2
fig2, axes = plt.subplots(1, 3, figsize=(8, 5), width_ratios=[1, 1, 0.7])

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]

plot_topic_bar(ax1, e5small_df, "E5-Small")
plot_topic_bar(ax2, mpnet_df, "All-MPNet-Base")
plot_topic_bar(ax3, glove_df, "Glove Word")

ax1.set_ylabel("Fraction of Game Outcomes")
ax3.set_ylabel("Fraction of Game Outcomes")
ax3.yaxis.set_label_position("right")
ax1.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=False,
    labelleft=True,
)
ax2.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=False,
    labelleft=False,
)
ax3.tick_params(
    which="both",
    axis="y",
    direction="in",
    left=True,
    right=True,
    labelright=True,
    labelleft=False,
)

fig2.legend(
    handles=ax1.get_legend_handles_labels()[0],
    labels=ax1.get_legend_handles_labels()[1],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
)

fig2.tight_layout()  # Adjust layout to prevent overlap

fig2.subplots_adjust(wspace=0.025, bottom=0.3)  # Reduce the space between subplots

fig2.savefig(fig_path_2)
