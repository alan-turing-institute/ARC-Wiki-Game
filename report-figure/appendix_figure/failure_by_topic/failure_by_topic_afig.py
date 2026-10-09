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
reasoning_path = results_dir / "reasoning_wiki_summary_game_count.csv"

# Load Data

embed_df = pd.read_csv(embed_path, sep="\t")
prompt_df = pd.read_csv(prompt_path, sep="\t")

reasoning_df = pd.read_csv(reasoning_path, sep="\t")
# Distinguishes the models run through both agents, e.g. Llama3.1:405B.
reasoning_df["Model"] = reasoning_df["Model"] + " (R)"

all_df = pd.concat([prompt_df, reasoning_df, embed_df])

# Models short names

rename_dict = {
    "All-MPNet-Base v2": "All-MPNet-Base",
    "Intfloat Multilingual-E5-Small": "E5-Small",
    "Average Word Embeddings Glove 6B300d": "Glove Word",
    "Roberta-Large v1": "Roberta-Large",
    "GPT 4o": "GPT-4o",
    "GPT 4o mini": "GPT-4o mini",
    "DeepSeek-R1:70B (R)": "DS-R1:70B (R)",
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
    "No_answer",
]

hatch_patterns = ["//", "\\\\", "||", "--", "OO", "..", "xx"]

# Only the reasoning table has a No_answer column.
all_df[outcomes] = all_df[outcomes].fillna(0)

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
dsr170_df = all_df[all_df["ModelName"] == "DS-R1:70B (R)"]
llama405_df = all_df[all_df["ModelName"] == "Llama3.1:405B"]
gemma27_df = all_df[all_df["ModelName"] == "Gemma2:27B"]
llama3b_df = all_df[all_df["ModelName"] == "Llama3.2:3B"]

e5small_df = all_df[all_df["ModelName"] == "E5-Small"]
mpnet_df = all_df[all_df["ModelName"] == "All-MPNet-Base"]
glove_df = all_df[all_df["ModelName"] == "Glove Word"]

# Set up plotting function


def plot_topic_bar(axis, data_df, title):
    # Plot each outcome as a stacked bar
    left_value = None
    for n, outcome in enumerate(outcomes):
        axis.barh(
            data_df["Topic_name"],
            data_df[outcome],
            left=left_value,
            label=outcome.replace("_", " "),
            color=cc.glasbey_category10[n + 2],
            hatch=hatch_patterns[n],
            edgecolor="black",  # Without an opaque edge the PDF sets a stroke alpha=0
            linewidth=0,
        )
        # Update the left value to stack the bars
        left_value = (
            data_df[outcome] if left_value is None else left_value + data_df[outcome]
        )

    # Add title
    axis.set_title(title)

    # Topics read down the panel in the same order as the other figures.
    axis.set_ylim(len(data_df) - 0.5, -0.5)

    # Add labels
    axis.grid(axis="x", linestyle="--", alpha=0.7)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.tick_params(which="both", axis="x", direction="in", top=True)


# Setting up the figure 1
fig1, axes = plt.subplots(5, 1, figsize=(8, 11.5), height_ratios=[3, 8, 8, 8, 8])

ax1, ax2, ax3, ax4, ax5 = axes

plot_topic_bar(ax1, gpt4o_df, "GPT-4o")
plot_topic_bar(ax2, dsr170_df, "DS-R1:70B (R)")
plot_topic_bar(ax3, llama405_df, "Llama3.1:405B")
plot_topic_bar(ax4, gemma27_df, "Gemma2:27B")
plot_topic_bar(ax5, llama3b_df, "Llama3.2:3B")

ax5.set_xlabel("Fraction of Game Outcomes")

fig1.legend(
    handles=ax1.get_legend_handles_labels()[0],
    labels=ax1.get_legend_handles_labels()[1],
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
)

fig1.tight_layout()  # Adjust layout to prevent overlap

fig1.subplots_adjust(right=0.78)  # Leave room for the legend

fig1.savefig(fig_path_1)

# Setting up the figure 2
fig2, axes = plt.subplots(3, 1, figsize=(8, 8.2), height_ratios=[10, 10, 6])

ax1, ax2, ax3 = axes

plot_topic_bar(ax1, e5small_df, "E5-Small")
plot_topic_bar(ax2, mpnet_df, "All-MPNet-Base")
plot_topic_bar(ax3, glove_df, "Glove Word")

ax3.set_xlabel("Fraction of Game Outcomes")

fig2.legend(
    handles=ax1.get_legend_handles_labels()[0],
    labels=ax1.get_legend_handles_labels()[1],
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
)

fig2.tight_layout()  # Adjust layout to prevent overlap

fig2.subplots_adjust(right=0.78)  # Leave room for the legend

fig2.savefig(fig_path_2)
