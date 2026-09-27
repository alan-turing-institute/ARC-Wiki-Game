from __future__ import annotations

from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# Set up paths
current_dir = Path(__file__).resolve().parent
fig_path = current_dir / "afig_failure_reasons_enwiki.pdf"
results_dir = current_dir.parents[2] / "agents" / "result" / "table"

embed_path = results_dir / "embed_model_wiki_summary_game_count.csv"
prompt_path = results_dir / "prompt_wiki_summary_game_count.csv"
tfidf_path = results_dir / "tfidf_wiki_summary_game_count.csv"
reasoning_path = results_dir / "reasoning_wiki_summary_game_count.csv"


# Load Data

embed_df = pd.read_csv(embed_path, sep="\t")
prompt_df = pd.read_csv(prompt_path, sep="\t")
tfidf_df = pd.read_csv(tfidf_path, sep="\t")

reasoning_df = pd.read_csv(reasoning_path, sep="\t")
# Distinguishes the models run through both agents, e.g. Llama3.1:405B.
reasoning_df["Model"] = reasoning_df["Model"] + " (R)"

all_df = pd.concat([prompt_df, reasoning_df, embed_df, tfidf_df])

# Models short names

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

# Filter data for model comparison

model_compare_list = [
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

enwiki_df = all_df[(all_df["Language"] == "English") & (all_df["Topic"] == "Wikipedia")]
enwiki_df = enwiki_df[enwiki_df["Model"].isin(model_compare_list)]

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
enwiki_df[outcomes] = enwiki_df[outcomes].fillna(0)

# Convert absolute counts to fractions of Total_games
for outcome in outcomes:
    enwiki_df[outcome] = enwiki_df[outcome] / enwiki_df["Total_games"]

enwiki_df = enwiki_df.sort_values("Success", ascending=True)


# Setting up the figure
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))


# Plot each outcome as a stacked bar
bottom_value = None
for n, outcome in enumerate(outcomes):
    ax1.barh(
        enwiki_df["ModelName"],
        enwiki_df[outcome],
        left=bottom_value,
        label=outcome.replace("_", " "),
        color=cc.glasbey_category10[n + 2],
        hatch=hatch_patterns[n],
    )
    # Update the bottom value to stack the bars
    bottom_value = (
        enwiki_df[outcome]
        if bottom_value is None
        else bottom_value + enwiki_df[outcome]
    )

# Add labels and legend
ax1.set_ylim(-0.8, len(enwiki_df) - 0.2)
ax1.set_xlabel("Fraction of Game Outcomes")
ax1.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

ax1.grid(axis="x", linestyle="--", alpha=0.7)
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(which="both", axis="x", direction="in")


fig.tight_layout()  # Adjust layout to prevent overlap

fig.savefig(fig_path)
