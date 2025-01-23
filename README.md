# ARC-Wiki-Game

In the Wikipedia Game participants start with a random Wikipedia article and a random target article and are expected to navigate between them by only clicking on links within each article.
This is often framed as a competition between players, where the winning player is who reaches the target article in the fewest number of clicks, or the fastest time.

The Wikipedia Game concept is not limited to English Wikipedia, and similar games can be played on other datasets.

This repo has the functionality to create graph datasets, create agents to play the Wikipedia Game and analyse the output of the games. The repo contains three packages, which are described below.

## Packages

### `agents`

This package provides code to create agents that can play the Wikipedia Game on a graph dataset. The `graph-prep` package can create graph datasets in the required format. Further details can be found in the [agents package README](agents/README.md)

### `graph-prep`

This package does the pre-processing on the Wikipedia and OpenAlex datasets in order to produce the navigable graphs. It also provides some tools for analysing the graphs produced.  Further details can be found in the [graph-prep package README](graph-prep/README.md)

### `report-figure`

This package only contains scripts to generate the figures seen in the paper.

## Getting Started

Clone this repository and change the current working directory.

```bash
git clone https://github.com/alan-turing-institute/ARC-Wiki-Game.git
cd ARC-Wiki-Game
```

Change the directory to the desired package and follow further instructions in the README file of the package.
