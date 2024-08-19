# Sentance Proof of Concept

This folder has a proof of concept for training a model with the Wikipedia page summaries that were obtained from the `WikiMediaParserFromHell`.
It has a notebook which trains a simple fully connected model using a data loader that hands the page summaries and a number indicating the minimum number of clicks between them.

## Running this Code

This code has its own `pyproject.toml` and all the dependencies can be directly pip installed.

```bash
git clone https://github.com/alan-turing-institute/ARC-Wiki-Game
cd ARC-Wiki-Game/sentance-poc
pip install .
```

## Getting the Training Data

The training data is available on the SharePoint.
