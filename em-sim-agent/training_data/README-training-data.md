# Embedding Training Data

This folder contains the training data for the embedding similarity model.
There are three files which are all extracted from the 2024-07-20 English Wikipedia dump.

* `matidx_sum_table.csv` - This has the metadata for all the pages.
* `summary.csv` - This has the text summaries for all the pages.
* `wiki-forward.hdf5` - This has the graph representation for Wikipedia in the form of a lookup for each page with what pages it links to.
