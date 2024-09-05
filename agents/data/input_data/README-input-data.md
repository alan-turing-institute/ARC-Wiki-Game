# Input Data

This folder contains the input data for the agents models.

The data exist in subfolders - one for each dataset.

Currently all the input data are currated from Wikipedia data dumps; however two different methods have been used to create the datasets which will result in differences in their content. The first method results in a dataset where all the wikipedia links on a page are recorded; this includes links such as categories that exist outside the main content of a page. The second method result in a dataset where only the links in the main content of the page are recorded.

The subsections below describe each of the datasets that have been used in this repo.

## EN Wikipedia - all links
These data need to be located in a subfolder called `en_wikipedia_all_links`. This dataset stores all the links on a page, not just the links in the main content. There are three files which are all extracted from the 2024-07-20 English Wikipedia dump.

* `matidx_sum_table.csv` - This has the metadata for all the pages.
* `summary.csv` - This has the text summaries for all the pages.
* `wiki-forward.hdf5` - This has the graph representation for Wikipedia in the form of a lookup for each page with what pages it links to.

## EN Wikipedia
These data need to be located in a subfolder called `en_wikipedia`. This dataset only stores the links in the main content of a page. There are four files were extracted from the 2024-08-20 Wikipedia dump.

* `enwiki_summary.tsv` - This file holds the summaries of the Wikipedia pages in a tab-separated file. It also has the PageID, MatrixIndex, and PageTitle before the Summary field.
* `enwiki_title_info.tsv` - This file holds the information of all the pages and redirects in the wiki. It includes PageID, MatrixIndex, PageTitle, IsRedirect, and RedirectTitle. These are all ordered by the PageTitle for easier searching.
* `enwiki_matrix_index_info.tsv` - This file holds the info on the pages ordered by the Matrix Index. It holds the PageID, MatrixIndex, PageTitle, and Offset. The Offset refers to the number of bytes the relevant page is in the summary file.
* `enwiki-forward.hdf5` - This has the graph representation for Wikipedia in the form of a lookup for each page with what pages it links to.

## FR Wikipedia
These data need to be located in a subfolder called `fr_wikipedia`. This dataset only stores the links in the main content of a page. There are four files were extracted from the 2024-08-20 Wikipedia dump. The descriptions of the files are as per the `en_wikipedia` dataset (but prefixed with `fr` instead of `en`)

## ZH Wikipedia
These data need to be located in a subfolder called `zh_wikipedia`. This dataset only stores the links in the main content of a page. There are four files were extracted from the 2024-08-20 Wikipedia dump. The descriptions of the files are as per the `zh_wikipedia` dataset (but prefixed with `zh` instead of `en`)
