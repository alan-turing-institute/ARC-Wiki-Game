# Data Output

This folder is where the output files that will be used by the other packages will be saved.

* **wiki_summary.tsv** - This file holds the summaries of the Wikipedia pages in a tab-separated file. It also has the PageID, MatrixIndex, and PageTitle before the Summary field.
* **wiki_title_info.tsv** - This file holds the information of all the pages and redirects in the wiki. It includes PageID, MatrixIndex, PageTitle, IsRedirect, and RedirectTitle. These are all ordered by the PageTitle for easier searching.
* **wiki_matrix_index_info.tsv** - This file holds the info on the pages ordered by the Matrix Index. It holds the PageID, MatrixIndex, PageTitle, and Offset. The Offset refers to the number of bytes the relevant page is in the summary file.
