# Data Output

This folder is where the output files used by the other packages will be saved.

* `xxx_forward_links.hdf5`  - This file holds the list of every link between the pages as a pair of page indexes.
* `wiki_summary.tsv` - This file holds the summaries of the Wikipedia pages in a tab-separated file. It also has the PageID, MatrixIndex, and PageTitle before the Summary field.
* `wiki_title_info.tsv` - This file holds the information of all the pages and redirects in the wiki. It includes PageID, MatrixIndex, PageTitle, IsRedirect, and RedirectTitle. These are all ordered by the PageTitle for easier searching.
* `wiki_matrix_index_info.tsv` - This file holds the info on the pages ordered by the Matrix Index. It holds the PageID, MatrixIndex, PageTitle, and Offset. The Offset refers to the number of bytes the relevant page is in the summary file.
* `openalex_info.ddb` - This contains all the information needed for the OpenAlex datasets apart from the graph connectivity, which will be in the `forward_links`. It is using the DuckDB format.
