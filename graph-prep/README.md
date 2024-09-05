# Graph Preparation Readme

This script set goes from a copy of a MediaWiki XML dump to a collection of files to explore Wikipedia as a graph.

The XML files are available to download, and the file required is called `page-articles`.

When the processing is complete four files are produced.

* `title_info` - Lists all the valid page titles (including redirects) in alphabetical order.
* `index_info` - List all the valid pages indexed by consecutive integers which relates to nodes in the graph.
* `summary` - Lists all the valid pages and includes the beginning of the page in plain text (default cut to a thousand charters).
* `forward_links` - Prodives which pages are linked to from every page in Wikipedia.

More details of how to run these scripts are given in `script/README-script.md`.
