# README Scripts

## Processing Wiki Data

Four scripts take the raw MediaWiki XML dump into a graph with associated metadata.

All of these scripts take a required argument called dataset_name.
This will give all the generated files consistent naming by prepending this to the beginning.
The examples below assume "wiki" as the dataset name.

#### wiki_xml_extract

This takes two arguments the first is the name of the file stored in "data/raw" which holds the MediaWiki XML dump, and the second is the dataset name.
This file reads the XML data and produces three outputs.

* `data/inter/wiki_metadata.tsv` - This holds information about the pages
* `data/inter/wiki_links.tsv` - This holds the link locations as they are given in the page.
* `data/output/wiki_summary.tsv` - This holds the plain text version of the first 1000 characters of the Wikipedia page.

#### wiki_make_title_table

This produces the `data/output/wiki_title_info.tsv`, which contains information about every page and redirect listed alphabetically.

#### wiki_make_index_info

This produces the `data/output/wiki_index_info.tsv`,  which holds the information about the pages, including the location in the summary file, in order of the pages.

#### wiki_build_graph

This produces the `data/output/wiki_forward_links.hdf5`, which contains an array (called `forward_vals`) of the destination of every link in the wiki and a second array (called `forward_locs`) which relates each page to the place in the list of destinations.
It is set up in this slightly complex way as it allows much quicker location of the linked pages while storing the data in a sparse format.
