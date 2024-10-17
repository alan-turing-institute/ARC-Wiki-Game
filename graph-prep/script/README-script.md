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

## Processing OpenAlex Data

### openalex_extract

This script requires the raw OpenAlex data to have been downloaded into the `data/raw/openalex` folder. These data will be stored in subfolders. This script extracts the data from these files and stores them into a DuckDB database in the `data/inter` folder. The rules about what data are extracted can be found in the `src/graph_prep/read_openalex.py` file. Currently these rules are hard-coded and include only extracting works of type `article` and only fields that are in one of:

* Medicine
* Arts and Humanities
* Biochemistry, Genetics and Molecular Biology
* Physics and Astronomy
* Social Sciences

### openalex_field_graph

This script takes the name of one of the fields as an argument and processes these data into its own database and and hdf5 file of forward links. Another argument is a `short_name` which is given to the output that is saved; for example, the field `Physics and Astronomy` might be given a `short_name` of `physics`. The output of this script is saved in the `data/output` folder and will be needed by the `agents` package to play the games.

## Graph Scripts

These scripts use the hdf5 file of forward links which are common to the output of both the Wiki and OpenAlex data processing.

### graph_step_histogram

This produces a .csv output of a histogram of the number of the minimum number of steps to complete a game between two randomly selected articles. The arguments that should be set are the dataset to use, the maximum number of steps to consider and the number of 'games' (paths) to generate. If a game isn't solved within the maximum number of steps then it is recorded as a value of zero.

### graph_step_list

This produces a .csv output of a list of 'games'. Each game consists of a start page/article ID and an end page/article ID. Also provided is the minimum number of steps needed to navigate from the start page to the end page. If the game cannot be completed within the maximum number of steps provided as an argument, then a value of zero is recorded. The other arguments are the dataset to use and the number of 'games' to generate.
