# Graph Preparation Readme

This package does the required data pre-processing needed to produce navigable graph datasets for agents to play the Wikipedia Game. The agents that play the game along with the experimental scripts are available in the `agents` package.

It is possible to either process data from [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia) or from [OpenAlex](https://help.openalex.org/hc/en-us/articles/24396686889751-About-us), which is a collection of scientific literature.
These data preparation tools can work on the various language versions of Wikipedia and can also filter the academic fields in OpenAlex to produce datasets focused on a particular topic.

## Table of Contents

- [Set-up](#set-up)
- [Data Sources](#data-sources)
- [Processing Wiki Data](#processing-wiki-data)
- [Processing OpenAlex Data](#processing-openalex-data)
- [Graph Analysis](#graph-analysis)
- [Output](#output)

## Set-up

### Installing

Assuming the repo has already been cloned, change the directory to the `graph-prep` package and install with `pip`:

```bash
pip install .
```

## Data Sources

This package processes the data from the openly available Wikipedia and OpenAlex datasets.

These datasets can be very large; the uncompressed size of the latest OpenAlex snapshot is 1.6 TB. Data from both of these datasets can also be accessed via an API ([Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page) and [OpenAlex API](https://docs.openalex.org/how-to-use-the-api/api-overview)). To construct a graph of these datasets, the full dataset needs to be accessed; for other use cases, the API might be more appropriate.

### Wikipedia Data XML

This script set goes from a copy of a MediaWiki XML dump to a collection of files to explore Wikipedia as a graph. The XML files are available to download, and the file required is called `page-articles`.

These are available at: https://dumps.wikimedia.org/

These page article dumps contain all the page articles on Wikipedia and all the necessary metadata to construct the graph; these can be large, around 100 GB. Various language versions exist for English Wikipedia; the dataset will named in the form `enwiki-YYYYMMDD-pages-articles.xml`.

### OpenAlex

The OpenAlex snapshot is available from an s3 bucket, which contains the data as a JSON line for each article. All of the various academic fields in OpenAlex are stored together so the whole snapshot will be required to produce any of the graph datasets. This dataset is large 330 GB and decompresses to about 1.6 TB.

This is available at  https://docs.openalex.org/download-all-data/openalex-snapshot

## Processing Wiki Data

Four scripts take the raw MediaWiki XML dump into a graph with associated metadata.

All of these scripts take a required argument called dataset_name.
This will give all the generated files consistent naming by prepending this to the beginning.
The examples below assume "wiki" as the dataset name.

#### Extract Wikimedia XML

`wiki_xml_extract.py`

This takes two arguments the first is the name of the file stored in "data/raw" which holds the MediaWiki XML dump, and the second is the dataset name.
This file reads the XML data and produces three outputs.

* `data/inter/wiki_metadata.tsv` - This holds information about the pages
* `data/inter/wiki_links.tsv` - This holds the link locations as they are given in the page.
* `data/output/wiki_summary.tsv` - This holds the plain text version of the first 1000 characters of the Wikipedia page.

#### Build the Dataset of Titles

`wiki_make_title_table.py`

This produces the `data/output/wiki_title_info.tsv`, which contains information about every page and redirects listed alphabetically.

#### Index the Articles in the Dataset

`wiki_make_index_info.py`

This produces the `data/output/wiki_index_info.tsv`,  which holds the information about the pages, including the location in the summary file, in order of the pages.

#### Build the Graph

`wiki_build_graph.py`

This produces the `data/output/wiki_forward_links.hdf5`, which contains an array (called `forward_vals`) of the destination of every link in the wiki and a second array (called `forward_locs`) which relates each page to the place in the list of destinations.
It is set up in this slightly complex way as it allows much quicker location of the linked pages while storing the data in a sparse format.

## Processing OpenAlex Data

#### Extract the OpenAlex Data

`openalex_extract.py`

This script requires the raw OpenAlex data to have been downloaded into the `data/raw/openalex` folder. These data will be stored in subfolders. This script extracts the data from these files and stores them in a DuckDB database in the `data/inter` folder. The rules about what data are extracted can be found in the `src/graph_prep/read_openalex.py` file. Currently, these rules are hard-coded and include only extracting works of type `article` and only fields that are in one of:

* Medicine
* Arts and Humanities
* Biochemistry, Genetics and Molecular Biology
* Physics and Astronomy
* Social Sciences

#### Build the OpenAlex Graph

`openalex_field_graph.py`

This script takes the name of one of the fields as an argument and processes these data into its own database and and hdf5 file of forward links. Another argument is a `short_name` which is given to the output that is saved; for example, the field `Physics and Astronomy` might be given a `short_name` of `physics`. The output of this script is saved in the `data/output` folder and will be needed by the `agents` package to play the games.

## Graph Analysis

These scripts use the HDF5 file of forward links common to the output of both the Wiki and OpenAlex data processing.

#### Generate Graph Connectivity Histogram

`graph_step_histogram.py`

This produces a .csv output of a histogram of the number of the minimum number of steps to complete a game between two randomly selected articles. The arguments that should be set are the dataset to use, the maximum number of steps to consider and the number of 'games' (paths) to generate. If a game isn't solved within the maximum number of steps then it is recorded as a value of zero.

#### Generate a List of Games

`graph_step_list.py`

This produces a .csv output of a list of 'games'. Each game consists of a start page/article ID and an end page/article ID. Also provided is the minimum number of steps needed to navigate from the start page to the end page. If the game cannot be completed within the maximum number of steps provided as an argument, then a value of zero is recorded. The other arguments are the dataset to use and the number of 'games' to generate.

## Output

The output files for Wikipedia and the OpenAlex datasets are slightly different. The choice was made to use DuckDB for the OpenAlex graphs to avoid needing to hold them in memory. The format of the graph information is the same for both formulations of the problem.

### Grpah Connectivity

* `forward_links.hdf5` - A HDF5 data file which provides which pages are linked to from every page. This contains an array of every link represented as a pair of integers. These are the start index and the target index.

### Wikipedia Data

When the processing is complete, four files are produced which describe the Wikipedia dataset.

* `title_info.tsv` - Lists all the valid page titles (including redirects) in alphabetical order.
* `summary.tsv` - Lists all the valid pages and includes the beginning of the page in plain text (default cut to a thousand charters).
* `index_info.tsv` - List all the valid pages indexed by consecutive integers which relate to nodes in the graph. This also contains the location of the associated page in the binary of the `summary` to allow quick access.

### OpenAlex Data

* `info.ddb` - This contains all the needed metadata from the OpenAlex dataset. This includes the title, page index, abstract, language, and OpenAlexID.
