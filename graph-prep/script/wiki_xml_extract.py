from __future__ import annotations

import os
import re

import mwparserfromhell as hell
import mwxml
from tqdm import tqdm

from graph_prep.wiki_parsing import make_summary, pull_link_location

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml_dump_path = os.path.join(main_dir, "data/raw/enwiki-20240720-pages-articles.xml")
metadata_tsv_path = os.path.join(main_dir, "data/inter/wiki_metadata.tsv")
summary_tsv_path = os.path.join(main_dir, "data/output/wiki_summary.tsv")
link_tsv_path = os.path.join(main_dir, "data/inter/wiki_links.tsv")

# This is a large file, so we count the number of pages first to give a time estimate
total_pages = 0
with open(xml_dump_path) as xml_f:
    for line in tqdm(xml_f, desc="Counting Pages"):
        if re.match(r"^  <page>", line):
            total_pages += 1


matrix_index = 0
with (
    open(xml_dump_path) as xml_f,
    open(metadata_tsv_path, "w") as metadata_f,
    open(summary_tsv_path, "w") as summary_f,
    open(link_tsv_path, "w") as link_f,
):
    # Start by setting up the files with the headers
    dump = mwxml.Dump.from_file(xml_f)
    metadata_f.write("PageID\tMatrixIndex\tPageTitle\tIsRedirect\tRedirectLocation\n")
    summary_f.write("PageID\tMatrixIndex\tPageTitle\tSummary\n")
    link_f.write("PageID\tMatrixIndex\tPageTitle\tLinks\n")
    # Loop through all the pages in the large XML file
    for page in tqdm(dump, total=total_pages, desc="Processing XML"):
        if page.namespace != 0:
            continue
        if page.redirect:
            # Save the redirect information
            metadata_f.write(f"{page.id}\t-1\t{page.title}\t1\t{page.redirect}\n")
        else:
            matrix_index += 1
            # Save the metadata
            metadata_f.write(
                f"{page.id}\t{matrix_index}\t{matrix_index}\t{page.title}\t0\t-\n"
            )
            for revision in page:
                wiki_text = hell.parse(revision.text)
                # Save the summary (The 3000 is just to speed this up a little)
                summary = make_summary(wiki_text.strip_code()[:3000], cut_off=1000)
                summary_f.write(f"{page.id}\t{matrix_index}\t{page.title}\t{summary}\n")
                # Save the links
                link_list = []
                for link in wiki_text.ifilter_wikilinks():
                    link_location = pull_link_location(link)
                    if link_location:
                        link_list.append(link_location)
                # The links are separated by a pipe character in a TSV file
                text_links = "|".join(link_list)
                link_f.write(f"{page.id}\t{matrix_index}\t{page.title}\t{text_links}\n")
                break  # Only need the first revision
