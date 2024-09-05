from __future__ import annotations

import re


class WikiInfoParser:
    """A class to parse and format information from a wikimedia xml dump.

    This class has three main functions and is only a class because it is quicker
    to compile the regular expressions once and store them as class variables.

    The three main functions are:
    make_summary: Generates a summary from the text of a page.
    format_title: Formats the title of a page.
    pull_link_location: Extracts the location from a link text.
    """

    def __init__(self):
        self.re_caption_cut = re.compile(r"^thumb\|.*?\n")
        self.re_title_clean = re.compile(r"^\s+|\n|\t|\||\s+$")
        self.re_link_match = re.compile(r"\[\[(.*?)[\||\]]")
        self.re_section_cut = re.compile(r"\#.*")
        self.re_file_cat_match = re.compile(r"File:|Category:")

    def make_summary(self, text: str, cut_off: int = 1000) -> str:
        """Generate a summary of the given text.

        Args:
        - text (str): The input text to generate a summary from. This is from the
            `wikimediaparserfromhell` library and has had `strip_code()` applied to it.
        - cut_off (int, optional): The maximum number of characters for the summary.
            Default is 1000.

        Returns:
        - str: The generated summary of the text.
        """
        # Strip the caption of an image if that is the first line of the text
        text = self.re_caption_cut.sub("", text)
        # Remove any newlines and tabs
        text = self.re_title_clean.sub(" ", text)
        return text[:cut_off]

    def format_title(self, title: str) -> str:
        """Formats the title of a page.

        Args:
        - title (str): The title of the page.

        Returns:
        - str: The formatted title of the page.
        """
        # Remove any leading or trailing whitespace
        title = self.re_title_clean.sub("", title)
        # Capitalize the first letter
        if title == "":
            return title

        return title[0].upper() + title[1:]

    def pull_link_location(self, link_text: str) -> str | None:
        """Extracts the link location from a link text.

        This parses the link text to that only the title remains, so it can be used in a
        lookup. If the location is empty or starts with 'File:' or 'Category:',then the
        link is rejected and returns None.

        Args:
            link_text (str): The link text. This is extracted from the wikimedia xml
                dump, using the wikimediaparserfromhell library.

        Returns:
            str | None: The extracted location or None if the location is empty or
                starts with 'File:' or 'Category:' or is a link to a section of the
                same page.
        """
        # Extract the location from the link text
        link_match = self.re_link_match.match(str(link_text))
        if not link_match:  # checks if link is badly formatted
            return None
        link_loc = link_match.group(1)
        # Remove any section links
        link_loc = self.re_section_cut.sub("", link_loc)
        # Remove any leading or trailing whitespace
        link_loc = self.re_title_clean.sub("", link_loc)
        # Reject if the location is empty or is in the File or Category namespace
        if link_loc == "" or self.re_file_cat_match.match(link_loc):
            return None
        # Capitalize the first letter
        return link_loc[0].upper() + link_loc[1:]
