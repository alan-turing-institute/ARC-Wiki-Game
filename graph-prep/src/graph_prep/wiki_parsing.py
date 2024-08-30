from __future__ import annotations

import re


def make_summary(text: str, cut_off: int = 1000) -> str:
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
    text = re.sub(r"^thumb\|.*?\n", "", text)
    # Remove any newlines and tabs
    text = re.sub(r"\t|\n", " ", text)
    return text[:cut_off]


def pull_link_location(link_text: str) -> str | None:
    """Extracts the link location from a link text.

    This parses the link text to that only the title remains, so it can be used in a
    lookup. If the location is empty or starts with 'File:' or 'Category:',then the
    link is rejected and returns None.

    Args:
        link_text (str): The link text. This is extracted from the wikimedia xml dump,
            using the wikimediaparserfromhell library.

    Returns:
        str | None: The extracted location or None if the location is empty or starts
            with 'File:' or 'Category:' or is a link to a section of the same page.
    """
    # Extract the location from the link text
    link_match = re.match(r"\[\[(.*?)[\||\]]", str(link_text))
    if not link_match:  # checks if link is badly formatted
        return None
    link_loc = link_match.group(1)
    # Remove any section links
    link_loc = re.sub(r"\#.*", "", link_loc)
    # Remove any leading or trailing whitespace
    link_loc = re.sub(r"^\s+", "", link_loc)
    link_loc = re.sub(r"\s+$", "", link_loc)
    link_loc = re.sub(r"[\t|\n]", "", link_loc)
    # Reject if the location is empty or is in the File or Category namespace
    if link_loc == "" or re.match(r"[File:|Category:]", link_loc):
        return None
    # Capitalize the first letter
    return link_loc[0].upper() + link_loc[1:]
