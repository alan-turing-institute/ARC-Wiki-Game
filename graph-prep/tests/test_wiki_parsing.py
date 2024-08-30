from __future__ import annotations

from graph_prep.wiki_parsing import make_summary, pull_link_location


def test_make_summary():
    # Test case 1: Text is shorter than cut-off length
    text = "This is a short text."
    expected_summary = "This is a short text."
    assert make_summary(text) == expected_summary

    # Test case 2: Text is longer than cut-off length
    text = "This is a long text. It has more than 1000 characters. " * 10
    expected_summary = "This is a long text. It has more than 1000 characters. " * 10
    assert make_summary(text) == expected_summary[:1000]

    # Test case 3: Text contains image caption
    text = "thumb|Image caption\nThis is a text with an image caption."
    expected_summary = "This is a text with an image caption."
    assert make_summary(text) == expected_summary

    # Test case 4: Text contains newlines and tabs
    text = "This\nis\ta\ttext\nwith\nnewlines\tand\ttabs."
    expected_summary = "This is a text with newlines and tabs."
    assert make_summary(text) == expected_summary

    # Test case 5: Text is empty
    text = ""
    expected_summary = ""
    assert make_summary(text) == expected_summary

    # Test case 6: Text is exactly the cut-off length
    text = "This is many characters long." * 10
    expected_summary = "This is many"
    assert make_summary(text, cut_off=12) == expected_summary


def test_pull_link_location():
    # Test case 1: Valid link text without section
    link_text = "[[Page]]"
    expected_location = "Page"
    assert pull_link_location(link_text) == expected_location

    # Test case 2: Link with information about the page
    link_text = "[[Page|talk about the page]]"
    expected_location = "Page"
    assert pull_link_location(link_text) == expected_location

    # Test case 3: Missing capitalization
    link_text = "[[page]]"
    expected_location = "Page"
    assert pull_link_location(link_text) == expected_location

    # Test case 4: Valid link text with section
    link_text = "[[Page#Section]]"
    expected_location = "Page"
    assert pull_link_location(link_text) == expected_location

    # Test case 5: Valid link text with leading and trailing whitespace
    link_text = "[[  Page  ]]"
    expected_location = "Page"
    assert pull_link_location(link_text) == expected_location

    # Test case 6: Invalid link text starting with 'File:'
    link_text = "[[File:Image.jpg]]"
    expected_location = None
    assert pull_link_location(link_text) == expected_location

    # Test case 7: Invalid link text starting with 'Category:'
    link_text = "[[Category:Category]]"
    expected_location = None
    assert pull_link_location(link_text) == expected_location

    # Test case 8: Invalid link text with empty location
    link_text = "[[]]"
    expected_location = None
    assert pull_link_location(link_text) == expected_location

    # Test case 9: Invalid link text with badly formatted link
    link_text = "Page]]"
    expected_location = None
    assert pull_link_location(link_text) == expected_location
