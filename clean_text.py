"""Removes sepcial characters from text."""
import regex as re

def clean_text(text, lower=False):
    """
    Remove special characters

    Parameters:
        text (str): The string to be cleaned
        lower (bool): A flag to determine whether to lower case the text or not
            (default is False)

    Returns:
        string: a string that is free of special/punctuation characters
    """
    text = text.lower() if lower else text
    # We have been replacing "'" with nothing. This leads to "countrys" or "hes"
    # instead of "country" and "he". Replace instead with space if anything.
    text = text.replace("'", " ")
    text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", text)
    return re.sub(r"[^\w ]", "", text)
