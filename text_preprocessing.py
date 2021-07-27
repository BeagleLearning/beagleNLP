import re

def remove_special_characters(sentence) -> str:
    """
    Takes in a string and returns the same string with special characters removed.
    """
    if type(sentence) == str:
        return re.sub(r"[^A-Za-z0-9,.? ]", r"", sentence)
    return "Input is not a string!"