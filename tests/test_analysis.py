import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import analysis

# Text cleaning
def test_clean_text():
    string_input = "There's this great, awesome, amazing place!! Have you been there? <html></html>"
    expected_output = 'theres this great awesome amazing place have you been there htmlhtml'
    assert analysis.clean_text(string_input, lower=True) == expected_output


# Text distances
word_one = "here"
word_two = "here"
phrase_one = "here is a string"
phrase_two = "here is a string"
phrase_three = "What about this other string"
phrase_four = "Are you sure about this"

def test_word_dist():
    assert analysis.dist(word_one, word_two) == 1

def test_prase_dist():
    assert analysis.dist(phrase_one, phrase_two) == 1

def test_diff_string_dist():
    assert analysis.dist(phrase_three, phrase_four) == 0.8753625873675908