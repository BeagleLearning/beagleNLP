# pylint: disable=C0413
"""Tests the greedy clusterer algorithm."""
import os
import inspect
import sys
import json
import spacy
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
import tfidf
from tagged_question_corpus import TaggedQuestionCorpus
from greedy_clusterer import cluster_greedily

NLP = spacy.load("en_core_web_lg")
NLP.add_pipe(tfidf.generate_text_frequency, name="text_frequency")

def import_test_data():
    """Load question data into format expected by cluster_greedily."""
    with open('./tests/global-sample.json', 'r') as myfile:
        loaded_data = json.loads(myfile.read())
        res = {"request": []}
        for i, question in enumerate(loaded_data["questions"]):
            res["request"].append({"id": i, "question": question})
    return res, loaded_data

print("Test of greedy clusterer")
DATA, TEST_DATA = import_test_data()
TAGGED_DATA = TaggedQuestionCorpus(DATA["request"], NLP)
RESULT = cluster_greedily(TAGGED_DATA)
for cluster in RESULT:
    print("\n\n\n", cluster[0], "\n\n\n")
    for q in cluster[1]:
        print(q)
