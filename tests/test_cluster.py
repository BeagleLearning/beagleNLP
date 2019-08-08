import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import analysis
import cluster
import json
import math

FRACTION_IN_CLUSTER = 0.05

with open('./tests/sample_questions.json', 'r') as myfile:
    data=json.loads(myfile.read())


# Text cleaning
def test_representation_matrix():
    tagged_corpus = analysis.tagAndVectorizeCorpus(data["request"])
    assert cluster.makeRepresentationMatrix(tagged_corpus).shape == (12, 100)


def test_getMinSamples():
    nDocs = len(data["request"])
    assert max(math.ceil(nDocs * FRACTION_IN_CLUSTER), 2) == 2