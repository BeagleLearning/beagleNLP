import math
import document_distance as dd

MAX_DOC_DIST = 0.25


def clusterOnKeywords(questions, keywords):
    # measure dist from each doc to each keyword
    # for each document, take a tf-weighted distance to each keyword
    # group with the closest keyword. But we have to figure out how to
    # choose if something is too far away. STarting with just a
    # line is probably OK
    # we do a one-way distance match for each one
    clusters = dict(zip(
        [k.text for k in keywords.documents],
        [[] for d in keywords.documents]
    ))
    clusters["uncategorized"] = []
    for doc in questions.documents:
        closestKeyword = None
        minDist = math.inf
        for kDoc in keywords.documents:
            dist = dd.directionalWordPairsDistance(kDoc, doc, "spacy")
            # print(f"{doc.text} to {kDoc.text}: {dist}")
            if dist < minDist:
                closestKeyword = kDoc
                minDist = dist

        if minDist < MAX_DOC_DIST:
            clusters[closestKeyword.text].append(doc)
        else:
            clusters["uncategorized"].append(doc)

    questions.clusters = clusters
    # print(questions.clusters)
    return questions


"""
So this system currently does clustering by looking at each suggested keyword
and measuring the distance from its words to the closest words in the document.
Interesting note about this method - if there are multiple words that are close
but different, only one of them gets counted.

So what we are not doing is figuring out if we can select a good set of keywords
from the questions that exist.

One method of doing this is to first cluster, then find defining keywords.

Another method is to cluster keywords, then choose popular clusters and average,
then reverse-search for the vector. This is going to be SLOW.

But I like the clustering method we are doing now. Its nice to consider every word
during the clustering and that doesn't match well with the keyword method.

So I was thinking we could still do our clustering first, but tag it with a keyword.
But that doesn't make much sense from an interface perspective. So maybe don't
do that right now.

So what is our keyword search method...

1) list out all keywords in questions
2) Cluster keywords
3) Filter clusters by work importance in some way (TFIDF from large dataset?)
4) Choose the single word closest to the center of the cluster
5) Convert keyword - cluster mapping to question - cluster mapping
6) For each question, measure distance to the center keyword of each cluster
that it is in. Assign it to the one it is closest to.
7)

What if we approach this like topic analysis.
Certain words relate to certain topics
Certain documents relate to certain topics

In our case we want to cluster questions and assign one keyword to each.
So what if we do look at this as tagging clusters again...
1) Cluster questions
2) Look at each token in all of those questions
3) Find the centroid of the questions
4) For each token, check distance to centroid
5) Title with closest
"""
