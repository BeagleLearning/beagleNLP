import math
import analysis as a
import document_distance as dd

MAX_DOC_DIST = 10000

# when exporting the clusters... we ought just to do it by id


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
            clusters[closestKeyword.text].append(doc._.tag)
        else:
            clusters["uncategorized"].append(doc._.tag)

    questions.clusters = clusters
    # print(questions.clusters)
    return questions
