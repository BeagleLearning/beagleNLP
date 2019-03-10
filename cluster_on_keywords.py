import math
import document_distance as dd

MAX_DOC_DIST = 1.5


def clusterOnKeywords(questions, keywords):
    # no keywords - leave everything undefined
    if len(keywords.documents) == 0:
        clusters = {}
        clusters["uncategorized"] = questions.documents
        questions.clusters = clusters
        return questions

    # measure dist from each doc to each keyword
    # for each document, take a tf-weighted distance to each keyword
    # group with the closest keyword. But we have to figure out how to
    # choose if something is too far away. Starting with just an
    # arbitrary line is probably OK
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
            if not keywords.atLeastOneTokenKnown(kDoc):
                # skip this one if the document does not have any tokens we
                # know. In that case we can't make an accurate distance measure.
                continue

            dist = dd.directionalWordPairsDistance(kDoc, doc, "cosine")
            if dist < minDist:
                closestKeyword = kDoc
                minDist = dist

        print(f"min dist is {minDist}")
        if minDist < MAX_DOC_DIST:
            clusters[closestKeyword.text].append(doc)
        else:
            clusters["uncategorized"].append(doc)

    questions.clusters = clusters
    return questions
