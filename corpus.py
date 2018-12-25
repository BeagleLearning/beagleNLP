class Corpus:
    def __init__(self, docs, nlp):
        self._rawDocs = docs
        self._nlp = nlp
        self.documents = []
        nDocs = len(self._rawDocs)
        self.distanceMatrix = [[1] * nDocs] * nDocs
        self.processDocs()

    def processDocs(self):
        if self._rawDocs and self._nlp:
            self.documents = [self._nlp(doc) for doc in self._rawDocs]
