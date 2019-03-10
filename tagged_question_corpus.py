from corpus import Corpus
from spacy.tokens import Doc
Doc.set_extension("tag", default=0)


class TaggedQuestionCorpus(Corpus):
    def processDocs(self):
        # docs that have an id attribute and a question attribute
        self.documents = []
        if self._rawDocs and self._nlp:
            for doc in self._rawDocs:
                doc["question"] = doc["question"].replace("\n", "")
                processedDoc = self._nlp(doc["question"])
                processedDoc._.tag = doc["id"]
                self.documents.append(processedDoc)
