from corpus import Corpus
from spacy.tokens import Doc
import analysis
Doc.set_extension("tag", default=0)
Doc.set_extension("lemmatized", default="")


class TaggedQuestionCorpus(Corpus):
    def processDocs(self):
        # docs that have an id attribute and a question attribute
        self.documents = []
        if self._rawDocs and self._nlp:
            for doc in self._rawDocs:
                doc["question"] = analysis.clean_text(doc["question"].replace("\n", ""))
                if len(doc["question"]) == 0:
                    doc["question"] = " "
                processedDoc = self._nlp(doc["question"])
                processedDoc._.tag = doc["id"]
                processedDoc._.lemmatized = ' '.join([d.lemma_ for d in processedDoc])
                self.documents.append(processedDoc)
