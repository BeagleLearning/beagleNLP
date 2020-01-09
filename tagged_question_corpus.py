#pylint: disable=R0903
"""Storage for a collection of documents with both a numerical id and question text."""
from spacy.tokens import Doc
from corpus import Corpus
from clean_text import clean_text

Doc.set_extension("tag", default=0)
Doc.set_extension("lemmatized", default="")

class TaggedQuestionCorpus(Corpus):
    """Class to store a collection of documents that have a numerical id attribute and a question
    attribute containing their text.
    """
    def process_docs(self):
        """Process docs that have an id attribute and a question attribute."""
        self.documents = []
        if self._raw_docs and self._nlp:
            for doc in self._raw_docs:
                doc["question"] = clean_text(doc["question"].replace("\n", ""))
                if len(doc["question"]) == 0:
                    doc["question"] = " "
                processed_doc = self._nlp(doc["question"])
                processed_doc._.tag = doc["id"]
                processed_doc._.lemmatized = ' '.join([d.lemma_ for d in processed_doc])
                self.documents.append(processed_doc)
