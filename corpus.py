#pylint: disable=R0903
"""Storage for documents that handles initial processing of raw docs with NLP."""
class Corpus:
    """Container class for a set of documents."""
    def __init__(self, docs, nlp):
        self._raw_docs = docs
        self._nlp = nlp
        self.documents = []
        n_docs = len(self._raw_docs)
        self.distance_matrix = [[1] * n_docs] * n_docs
        self.process_docs()
        

    def process_docs(self):
        """Processes passed-in documents with passed-in NLP."""
        if self._raw_docs and self._nlp:
            self.documents = [self._nlp(doc) for doc in self._raw_docs]

    def remove_unknown_docs(self):
        """Removes documents for which we don't recognize any words.

        Returns:
            bool: True if any documents were removed
        """
        count_before = len(self.documents)
        self.documents = list(filter(__at_least_one_token_known, self.documents))
        return count_before > len(self.documents)

def __at_least_one_token_known(doc):
    """Returns true if any word in a doc is recognized."""
    one_good = False
    for token in doc:
        if not token.is_oov:
            one_good = True
    return one_good
