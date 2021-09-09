from corpus import Corpus
from spacy.tokens import Doc

Doc.set_extension("raw_text", default="")
Doc.set_extension("qid", default=0)
Doc.set_extension("embedding", default=0)
Doc.set_extension("cluster_id", default=0)
Doc.set_extension("lemma_list", default=[])
Doc.set_extension("clusterlabel", default=" ")

class LabellingClusterCorpus(Corpus):
    def process_docs(self, embeddings, q_ids, clusters):
        parts_of_speech = ["NOUN", "PROPN", "VERB", "ADJ"] #Categories of Keywords/Phrases to consider as labels
        self.documents = [0 for x in range(len(embeddings))] #Initialisation
        for clus_id in clusters:
            for q_id in clusters[clus_id]:
                q_index = q_ids.index(q_id)
                processed_doc = self._nlp(self._raw_docs[q_index])
                processed_doc._.raw_text = self._raw_docs[q_index] #Question
                processed_doc._.qid = q_id
                processed_doc._.embedding = embeddings[q_index]
                processed_doc._.cluster_id = clus_id
                for token in processed_doc:
                    if token.pos_ in parts_of_speech and token.is_stop is False and token.is_punct is False:
                        processed_doc._.lemma_list.append(str(token.lemma_).upper())
                self.documents[q_index] = processed_doc


        
            
        