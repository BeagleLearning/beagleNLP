from corpus import Corpus
from spacy.tokens import Doc

Doc.set_extension("raw_text", default="")
Doc.set_extension("question_id", default=0)
Doc.set_extension("embedding", default=0)
Doc.set_extension("cluster_id", default=0)
Doc.set_extension("lemma_list", default=[])
Doc.set_extension("clusterlabel", default=" ")

class LabellingClusterCorpus(Corpus):
    def process_docs(self, embeddings, q_ids, clusters):

        #Constructing Dictionary with Question_Id as Key and Cluster ID as the value
        question_cluster_map = {}
        for clus_id in clusters:
            for q_id in clusters[clus_id]:
                question_cluster_map[q_id] = clus_id
        
        parts_of_speech = ["NOUN", "PROPN", "VERB", "ADJ"] #Categories of Keywords/Phrases to consider as labels
        self.documents = [0 for x in range(len(embeddings))] #Initialisation
        
        #Parsing through q_ids list
        for q_index, q_id in enumerate(q_ids):
            processed_doc = self._nlp(self._raw_docs[q_index].upper())
            processed_doc._.raw_text = self._raw_docs[q_index] #Question
            processed_doc._.question_id = q_id
            processed_doc._.embedding = embeddings[q_index]
            processed_doc._.cluster_id = question_cluster_map[q_id]
            for token in processed_doc:
                if token.pos_ in parts_of_speech and token.is_stop is False and token.is_punct is False:
                    processed_doc._.lemma_list.append(str(token.lemma_).upper())
            self.documents[q_index] = processed_doc


        
            
        