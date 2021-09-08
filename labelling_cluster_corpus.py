from corpus import Corpus

class LabellingClusterCorpus(Corpus):
    def process_docs(self, embeddings, q_ids, clusters, parts_of_speech):
        self.documents = [0 for 0 in range(embeddings)] #Initialisation
        for clus_id in clusters:
            for q_id in clusters[clus_id]:
                q_index = q_ids.index(q_id)
                processed_doc = self._nlp(self._raw_docs[q_index])
                processed_doc.raw_text = self._raw_docs[q_index] #Question
                processed_doc.qid = q_id
                processed_doc.embedding = embeddings[q_index]
                processed_doc.cluster_id = clus_id
                processed_doc.lemma_list = []
                for token in processed_doc:
                    if token.pos_ in parts_of_speech and token.is_stop is False and token.is_punct is False:
                        processed_doc.lemma_list.append(str(token.lemma_).upper())
                processed_doc.label = ' '
                self.documents[q_index] = processed_doc


        
            
        