from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.fasttext import FastText, load_facebook_model
import math
import numpy as np
from clustering_methods import ClusteringMethods
import io

class WordEmbeddings:

    def load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def get_vectors_for_questions(self, model, questions):
        vectors = []
        length = 0
        for ques in questions:
            vect = []
            for word in ques:
                try:
                    word_vector = model[word]
                    if length == 0:
                        length = len(word_vector)
                    vect.append(word_vector)
                except:
                    #print(word, "is not in the vocabulary")
                    pass
            
            if len(vect):
                vect = sum(vect)
                vectors.append(vect)
            else:
                if length:
                    vect = [0] * length
                    vectors.append(vect)
            
        vectors = np.array(vectors)
        return vectors

    def word_2_vec(self, questions, model_type = 'pretrained'):
        #model type: pretrained OR custom.

        if model_type == 'pretrained':
            model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True, limit = 100000)
        else:
            model = Word2Vec(questions, min_count = 1, size = 300)
        
        vectors = self.get_vectors_for_questions(model,questions)
        clusters = ClusteringMethods.agglomerative_clustering(vectors, math.floor(math.sqrt(len(questions))))
        return clusters
    
    def fasttext(self, questions, model_type = 'pretrained'):

        if model_type == 'pretrained':
            fname = 'wiki-news-300d-1M.vec'
            model = self.load_vectors(fname)
        else:
            # Defining values for parameters
            embedding_size = 300
            window_size = 5
            min_word = 5
            down_sampling = 1e-2
            
            model = FastText(questions, size=embedding_size, window=window_size, min_count=min_word, sample=down_sampling, sg=1,iter=100)

        vectors = self.get_vectors_for_questions(model,questions)
        clusters = ClusteringMethods.agglomerative_clustering(vectors, math.floor(math.sqrt(len(questions))))
        return clusters

    def doc_2_vec(self, questions):
        tagged_docs = [TaggedDocument(ques, [i]) for i, ques in enumerate(questions)]
        dmodel = Doc2Vec(tagged_docs, vector_size=10, window=2, min_count=1, workers=4)

        training_data = []
        for i, ques in enumerate(questions):
            training_data.append(dmodel.docvecs[i])
        
        clusters = ClusteringMethods.agglomerative_clustering(training_data, math.sqrt(len(questions)))
        return clusters