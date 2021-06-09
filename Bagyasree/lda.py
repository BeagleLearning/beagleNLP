from gensim import matutils, models, corpora
from scipy import sparse
import pandas as pd

class LatentDirichletAllocation:
    
    def get_vocab(self, questions):
        words = sum(questions, [])
        words = sorted(list(set(words)))
        return words

    def get_matrix(self, questions, td = 1):
        words = self.get_vocab(questions)
        
        index = 0

        no_of_docs = len(questions)
        empty = [0 for i in range(no_of_docs)]

        word_dict = {}
        for word in words:
            word_dict[word] = empty

        matrix = pd.DataFrame(word_dict)

        for doc in questions:   
            for word in doc:
                matrix.iloc[index][word] = 1
            index += 1
        
        if td == 1:
            matrix = matrix.transpose()
        
        return matrix

    def create_sparse_obj(self, matrix):
        counts = sparse.csr_matrix(matrix)
        sparse_matrix = matutils.Sparse2Corpus(counts)
        return sparse_matrix

    def make_vocab_dict(self, words):
        ret_dict = corpora.Dictionary(words)
        return ret_dict

    def lda(self,questions):
        vocab = self.get_vocab(questions)
        matrix = self.get_matrix(questions)
        sp_matrix = self.create_sparse_obj(matrix)
        vocab_dict = self.make_vocab_dict([vocab])
        lda = models.LdaModel(corpus = sp_matrix, id2word = vocab_dict, num_topics = 4, passes = 80)
        for ques in questions[:10]:
            print(lda[ques])