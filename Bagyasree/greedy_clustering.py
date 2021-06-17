from math import ceil
from spacy.lang.en import STOP_WORDS
from wordfreq import word_frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from wordfreq.tokens import tokenize
from helper_functions import preprocess

class GreedyClustering:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = {}
        self.bigram_matrix = {}
    
    def make_bigram_matrix(self, questions):
        all_bigrams = {}
        for question in questions:
            for i in range(0,len(question)-1):
                bigram = (question[i],question[i+1])
                if bigram not in all_bigrams:
                    all_bigrams[bigram] = 1
                else:
                    all_bigrams[bigram] += 1
        
        #print("ALL_BIGRAMS:",all_bigrams)

        for bigram in all_bigrams:
            self.bigram_matrix[bigram] = all_bigrams[bigram] * 1/(word_frequency(bigram[0],"en")+1) * 1/(word_frequency(bigram[1],"en")+1) * self.tfidf_matrix[bigram[0]] * self.tfidf_matrix[bigram[1]]

        #print(self.bigram_matrix)
    
    def make_tfidf_matrix(self,questions):
        all_words = []
        for question in questions:
            all_words.extend([word for word in question])
        all_words = list(set(all_words))

        for word in all_words:
            tf = 0
            idf = 0
            for question in questions:
                freq = question.count(word)
                if freq:
                    #print(word, ":",question)
                    idf += 1
                tf += freq
            self.tfidf_matrix[word] = tf/idf

    def get_all_bigram_clusters(self,questions):

        all_bigram_clusters = {}
        for bigram in self.bigram_matrix:
            substring = bigram[0] + ' ' + bigram[1]
            if bigram not in all_bigram_clusters:
                all_bigram_clusters[bigram] = set()
            for question in questions:
                if substring in ' '.join(question):
                    all_bigram_clusters[bigram].add(' '.join(question))
        
        return all_bigram_clusters

    def find_next_bigram_cluster(self,bigram_clusters,min_cluster_size = 2, max_cluster_size = 1000):
        max_score = 0
        bigram = ()
        for bi in bigram_clusters:
            if len(bigram_clusters[bi]) >= min_cluster_size and len(bigram_clusters[bi]) <= max_cluster_size:
                score = self.bigram_matrix[bi]
                if score > max_score:
                    max_score = score
                    bigram = bi
        
        if len(bigram) == 0:
            return

        return bigram

    def get_all_keyword_clusters(self, questions):
        all_keyword_clusters = {}
        for question in questions:
            for word in question:
                if word not in all_keyword_clusters:
                    all_keyword_clusters[word] = set()
                all_keyword_clusters[word].add(' '.join(question))
        return all_keyword_clusters

    def find_next_cluster(self, keyword_clusters, metric = 'word_frequency', min_cluster_size = 2, max_cluster_size = 1000):
        lowest_freq = 1
        keyword = ''
        invalid = []
        for key in keyword_clusters:
            if len(keyword_clusters[key]) >= min_cluster_size and len(keyword_clusters[key]) <= max_cluster_size:
                # if metric == 'word_frequency':
                #     freq = word_frequency(key, "en")
                # elif metric == 'tf_idf':
                #     freq = self.tfidf(key)

                freq = word_frequency(key, "en") * self.tfidf_matrix[key]
                if 0.0 < freq < lowest_freq:
                    keyword = key
                    lowest_freq = freq
                
            else:
                invalid.append(len(keyword_clusters[key]))
        
        if keyword == '':
            # print(invalid)
            return 
    
        return keyword
    
    def greedy_clustering(self, questions):
        #preprocessed_questions = preprocess(questions, tokenize = True)
        #print(preprocessed_questions)
        preprocessed_questions = preprocess(questions, remove_stop_words = False, tokenize = True)
        self.make_tfidf_matrix(preprocessed_questions)
        self.make_bigram_matrix(preprocessed_questions)
        # print(self.bigram_matrix)
        bigram_matrix = []
        for key in self.bigram_matrix:
            bigram_matrix.append((key,self.bigram_matrix[key]))
        bigram_matrix.sort(key = lambda x: x[1])
        print(bigram_matrix[-10:])
        print()
        print()
        print()
        # all_keyword_clusters = self.get_all_keyword_clusters(preprocessed_questions)
        # # print("ALL KEYWORD CLUSTERS:",len(all_keyword_clusters))
        # next_cluster_keyword = self.find_next_cluster(all_keyword_clusters)

        all_bigram_clusters = self.get_all_bigram_clusters(preprocessed_questions)
        next_cluster_bigram = self.find_next_bigram_cluster(all_bigram_clusters)

        uncategorized_questions = set()
        for q in preprocessed_questions:
            uncategorized_questions.add(' '.join(q))
        # print("UNCATEGORIZED: ",len(uncategorized_questions))
        # print("NEXT CLUSTER KEYWORD",next_cluster_keyword)
        clusters = []
        # while(next_cluster_keyword):
        #     clusters.append(list(all_keyword_clusters[next_cluster_keyword]))
        #     newly_categorized = all_keyword_clusters[next_cluster_keyword]
        #     del all_keyword_clusters[next_cluster_keyword]
        #     # print("ALL KEYWORD CLUSTERS:",len(all_keyword_clusters))
        #     uncategorized_questions = uncategorized_questions.difference(newly_categorized)
        #     # print("UNCATEGORIZED: ",len(uncategorized_questions))
        #     for question_groups in all_keyword_clusters.values():
        #         for ques in newly_categorized:
        #             question_groups.discard(ques)
        #     next_cluster_keyword = self.find_next_cluster(all_keyword_clusters)
        #     # print("NEXT CLUSTER KEYWORD",next_cluster_keyword)

        while(next_cluster_bigram):
            clusters.append(list(all_bigram_clusters[next_cluster_bigram]))
            newly_categorized = all_bigram_clusters[next_cluster_bigram]
            del all_bigram_clusters[next_cluster_bigram]
            uncategorized_questions = uncategorized_questions.difference(newly_categorized)
            for question_groups in all_bigram_clusters.values():
                for ques in newly_categorized:
                    question_groups.discard(ques)
            next_cluster_bigram = self.find_next_bigram_cluster(all_bigram_clusters)

        clusters.append(uncategorized_questions)
        
        cluster_maps = []

        for q in preprocessed_questions:
            for i in range(0,len(clusters)):
                if ' '.join(q) in clusters[i]:
                    cluster_maps.append(i)
                    break

        return cluster_maps