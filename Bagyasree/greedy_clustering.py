from math import ceil
from spacy.lang.en import STOP_WORDS
from wordfreq import word_frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from helper_functions import preprocess

class GreedyClustering:

    def tfidf(word):
        return
    
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
                if metric == 'word_frequency':
                    freq = word_frequency(key, "en")
                elif metric == 'tf_idf':
                    freq = self.tfidf(key)
            
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
        preprocessed_questions = preprocess(questions, tokenize = True)
        # print(preprocessed_questions)
        all_keyword_clusters = self.get_all_keyword_clusters(preprocessed_questions)
        # print("ALL KEYWORD CLUSTERS:",len(all_keyword_clusters))
        next_cluster_keyword = self.find_next_cluster(all_keyword_clusters)
        uncategorized_questions = set()
        for q in preprocessed_questions:
            uncategorized_questions.add(' '.join(q))
        # print("UNCATEGORIZED: ",len(uncategorized_questions))
        # print("NEXT CLUSTER KEYWORD",next_cluster_keyword)
        clusters = []
        while(next_cluster_keyword):
            clusters.append(list(all_keyword_clusters[next_cluster_keyword]))
            newly_categorized = all_keyword_clusters[next_cluster_keyword]
            del all_keyword_clusters[next_cluster_keyword]
            # print("ALL KEYWORD CLUSTERS:",len(all_keyword_clusters))
            uncategorized_questions = uncategorized_questions.difference(newly_categorized)
            # print("UNCATEGORIZED: ",len(uncategorized_questions))
            for question_groups in all_keyword_clusters.values():
                for ques in newly_categorized:
                    question_groups.discard(ques)
            next_cluster_keyword = self.find_next_cluster(all_keyword_clusters)
            # print("NEXT CLUSTER KEYWORD",next_cluster_keyword)

        clusters.append(uncategorized_questions)
        
        cluster_maps = []

        for q in preprocessed_questions:
            for i in range(0,len(clusters)):
                if ' '.join(q) in clusters[i]:
                    cluster_maps.append(i)
                    break

        return cluster_maps