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
        self.combined_matrix = {}

    def make_combined_matrix(self, questions):
        all_bigrams = {}

        for question in questions:
            bigrams = []
            for i in range(0,len(question)-1):
                bigram = (question[i],question[i+1])
                if bigram not in all_bigrams:
                    all_bigrams[bigram] = 1.0
                else:
                    all_bigrams[bigram] += 1.0

        #normalize wrt number of bigrams/words
        for bigram in all_bigrams:
            bigram_freq = all_bigrams[bigram]
            word_freq = self.tfidf_matrix[bigram[0]]
            if bigram_freq/word_freq > 0.5:
                self.combined_matrix[bigram[0]] = 1/bigram_freq #normalize 
                print("COMBINED:",self.combined_matrix[bigram[0]])
            else:
                self.combined_matrix[bigram[0]] = word_freq * word_frequency(bigram[0],"en") #normalize
                print("PLAIN KEYWORD:",self.combined_matrix[bigram[0]])
            
            word_freq = self.tfidf_matrix[bigram[1]]
            if bigram_freq/word_freq > 0.5:
                self.combined_matrix[bigram[1]] = bigram_freq #normalize
            else:
                #print(word_frequency(bigram[1],"en")+1)
                self.combined_matrix[bigram[1]] = word_freq * word_frequency(bigram[1],"en") #normalize
        
        # for key in self.combined_matrix:
        #     print(key, ":", self.combined_matrix[key])
    
    def make_bigram_matrix(self, questions):
        all_bigrams = {}
        total_bigram_frequency = {}
        document_bigram_frequency = {}
        for question in questions:
            bigrams = []
            for i in range(0,len(question)-1):
                bigram = (question[i],question[i+1])

                #Document frequency for each bigram
                # if bigram not in bigrams:
                #     if bigram not in document_bigram_frequency:
                #         document_bigram_frequency[bigram] = 1
                #     else:
                #         document_bigram_frequency[bigram] += 1
                #     bigrams.append(bigram)

                #Total bigram frequency, across all documents.
                if bigram not in total_bigram_frequency:
                    total_bigram_frequency[bigram] = 1
                else:
                    total_bigram_frequency[bigram] += 1

                
        
        # #print("ALL_BIGRAMS:",all_bigrams)
        # print("TOTAL:",total_bigram_frequency)
        # print("DOC:",document_bigram_frequency)

        #print(total_bigram_frequency.keys())

        for bigram in total_bigram_frequency:
            print(bigram, ":", total_bigram_frequency[bigram], self.tfidf_matrix[bigram[0]], self.tfidf_matrix[bigram[1]])
            self.bigram_matrix[bigram] = total_bigram_frequency[bigram] * (((word_frequency(bigram[0],"en")) * (word_frequency(bigram[1],"en"))) * (self.tfidf_matrix[bigram[0]] * self.tfidf_matrix[bigram[1]]))
            #self.bigram_matrix[bigram] = total_bigram_frequency[bigram] * 
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
                    idf += 1.0
                tf += float(freq)
            
            #tf-idf
            self.tfidf_matrix[word] = tf/idf

            #tf only
            self.tfidf_matrix[word] = tf

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
        least_score = 1
        bigram = ()
        for bi in bigram_clusters:
            if len(bigram_clusters[bi]) >= min_cluster_size and len(bigram_clusters[bi]) <= max_cluster_size:
                score = self.bigram_matrix[bi]
                if score > max_score:
                    max_score = score
                    bigram = bi
                #print(score)
                # if 0 < score < least_score:
                #     least_score = score
                #     bigram = bi
        
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

    def find_next_cluster(self, keyword_clusters, metric = 'combined', min_cluster_size = 2, max_cluster_size = 1000):
        lowest_freq = 1
        highest_freq = 0
        keyword = ''
        invalid = []
        for key in keyword_clusters:
            if len(keyword_clusters[key]) >= min_cluster_size and len(keyword_clusters[key]) <= max_cluster_size:
                # if metric == 'word_frequency':
                #     freq = word_frequency(key, "en")
                # elif metric == 'tf_idf':
                #     freq = self.tfidf(key)

                #print(word_frequency(key,"en"))
                if metric == 'word_frequency':
                    freq = word_frequency(key, "en") * self.tfidf_matrix[key]
                    if 0.0 < freq < lowest_freq:
                        keyword = key
                        lowest_freq = freq
                
                else:
                    freq = self.combined_matrix[key]
                    if 0 < freq < lowest_freq:
                        keyword = key
                        lowest_freq = freq
                    # if freq > highest_freq:
                    #     keyword = key
                    #     highest_freq = freq

                # if freq > highest_freq:
                #     keyword = key
                #     highest_freq = freq
                
            else:
                invalid.append(len(keyword_clusters[key]))
        
        if keyword == '':
            # print(invalid)
            return 
    
        return keyword
    
    def greedy_clustering(self, questions, cluster_by = 'keyword'):
        
        preprocessed_questions = preprocess(questions, tokenize = True)
        self.make_tfidf_matrix(preprocessed_questions)
        self.make_combined_matrix(preprocessed_questions)

        clusters = []

        if cluster_by == 'bigram':
            self.make_bigram_matrix(preprocessed_questions)
            bigram_matrix = []
            for key in self.bigram_matrix:
                bigram_matrix.append((key,self.bigram_matrix[key]))
            bigram_matrix.sort(key = lambda x: x[1], reverse=True)
            print(bigram_matrix[:10])
            print()
            print()
            print()
            
            all_bigram_clusters = self.get_all_bigram_clusters(preprocessed_questions)
            next_cluster_bigram = self.find_next_bigram_cluster(all_bigram_clusters)

            uncategorized_questions = set()
            for q in preprocessed_questions:
                uncategorized_questions.add(' '.join(q))

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

        else:
            
            combined_matrix = []
            for key in self.combined_matrix:
                combined_matrix.append((key,self.combined_matrix[key]))
            combined_matrix.sort(key = lambda x: x[1])
            print(combined_matrix[:10])
            print()
            print()
            print()

            all_keyword_clusters = self.get_all_keyword_clusters(preprocessed_questions)
            # print("ALL KEYWORD CLUSTERS:",len(all_keyword_clusters))
            next_cluster_keyword = self.find_next_cluster(all_keyword_clusters)

            uncategorized_questions = set()
            for q in preprocessed_questions:
                uncategorized_questions.add(' '.join(q))
        
            print("UNCATEGORIZED: ",len(uncategorized_questions))
            print("NEXT CLUSTER KEYWORD",next_cluster_keyword)
            
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
        
        print(len(cluster_maps))
        return cluster_maps