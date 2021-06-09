from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from numpy import dot
from numpy.linalg import norm
import numpy as np

class LatentSemanticAnalyis:
    def find_svd(self, group, vectorizer):
    
        matrix = vectorizer.fit_transform(group)
        svd = TruncatedSVD(n_components = 2)
        lsa = svd.fit_transform(matrix)
        
        lsa = lsa.tolist()
        topic1 = []
        topic2 = []
        for doc in lsa:
            if doc[0] > doc[1]:
                topic1.append(group[lsa.index(doc)])
            else:
                topic2.append(group[lsa.index(doc)])

        return topic1, topic2


    def manual_controller(self, docs, vectorizer):
        cont = 1
        groups = [docs]
        final_groups = []
        while(len(groups)):
            group = groups[0]
            group1, group2 = self.find_svd(group, vectorizer)
            print("1:")
            print(len(group1))
            print("2:")
            print(len(group2))
            
            inp = int(input("0: Decompose none     1: Decompose 1     2: Decompose 2     3: Decompose both.     Enter number: "))
            if inp == 0:
                final_groups.append(group1)
                final_groups.append(group2)
            elif inp == 1:
                groups.append(group1)
                final_groups.append(group2)
            elif inp == 2:
                groups.append(group2)
                final_groups.append(group1)
            else:
                groups.append(group1)
                groups.append(group2)
        
            groups.pop(0)
        
        return final_groups
    

    def automatic_controller(self, docs, docs_per_topic, vectorizer):
        groups = [docs]
        final_groups = []
        while(len(groups)):
            #print("while")
            group = groups[0]
            group1, group2 = self.find_svd(group, vectorizer)
            
            len1 = len(group1)
            len2 = len(group2)
            
            if (len1 == 0 or len2 == 0) or (len1<=docs_per_topic and len2<=docs_per_topic):
                final_groups.append(group1)
                final_groups.append(group2)
            
            elif len1<=docs_per_topic and len2>docs_per_topic:
                final_groups.append(group1)
                groups.append(group2)
            
            elif len1>docs_per_topic and len2<=docs_per_topic:
                groups.append(group1)
                final_groups.append(group2)
            
            elif len1>docs_per_topic and len2>docs_per_topic:
                groups.append(group1)
                groups.append(group2)
            
            else:
                print(len1, len2)
        
            groups.pop(0)
        
        return final_groups

    def avg_similarity(self, group):
        vect = CountVectorizer(group, binary = True)
        matrix = vect.fit_transform(group)
        one_hot_vectors = matrix.toarray()
        length = len(one_hot_vectors)
        similarities = []
        for i in range(0,length):
            for j in range(i+1, length):
                cos_sim = dot(one_hot_vectors[i], one_hot_vectors[j])/(norm(one_hot_vectors[i])*norm(one_hot_vectors[j]))
                #print(cos_sim)
                similarities.append(cos_sim)
                
        ret = np.mean(similarities)
        return ret

    def lsa(self, questions):
        ngram_vect = TfidfVectorizer(ngram_range = (0,3))
        groups = self.automatic_controller(questions, 120, ngram_vect)
    
        clusters = []
        for ques in questions:
            flag = 0
            for i in range(0,len(groups)):
                if ques in groups[i]:
                    clusters.append(i)
                    flag = 1
                    break
            if flag == 0: 
                clusters.append(len(groups)) #for some reason, some questions are seemingly not part of any group. putting them all into the last cluster here.
        
        return clusters