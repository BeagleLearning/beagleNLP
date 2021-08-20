from basic_keyword_extraction import BasicKeywordExtraction
import wikipedia
import numpy as np
from helper_functions import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from wordfreq import word_frequency
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec, KeyedVectors
import io
import pickle
from scipy.spatial.distance import cosine
import re
import tensorflow_hub as hub

class WikipediaFunctions:

    def load_vectors(self, fname):
        print("in load vectors")
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            print("-",end = '')
        return data

    def tag_questions_using_summaries(self, questions, links, embed):
        ambiguous = 0
        question_tag_dict = {}
        question_vectors = embed(questions)
        threshold = 0.15
        for link in links:
            try:
                summary = wikipedia.summary(link)
                summary_vector = embed([summary])
                for i in range(0, len(questions)):
                    if questions[i] not in question_tag_dict:
                        question_tag_dict[questions[i]] = []
                
                    similarity = 1 - cosine(question_vectors[i],summary_vector)
                    if similarity >= threshold:
                        question_tag_dict[questions[i]].append(link)
            except:
                ambiguous += 1
        print("AMBIGUOUS: ", ambiguous)
        return question_tag_dict
    
    def tag_question_using_titles(self, questions, links, embed):
        question_vectors = embed(questions)
        question_tag_dict = {}
        threshold = 0.3
        for title in links:
            title_vector = embed([title])
            for i in range(0, len(questions)):
                    if questions[i] not in question_tag_dict:
                        question_tag_dict[questions[i]] = []
                
                    similarity = 1 - cosine(question_vectors[i],title_vector)
                    if similarity >= threshold:
                        question_tag_dict[questions[i]].append(title)
        
        return question_tag_dict


    def method_3(self, questions):
        BKE = BasicKeywordExtraction()
        keyword_set = BKE.extract_keywords_tfidf(questions)
        for question in keyword_set:
            print("QUESTION: ",question)
            print()
            for keyword in keyword_set[question]:
                print("Tag:",keyword)
                try:
                    summary = wikipedia.summary(keyword)
                    print(summary)
                except:
                    pass
                print()
            print()
            print()
            print()
            print()
        return {}

    def method_2(self, questions):

        #USING TF-IDF
        # preprocessed_questions = preprocess(questions, tokenize = False)
        # vectorizer = TfidfVectorizer(ngram_range=(1,2))

        # doc_term_matrix = vectorizer.fit_transform(preprocessed_questions)
        # # print(doc_term_matrix)
        # all_terms = vectorizer.get_feature_names()
        # doc_term_array = doc_term_matrix.toarray()
        
        # tagged_questions = {}

        # threshold = 0.4
        # for i in range(0, len(questions)):
        #     doc = doc_term_array[i]
        #     top_keywords = (-doc).argsort()[:5]
        #     print(questions[i], ": ", end = '')
        #     for keyword in top_keywords: 
        #         print(all_terms[keyword], end = ', ')
        #     print()

        #USING POS
        BKE = BasicKeywordExtraction()
        keyword_set = BKE.extract_keywords_pos(questions, ['NOUN','PROPN'])
        for question in keyword_set:
            print(question,": ",keyword_set[question])
            # print()
            # for keyword in keyword_set[question]:
            #     print("Tag:",keyword)

        return {}

    def method_1(self, questions):
        preprocessed_questions = preprocess(questions, tokenize = False)
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        doc_term_matrix = vectorizer.fit_transform(preprocessed_questions)
        all_terms = vectorizer.get_feature_names()
        doc_term_array = doc_term_matrix.toarray()
        
        term_scores = [0 for i in range(0,len(all_terms))]

        # fetching the top scoring terms by adding the scores received by the term for all questions. tfi-idf. 
        for i in range(0, len(questions)):
            doc = doc_term_array[i]
            for term_index in range(0, len(doc)):
                term_scores[term_index] += doc[term_index]
        
        term_scores = np.array(term_scores)
        top_n = 30
        top_terms = (-term_scores).argsort()[:top_n]
        # for term in top_terms:
        #     print(all_terms[term])
        # print()

        #filtering by frequency in english
        invalid_count = 0
        freqs = []
        for term in top_terms:
            try:
               freqs.append(word_frequency(all_terms[term],"en"))
            except:
                invalid_count += 1

        final_n = 10
        freqs = np.array(freqs)
        top = (freqs).argsort()[:final_n]
        roots = []
        for term in top:
            roots.append(all_terms[top_terms[term]])
        
        print("obtained roots.")

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
        print("loaded vector model.")
        
        similarity_threshold = 0.72
        final_roots = []
        for root in roots:
            similarities = []
            suggestions = wikipedia.search(root)
            root_vector = embed([root])[0]
            vectors = embed(suggestions)
            max_similarity = 0
            top_suggestion = ''
            for i in range(0, len(suggestions)):
                similarity = 1 - cosine(root_vector, vectors[i])
                similarities.append(similarity)
                
            top_similarities = (-np.array(similarities)).argsort()[:3]
            top_suggestions = [suggestions[index] for index in top_similarities if similarities[index] > similarity_threshold]
            final_roots.extend(top_suggestions)
        
        final_roots = list(set(final_roots))
        print("ALL ROOTS: ",final_roots)

        links = []
        ambiguous = 0
        non_ambiguous_roots = []
    

        for root in final_roots:  
            try:
                page = wikipedia.page(root)
                links.extend(page.links)
                non_ambiguous_roots.append(page.title)
            except: 
                ambiguous += 1
        
        unique_links = list(set(links))
        useful_links = []
        proportion = 0.5
        for link in unique_links: 
            if(links.count(link) >= proportion*final_n):
                useful_links.append(link)
        
        useful_links.extend(non_ambiguous_roots)
        # print("ROOTS:",final_roots)
        print("USEFUL LINKS:",useful_links)
        # print("NO. OF USEFUL LINKS:",len(useful_links))
        # print("NO. OF AMBIGUOUS LINKS:",ambiguous)
        # print()

        # tagged_questions = self.tag_questions(questions, useful_links, embed)
        # tagged_questions = self.tag_questions_using_summaries(questions, final_roots, embed)
        tagged_questions = self.tag_question_using_titles(questions, useful_links, embed)

        # for ques in tagged_questions:
        #     print(ques)
        #     print(tagged_questions[ques])
        #     print()
        
        print()
        print()
        print()
        print()
        return {}