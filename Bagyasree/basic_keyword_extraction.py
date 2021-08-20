from helper_functions import preprocess
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.stem import WordNetLemmatizer

class BasicKeywordExtraction:
    def extract_keywords_basic(self, questions):
        formatted_questions = [ques.replace('-',' ') for ques in questions]
        preprocessed_questions = preprocess(formatted_questions, tokenize = True)
        tagged_questions = {}
        for i in range(0, len(questions)):
            tagged_questions[questions[i]] = list(set(preprocessed_questions[i]))
        return tagged_questions

    def extract_keywords_pos(self, questions, pos = []):
        lemmatizer = WordNetLemmatizer()
        tagged_questions = {}
        if pos:
            nlp = spacy.load("en_core_web_sm")
            for ques in questions:
                doc = nlp(ques)
                tags = list(set([lemmatizer.lemmatize(token.text) for token in doc if token.pos_ in pos]))
                # tags = [lemmatizer.lemmatize(tag) for tag in tags]
                tagged_questions[ques] = tags

        return tagged_questions
    
    
    def extract_keywords_tfidf(self, questions):
        # print(questions)
        preprocessed_questions = preprocess(questions, tokenize = False)
        # all_keywords = list(set([word for question in preprocessed_questions for word in question]))
        # print(all_keywords)
        vectorizer = TfidfVectorizer(ngram_range=(1,2))

        # doc_term_matrix = vectorizer.fit_transform(questions)

        # for term in doc_term_matrix:
        #     print(type(term))
        #     print(len(term))
        doc_term_matrix = vectorizer.fit_transform(preprocessed_questions)
        print(doc_term_matrix)
        all_terms = vectorizer.get_feature_names()
        doc_term_array = doc_term_matrix.toarray()
        
        tagged_questions = {}

        threshold = 0.4
        for i in range(0, len(questions)):
            doc = doc_term_array[i]
            tags = [all_terms[term_index] for term_index in range(0,len(doc)) if doc[term_index] >= threshold]
            tagged_questions[questions[i]] = tags
        
        return tagged_questions