from helper_functions import get_questions, display_tags
from basic_keyword_extraction import BasicKeywordExtraction
from Testing.testing import Test
from wikipedia_keywords import WikipediaFunctions

test_class = Test()

def basic_keyword_extraction(test = True, display = False):
    BKE = BasicKeywordExtraction()
    if test == False:
        questions = get_questions('normal', tokenized = False)
        # tagged_questions = BKE.extract_keywords_basic(questions)
        # tagged_questions = BKE.extract_keywords_pos(questions, ['PROPN','NOUN','ADJ'])
        tagged_questions = BKE.extract_keywords_tfidf(questions)
        if display:
            display_tags(tagged_questions)
    else:
        # test_class.test_tagging_algorithm(BKE.extract_keywords_basic, display_tags = display)
        # test_class.test_tagging_algorithm(BKE.extract_keywords_pos, arguments = [['NOUN','PROPN','ADJ']], display_tags = display)
        test_class.test_tagging_algorithm(BKE.extract_keywords_tfidf, display_tags = display)

def wikipedia_functions(test = True, display = False):
    WKE = WikipediaFunctions()
    if test == False:
        questions = get_questions('normal',tokenized = False)
        tagged_questions = WKE.method_3(questions)
        if display:
            display_tags(tagged_questions)
    else:
        test_class.test_tagging_algorithm(WKE.method_1, display_tags = display)

# basic_keyword_extraction(test = True, display = True)
wikipedia_functions(test = True, display = False)