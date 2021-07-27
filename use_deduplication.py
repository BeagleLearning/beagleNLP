import collections
from beagleError import BeagleError
import errors
import numpy as np
from text_preprocessing import remove_special_characters

def deduplicate(questions: list, embedder, threshold=0.7) -> list:
    """
    Accepts a list of dictionaries.
    Each dictionary is expected to contain the following keys and data types:
    {'id': integer, 'content': string}

    Returns a list of lists, where each list represent a group of questions with
    high mutual similarity scores.

    """
    # check if the input is a list and if not, raise an error
    if type(questions) is not list:
        raise BeagleError(errors.INVALID_INPUT_NOT_A_LIST)
    
    # check if the list has any contents and if not, raise an error
    if len(questions) == 0:
        raise BeagleError(errors.INVALID_INPUT_EMPTY_LIST)
        
    # need at least two questions to find duplicates in the set
    if len(questions) < 2:
        raise BeagleError(errors.TOO_FEW_QUESTIONS_TO_DEDUPLICATE)
    
    for question_dict in questions:
        # check that each element of the input list is indeed a dictionary
        if type(question_dict) is not dict:
            raise BeagleError(errors.INVALID_FORMATTING_ERROR)
        # check that each element is properly formatted
        if ('id' not in question_dict) or ('content' not in question_dict):
            raise BeagleError(errors.INVALID_FORMATTING_ERROR)
        # check that the id and/or content are not empty
        if (question_dict['id'] is None ) or (question_dict['content'] is None):
            raise BeagleError(errors.EMPTY_VALUE_ERROR)
        # make sure that the id is an integer and the content is a string
        if  (type(question_dict['id']) is not int) or (type(question_dict['content']) is not str):
            raise BeagleError(errors.UNEXPECTED_DATA_TYPE_ERROR)
    # remove special characters and place all questions to one list
    questions_without_special_characters = [remove_special_characters(question['content']) for question in questions]
    # get the embeddings, the embedder accepts a single string or a list of strings and returns a list of lists
    embeddings = embedder(questions_without_special_characters)
    # use numpy to get the dot product matrix
    similarities = np.inner(embeddings, embeddings)
    # initiate the result dictionary with keys 'id' (original ID) and a corresponding 'unique_id, which is 0 initially'
    result_dict = dict()
    unique_id = 0
    # iterate through the lists by indices
    for array_ix in range(len(similarities)):
        # for each sublist of similarity scores, check whether the index of the element is the same
        # as the sublist index in the 'similarities' list. If so, end the loop, since the loop reached
        # the diagonal of 1s in the matrix and we only need to check for similarity once per each pair
        for similarity_score_ix in range(len(similarities[array_ix])):
            if similarity_score_ix == array_ix:
                break
            # if a similarity score higher than the threshold is detected, then pair the question
            # to the corresponding previous similar question by its unique_id
            if similarities[array_ix][similarity_score_ix] > threshold:
                result_dict[array_ix] = result_dict[similarity_score_ix]
        # if the corresponding index does not have any unique_id yet, assign a unique_id and increase
        # the unique_id variable by 1 
        if result_dict.get(array_ix) is None:
            result_dict[array_ix] = unique_id
            unique_id+=1
    # get a list of questions with the same unique_id per unique_id
    # and return a list of lists, where each list are similar questions' ids put together
    similar_questions_grouped_by_unique_id = collections.defaultdict(list)
    for question_id, unique_id in result_dict.items():
        similar_questions_grouped_by_unique_id[unique_id].append(question_id)

    return [similar_questions_list for similar_questions_list in similar_questions_grouped_by_unique_id.values()]