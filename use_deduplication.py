import collections
from beagleError import BeagleError
import errors
import numpy as np
from text_preprocessing import remove_special_characters
import tensorflow_hub as hub

##### UNIVERSAL SENTENCE ENCODER INITIATION #####
use_link = "https://tfhub.dev/google/universal-sentence-encoder/4"
# make hub download the model instead of looking for the cached version
force_download_upon_reset = "?tf-hub-format=compressed"
try:
    embedder = hub.load(use_link+force_download_upon_reset)
except:
    raise BeagleError(errors.USE_LOAD_ERROR) #If model not loaded
##### ##### ##### #####


def check_right_dict_formatting(question_dict: dict) -> None:
    """
    Responsible for checking whether each element representing a question
    has the right contents.
    """
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



def group_duplicates(questions: list, threshold=0.7) -> list:
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

    # need at least two questions to find duplicates in the set
    if len(questions) < 2:
        raise BeagleError(errors.TOO_FEW_QUESTIONS_TO_DEDUPLICATE)

    # check the right formatting of each dictionary
    for question_dict in questions:
        check_right_dict_formatting(question_dict)

    # remove special characters and place all questions to one list
    questions_without_special_characters = [remove_special_characters(question['content']) for question in questions]
    # get the embeddings, the embedder accepts a single string or a list of strings and returns a list of lists
    embeddings = embedder(questions_without_special_characters)
    # use numpy to get the dot product matrix
    # the USE embeddings are normalized, meaning that the dot product of each vector is a value
    # between 0 and 1, 1 meaning "the same" and 0 meaning "no similarities at all"
    similarities = np.inner(embeddings, embeddings)
    # initiate the result dictionary with keys 'id' (original ID) and a corresponding 'unique_id, which is 0 initially'
    result_dict = dict()
    unique_id = 0
    # iterate through the lists by indices
    for array_ix in range(len(similarities)):
        # for each sublist of similarity scores, check whether the index of the element is the same
        # as the sublist index in the 'similarities' list. If so, end the loop, since the loop reached
        # the diagonal of 1s in the matrix and we only need to check for similarity once per each pair
        for similarity_score_ix in range(array_ix):
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


def find_duplicates_one_to_many(target_question: dict, questions_to_compare: list, threshold=0.7) -> list:
    """
    Accepts a target question and a list of dictionaries representing the questions to compare.
    Each dictionary is expected to contain the following keys and data types:
    {'id': integer, 'content': string}

    Returns a list of ids of the questions similar to the target question
    according to the given threshold.

    """
    # check the right format of the target question
    check_right_dict_formatting(target_question)
    if len(target_question['content'].split()) == 0:
        raise BeagleError(errors.INVALID_QUESTION_EMPTY_STRING_ERROR)

    # check if the questions_to_compare is a list and if not, return an alert
    if type(questions_to_compare) is not list:
        raise BeagleError(errors.INVALID_INPUT_NOT_A_LIST)

    # check if the questions_to_compare list has any contents and if not, return an alert
    if len(questions_to_compare) == 0:
        raise BeagleError(errors.INVALID_INPUT_EMPTY_LIST)

    # perform the formatting check again for each element in the questions_to_compare list
    for question_dict in questions_to_compare:
        check_right_dict_formatting(question_dict)

    # remove special characters from the target question
    single_question_without_special_characters = remove_special_characters(target_question['content'])
    # get the target question embedding
    single_question_embedding = embedder([single_question_without_special_characters])
    # initialize list with found high scored ids
    similar_ids = []
    # loop through the questions and compare them to the target question
    for question_dict in questions_to_compare:
        question_without_special_characters = remove_special_characters(question_dict['content'])
        question_embedding = embedder([question_without_special_characters])
        sim_score = np.inner(single_question_embedding, question_embedding)
        if sim_score >= threshold:
            similar_ids.append(question_dict['id'])
    
    return similar_ids