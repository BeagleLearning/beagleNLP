import collections
import time
import numpy as np
from text_preprocessing import remove_special_characters

def deduplicate(questions: list, embedder, threshold=0.7) -> dict:
    """
    Accepts a list of dictionaries of format (id, content)
    Returns a dictionary with sentence ID and a corresponding unique ID
    """
    # check if the input is a list and if not, return an alert
    if type(questions) is not list:
        return "Invalid input. A list of dictionaries expected."
    
    # check if the list has any contents and if not, return an alert
    if len(questions) == 0:
        return "Invalid input. Received an empty list."

    
    for question_dict in questions:
        # check that each element of the input list is indeed a dictionary
        if type(question_dict) is not dict:
            return "Invalid formatting detected. Make sure that every element is a dictionary containing 'id' and 'content' keys."
        # check that each element is properly formatted
        if (type(question_dict) is dict) & ('id' not in question_dict or 'content' not in question_dict):
            return "Invalid formatting detected. Make sure that every element is a dictionary containing 'id' and 'content' keys."
        # check that the id and content are not empty
        if (question_dict['id'] is None ) | (question_dict['content'] is None):
            return "Empty id or content detected. Make sure that every element is a dictionary containing 'id' and 'content' keys."
    # remove special characters and keep the ids, the mapping later will be performed on CONTENT, not ID
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