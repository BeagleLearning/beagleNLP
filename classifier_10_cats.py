import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from text_preprocessing import remove_special_characters


model_location = './resources/10_cats_deberta/torch_hf_deberta_epoch_5.model'



def run_prediction(question: str, device, tokenizer, model) -> int:
    """
    Takes a question in string format.
    Returns category index for the given string.
    TODO: cumbersome double-level passing of arguments
    """

    # if a sentence is not a string, return nothing
    # the wrapping function will be running in a loop, hence interrupting or raising an error 
    # for a single sentence among possibly hundreds or thousands is not desirable
    if type(question) is not str:
        return None

    # remove special characters from the string that might affect the quality of the prediction
    question_without_special_characters = remove_special_characters(question)
    
    # encode the string with the corresponding tokenizer
    encoded_question = tokenizer.encode_plus(
        question_without_special_characters,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        max_length = 256,
        truncation=True,
        return_tensors='pt')
    
    # pass the string to the corresponding torch mount and reformat as the deBERTa input
    encoded_question = encoded_question.to(device)

    inputs = {'input_ids':      encoded_question['input_ids'],
          'attention_mask': encoded_question['attention_mask'],
          }
    # run the prediction for the given string
    with torch.no_grad():
        output = model(**inputs).logits.detach().cpu().numpy().flatten()

    # prediction is an array of logits, we only need the index of the highest value
    # the values start from 0 index, hence add 1 to the result
    result = int(np.argmax(output)+1) # IMPORTANT - converting to int from numpy type value
    
    # and return the result
    return result



def get_predictions(questions: list, device, tokenizer, model) -> list:
    """
    Gets a list of question dictionaries in format id, content
    Return a list of dictionaries in format id, prediction
    TODO: return a category index or a name?
    TODO: check duplicate ids?
    TODO: solve the mounting differently?
    """

    # check if the input is a list and if not, return an alert
    if type(questions) is not list:
        return "Invalid input. A list of dictionaries expected."
    
    #check if the list has any contents and if not, return an alert
    if len(questions) == 0:
        return "Invalid input. Received an empty list."
    
    # initiate the final output list
    result_list = []
    
    # check if the dictionaries in the input list have the right formatting before starting predictions
    for question_dict in questions:
        if (type(question_dict) is dict) & ('id' not in question_dict or 'content' not in question_dict):
            return "Invalid formatting detected. Make sure that every element is a dictionary containing 'id' and 'content' keys."

   
    for question_dict in questions:
        # if the current index is not a dict, append a null value, might be an exception
        if type(question_dict) is not dict:
            result_list.append(None)
            continue
        # initiate the dictionary for the current question
        single_question_result_dict = {}
        single_question_result_dict['id'] = question_dict['id']
        # pass the content to the prediction function
        single_question_result_dict['type'] = run_prediction(question_dict['content'], device = device,\
            tokenizer = tokenizer, model=model)
        # append the dictionary to the list of results
        result_list.append(single_question_result_dict)
    
    return result_list