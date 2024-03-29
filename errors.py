details = {}

NO_TOKEN_FOUND = "NO_TOKEN_FOUND"
details[NO_TOKEN_FOUND] = {
    "message": "No word token found in url.",
    "status_code": 404
}

NO_VECTORS_FOUND = "NO_VECTORS_FOUND"
details[NO_VECTORS_FOUND] = {
    "message": "No vectors were found for the words in these documents.",
    "status_code": 404
}

MISSING_PARAMETERS_FOR_ROUTE = "MISSING_PARAMETERS_FOR_ROUTE"
details[MISSING_PARAMETERS_FOR_ROUTE] = {
    "message": "Missing parameters for this route",
    "status_code": 404
}

TOO_FEW_QUESTIONS = 2801
details[TOO_FEW_QUESTIONS] = {
    "message": "There are too few questions to cluster. Try again when there \
      are six or more.",
    "status_code": 400
}

NO_WORD_DATA = 2802
details[NO_WORD_DATA] = {
    "message": "We couldn't categorize these questions because we don't \
    recognize the words in them.",
    "status_code": 400
}

UNKNOWN_KEYWORDS = 2803
details[UNKNOWN_KEYWORDS] = {
    "message": "Sorry, we don't know the words you're trying to cluster around.",
    "status_code": 400
}

USE_EMBED_ERROR = 2804
details[USE_EMBED_ERROR] = {
    "message": "Sorry, our USE embeddings failed on the data you sent. kindly check if you sent some garbage values",
    "status_code": 400
}

KEY_ERROR = 2805
details[KEY_ERROR] = {
    "message": "Looks like you forgot or gave an invalid key in one or more of your key-pairs in the data. Kindly check",
    "status_code": 400
}

USE_LOAD_ERROR = 2806
details[USE_LOAD_ERROR] = {
    "message": "USE Model was unable to load for some reason",
    "status_code": 500
}

INVALID_DATA_TYPE = 2807
details[INVALID_DATA_TYPE] = {
    "message": "Invalid Datatype in one or more fields in the data",
    "status_code": 400
}

HAC_SPARS_FUN_ERROR = 2808
details[HAC_SPARS_FUN_ERROR] = {
    "message": "The HAC Sparsification Function couldn't process the data for some reason",
    "status_code": 500
}

HAC_FUN_ERROR = 2809
details[HAC_FUN_ERROR] = {
    "message": "The HAC Function couldn't process the data for some reason",
    "status_code": 500
}


#404: Not Found
#400: Bad Request
#500: Internal Server Error
