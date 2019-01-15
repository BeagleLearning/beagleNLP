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
        are two or more.",
    "status_code": 400
}
