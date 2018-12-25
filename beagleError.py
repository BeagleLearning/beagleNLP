from flask import jsonify
import errors

class BeagleError(Exception):
    def __init__(self, errorCode, details):
        Exception.__init__(self)
        self.message = errors.details[errorCode]["message"]
        self.statusCode = errors.details[errorCode]["statusCode"]
        self.details = details

    def toDict(self):
        retVal = dict(self.details or {})
        retVal.message = self.message
        return retVal
