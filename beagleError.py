import errors


class BeagleError(Exception):
    def __init__(self, error_code, details={}):
        Exception.__init__(self)
        self.message = errors.details[error_code]["message"]
        self.status_code = errors.details[error_code]["status_code"]
        self.code = error_code
        self.details = details

    def to_dict(self):
        retVal = dict(self.details)
        retVal["message"] = self.message
        retVal["code"] = self.code
        return retVal
