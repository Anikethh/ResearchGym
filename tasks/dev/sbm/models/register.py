

model_dict = {}
def register_model(name):
    def decorator(cls):
        if type(name) == list:
            for n in name:
                model_dict[n] = cls
        else:
            model_dict[name] = cls
        return cls
    return decorator
