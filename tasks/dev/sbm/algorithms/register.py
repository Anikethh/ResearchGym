
algorithm_dict = {}
def register_algorithm(name):
    def decorator(cls):
        if type(name) == list:
            for n in name:
                algorithm_dict[n] = cls
        else:
            algorithm_dict[name] = cls
        return cls
    return decorator
