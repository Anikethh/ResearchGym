

dataset_dict = {}
def register_dataset(name):
    def decorator(cls):
        if type(name) == list:
            for n in name:
                dataset_dict[n] = cls
        else:
            dataset_dict[name] = cls
        return cls
    return decorator

transform_funcs = {}
def register_transform(name):
    def decorator(cls):
        if type(name) == list:
            for n in name:
                transform_funcs[n] = cls
        else:
            transform_funcs[name] = cls
        return cls
    return decorator
    
