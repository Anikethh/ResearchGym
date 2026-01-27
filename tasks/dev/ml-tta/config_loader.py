import json
import copy
import os

class Dict2Class(object): 
    def __init__(self, my_dict : dict): 
        for key in my_dict: 
            setattr(self, key, my_dict[key]) 
    def add_key(self, added_dict : dict):
        for key in added_dict:
            if hasattr(self, key):
                self.key = added_dict[key]
            else:
                setattr(self, key, added_dict[key])

class Dict2Config(object): 
    def __init__(self, my_dict : dict): 
        self.set_attr_at_branch(self, my_dict)
                
    def set_attr_at_branch(self, branch, my_dict : dict):
        for key in my_dict: 
            if isinstance(my_dict[key], dict):
                setattr(branch, key, Dict2Class({}))
                self.set_attr_at_branch(branch.__dict__[key], my_dict[key])
            else:
                setattr(branch, key, my_dict[key])
            
    def add_key_at_branch(self, branch, added_dict : dict):
        for key in added_dict:
            if hasattr(branch, key):
                if isinstance(added_dict[key], dict):
                    self.add_key_at_branch(branch.__dict__[key], added_dict[key])
                else:
                    branch.__dict__[key] = added_dict[key]
            else:
                if isinstance(added_dict[key], dict):
                    setattr(branch, key, Dict2Class({}))
                    self.set_attr_at_branch(branch.__dict__[key], added_dict[key])
                else:
                    setattr(self, key, added_dict[key])
    
    def add_key(self, added_dict : dict):
        self.add_key_at_branch(self, added_dict)
            
def load_config_from_path(path):
    if not os.path.exists(path):
        raise ValueError("Configuration path not exist!")
    with open(path, "r") as f:
        config_dict = json.load(f)
    config = Dict2Config(config_dict)
    return config

def load_extra_config_from_path(origin_config, path):
    if not isinstance(origin_config, Dict2Config):
        config = vars(origin_config)
        config = Dict2Config(config)
    else:
        config = copy.deepcopy(origin_config)
        
    
    if not os.path.exists(path):
        raise ValueError("Configuration path not exist!")
    with open(path, "r") as f:
        config_dict = json.load(f)
        
    config.add_key(config_dict)
    return config

def load_extra_config_from_dict(origin_config, added_dict):
    if not isinstance(origin_config, Dict2Config):
        config = vars(origin_config)
        config = Dict2Config(config)
    else:
        config = copy.deepcopy(origin_config)
    config_dict = added_dict
    config.add_key(config_dict)
    return config

# Test
if __name__ == "__main__":
    base_file = "../configs/base.json"
    config_file = "../configs/Caption_distill_double_config.json"
    
    base_config = load_config_from_path(base_file)
    config = load_extra_config_from_path(base_config, config_file)
    
    