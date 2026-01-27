import torch
import os
from utils import *
from config import get_args
import models
import data
import algorithms

from algorithms.utils import init_algorithm
if __name__ == '__main__':
    
    args = get_args()

    save_folder = prepare_experiment(args)

    algorithm = init_algorithm(args)

    if args.mode == "train" or args.mode == "debug":
        algorithm.train(save_folder, args.train_split)

    algorithm.test(save_folder, args.test_split, args.result_path)
