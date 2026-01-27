import os
from argparse import ArgumentParser
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--problem_name', type=str, default='oscillator1')
args = parser.parse_args()


if __name__ == '__main__':
    problem_name = args.problem_name
    data_dir = os.path.join('./data', problem_name)
    train_path = os.path.join(data_dir, 'train.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Expected dataset at {train_path}")

    df = pd.read_csv(train_path)
    print(f"Loaded dataset '{problem_name}': {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", ', '.join(df.columns))
    print("This is a skeleton repository. Implement your own training/evaluation pipeline here.")
