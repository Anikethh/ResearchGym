import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process experiment results and aggregate metrics.')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name or path of the model to use.'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='',
        help='A name for the experiment. Default is "_final".'
    )

    parser.add_argument(
        '--top_value',
        type=int,
        default=100,
        help='An integer representing the top value parameter. Default is 100.'
    )

    return parser.parse_args()

# Define a function to calculate standard error
def standard_error(x):
    if len(x) != 5:
        print("Please check")
    return np.std(x, ddof=1) / np.sqrt(len(x))

def main():
    # Parse command-line arguments
    args = parse_arguments()

    model = args.model
    data = args.data
    experiment_name = args.experiment_name
    top_value = args.top_value

    # Define the base file name pattern
    file_pattern = "{model}_{data}_{cv}_{top}_results{experiment_name}.csv"

    # Assuming you have 5 folds (CV = 0 to 4) and a specific topk value (e.g., 0.2)
    cv_folds = 5

    # Define the columns in the CSV file
    columns = [
        "Seed", "CV", "Baseline", "Topk", "Explainer", "Lambda_1", "Lambda_2", "Lambda_3",
        "top50_cum", f"top{top_value}_cum", "AUCC", "Accuracy", "Comprehensiveness", "CrossEntropy", "LogOdds", "Sufficiency"
    ]

    # Initialize a dictionary to store dataframes for each CV fold
    dataframes = []


    # Loop through each CV fold and read the corresponding CSV file
    for cv in range(cv_folds):
        file_name = file_name = file_pattern.format(
            model=model,
            data=data,
            cv=cv,
            top=top_value,
            experiment_name=experiment_name
        )
        if not Path(file_name).is_file():
            print(f"Warning: File {file_name} does not exist. Skipping this fold.")
            continue
        df = pd.read_csv(file_name, header=None, names=columns)
        dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dataframes)

    # Group by the relevant columns (excluding Seed, CV, and Lambda columns)
    grouped = combined_df.groupby(["Baseline", "Topk", "Explainer"])



    # Calculate mean and standard error across the CV folds
    result = grouped.agg(
        mean_Accuracy=("Accuracy", np.mean),
        std_error_Accuracy=("Accuracy", standard_error),
        mean_CrossEntropy=("CrossEntropy", np.mean),
        std_error_CrossEntropy=("CrossEntropy", standard_error),
        mean_Sufficiency=("Sufficiency", np.mean),
        std_error_Sufficiency=("Sufficiency", standard_error),
        mean_Comprehensiveness=("Comprehensiveness", np.mean),
        std_error_Comprehensiveness=("Comprehensiveness", standard_error),
        mean_top50=("top50_cum", np.mean),
        std_error_top50=("top50_cum", standard_error),
        mean_topK=(f"top{top_value}_cum", np.mean),
        std_error_topK=(f"top{top_value}_cum", standard_error),
        mean_AUCC=("AUCC", np.mean),
        std_error_AUCC=("AUCC", standard_error),
        mean_LogOdds=("LogOdds", np.mean),
        std_error_LogOdds=("LogOdds", standard_error),
    )

    for col in result.columns:
        if "Comprehensiveness" in col or "Sufficiency" in col:
            result[col] = result[col] * 100

    # for mimic, only when my code is wrong...
    # for col in result.columns:
    #     if "top" in col:
    #         result[col] = result[col] / 2

    result = result.applymap(lambda x: f"{x:.3f}")

    # Define the output directory and file path
    output_dir = Path(f"./results/{data}")
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_file = output_dir / f"{model}_{top_value}{experiment_name}.csv"

    # Save the result to the CSV file
    result.to_csv(output_file)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()