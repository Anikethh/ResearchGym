import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

model = "state"
data = "mimic3"
experiment_name = ""
batch_size=1

# Define the base file name pattern
file_pattern = "runtime_{model}_{data}_{cv}_{top}_runtime_batch_size{batch_size}.csv"

# Assuming you have 5 folds (CV = 0 to 4) and a specific topk value (e.g., 0.2)
cv_folds = 5
top_value = 100

# Define the columns in the CSV file
columns = [
    "Seed", "CV", "Baseline", "Topk", "Explainer", "runtime(s)"
]

# Initialize a dictionary to store dataframes for each CV fold
dataframes = []


# Loop through each CV fold and read the corresponding CSV file
for cv in range(cv_folds):
    file_name = file_pattern.format(model=model, data=data,cv=cv, top=top_value, batch_size=batch_size)
    df = pd.read_csv(file_name, header=None, names=columns)
    dataframes.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# Group by the relevant columns (excluding Seed, CV, and Lambda columns)
grouped = combined_df.groupby(["Baseline", "Topk", "Explainer"])

# Define a function to calculate standard error
def standard_error(x):
    if len(x) != 5:
        print(x)
    return np.std(x, ddof=1) / np.sqrt(len(x))

# Calculate mean and standard error across the CV folds
result = grouped.agg(
    mean_runtime=("runtime(s)", np.mean),
    std_error_runtime=("runtime(s)", standard_error),
)

result = result.applymap(lambda x: f"{x:.3f}")

# Define the output directory and file path
output_dir = Path(f"./results/{data}")
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
output_file = output_dir / f"{model}_{batch_size}_runtime.csv"

# Save the result to the CSV file
result.to_csv(output_file)

print(f"Results saved to {output_file}")