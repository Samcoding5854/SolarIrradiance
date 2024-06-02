import pandas as pd
import random

# Read the CSV file into a DataFrame
input_file = 'SolarPrediction.csv'  # Replace with your actual file name
df = pd.read_csv(input_file)

# Ensure the file has at least 30,000 rows
if len(df) < 30000:
    raise ValueError("The input file must contain at least 30,000 rows")
n = 200
# Randomly sample 50 rows
sampled_df = df.sample(n=n, random_state=1)  # random_state is used for reproducibility

# Save the sampled rows to a new CSV file
output_file = 'output_file.csv'  # Replace with your desired output file name
sampled_df.to_csv(output_file, index=False)

print(f"{n} random rows have been saved to {output_file}")
