import pandas as pd
import numpy as np

# Load the dataframe from your CSV file
# df = pd.read_csv("1. update_simulation_logs.csv")
df = pd.read_csv("./data/expanded_update_simulation_logs.csv")

# --- Function to create the input prompt ---
def create_input_prompt(row):
    """
    Converts a single row of the DataFrame into a descriptive text prompt
    for an LLM.
    """
    return (
        f"A simulation log details an update at time {row['t']} for node {row['node']} with a '{row['role']}' role. "
        f"The update traveled via the {row['channel']} channel, had a path length of {row['path_len']}, "
        f"and was created by {row['creator']} (version {row['ver']}). "
        f"The baseline system acceptance was {row['baseline_accept']}, while an LLM scored it {row['llm_score']} "
        f"and accepted it with status {row['llm_accept']}. "
        f"Authenticity was {row['authentic']}, rollforward was {row['rollforward']}, and chain integrity was {row['chain_ok']}."
    )

# --- Generate input prompts and output labels for all rows ---

# Apply the function to each row to create a Series of input prompts
input_prompts = df.apply(create_input_prompt, axis=1)

# Extract the 'is_malicious' column as the output labels
output_labels = df['is_malicious']

# --- Verify the total number of rows processed ---
total_rows = len(df)
print(f"Total number of rows processed: {total_rows}")

#save 
processed_df = pd.DataFrame({'input_prompt': input_prompts, 'output_label': output_labels.astype(int)})
processed_df.to_csv("./data/expanded_processed_data.csv", index=False)
print("Saved to 'expanded_processed_data.csv'")


# Split the processed_df into training, validation, and test sets
train_df, val_df, test_df = np.split(processed_df.sample(frac=1, random_state=42), [int(0.7 * len(processed_df)), int(0.8 * len(processed_df))])

# Save the dataframes to CSV files
train_df.to_csv("./data/training_set.csv", index=False)
val_df.to_csv("./data/validation_set.csv", index=False)
test_df.to_csv("./data/test_set.csv", index=False)
print("Saved training_set.csv, validation_set.csv, and test_set.csv")


# You can now use `input_prompts` and `output_labels` for your LLM training.
# To see all generated prompts and labels, you can print the entire Series.
# print(input_prompts)
# print(output_labels)