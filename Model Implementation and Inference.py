import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Step 1: Define the path where the fine-tuned model was saved ---
# The Trainer saved the best model to a subdirectory in './results'
# The specific path will have a format like './results/checkpoint-XXX'
model_path = "./results/checkpoint-81" # <--- IMPORTANT: Update this path with the actual checkpoint folder name

# --- Step 2: Load the fine-tuned model and tokenizer ---
# The model and tokenizer are saved in the same directory by the Trainer
print("Step 1: Loading fine-tuned model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model loaded successfully!")
except OSError as e:
    print(f"Error loading model from {model_path}. Please make sure the path is correct and the model has been trained and saved.")
    print(f"Original error: {e}")
    exit()

# Set the model to evaluation mode
model.eval()

# --- Step 3: Create a prediction function ---
# This function encapsulates the entire inference process
def predict_maliciousness(text_prompt):
    """
    Predicts whether a given text prompt corresponds to a malicious update.

    Args:
        text_prompt (str): The new update log to classify.

    Returns:
        int: A prediction of 0 (malicious) or 1 (not malicious).
    """
    # Tokenize the input text
    inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)

    # Use a 'no_grad' context to disable gradient calculation for inference
    with torch.no_grad():
        # Get the model's raw output (logits)
        outputs = model(**inputs)

    # Get the predicted class by finding the index of the highest logit
    # For binary classification, index 0 is malicious, index 1 is not malicious
    # based on our fine-tuning labels.
    predicted_class_id = torch.argmax(outputs.logits).item()
    return predicted_class_id

# --- Step 4: Example of using the function with new data ---
print("\nStep 3: Making a prediction on new, unclassified data...")
new_update_log_honest = "A simulation log details an update at time 48 for node N04 with a 'honest' role. The update traveled via the OTA channel, had a path length of 2, and was created by OEM (version 5). The baseline system acceptance was 1, while an LLM scored it 0.8 and accepted it with status 1. Authenticity was 1, rollforward was 1, and chain integrity was 1."
new_update_log_adversary = "A simulation log details an update at time 241 for node N14 with a 'adversary' role. The update traveled via the OTA channel, had a path length of 2, and was created by OEM (version 14). The baseline system acceptance was 0, while an LLM scored it 0.5 and accepted it with status 0. Authenticity was 1, rollforward was 1, and chain integrity was 0."

# Call the function for both examples
prediction_honest = predict_maliciousness(new_update_log_honest)
prediction_adversary = predict_maliciousness(new_update_log_adversary)

# Interpret and print the results
print(f"Prediction for 'honest' update log: {prediction_honest}")
print(f"Prediction for 'adversary' update log: {prediction_adversary}")

# --- Step 5: Interpretation of the Output Label ---
print("\nStep 4: Interpreting the results...")
print("Prediction '0' corresponds to a malicious update.")
print("Prediction '1' corresponds to a non-malicious (honest) update.")
