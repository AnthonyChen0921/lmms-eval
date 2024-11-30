import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Load the JSON data
file_path = "evaluation_results.json"  # Replace with your JSON file path
with open(file_path, "r") as f:
    data = json.load(f)

# Extract predictions and ground truths
samples = data["samples"]["pope"]

# Initialize lists for predictions and ground truths
predictions = []
ground_truths = []
yes_count_pred = 0
no_count_pred = 0
yes_count_gt = 0
no_count_gt = 0

# Helper function to extract the answer from the prediction text
def extract_answer(prediction_text):
    if "model\n" in prediction_text:
        return prediction_text.split("model\n")[-1].strip().lower()
    return prediction_text.strip().lower()

# Process each sample
for sample in samples:
    prediction_text = sample["pope_accuracy"]["prediction"]
    ground_truth = sample["pope_accuracy"]["ground_truth"].strip().lower()

    # Extract the actual prediction after "model\n"
    prediction = extract_answer(prediction_text)

    # Convert predictions and ground truths to binary (1 for "yes", 0 for "no")
    pred_binary = 1 if prediction == "yes" else 0
    gt_binary = 1 if ground_truth == "yes" else 0

    # Count "yes" and "no" for predictions and ground truths
    if pred_binary == 1:
        yes_count_pred += 1
    else:
        no_count_pred += 1

    if gt_binary == 1:
        yes_count_gt += 1
    else:
        no_count_gt += 1

    predictions.append(pred_binary)
    ground_truths.append(gt_binary)

# Calculate metrics
accuracy = accuracy_score(ground_truths, predictions)
precision = precision_score(ground_truths, predictions, zero_division=0)
recall = recall_score(ground_truths, predictions, zero_division=0)
f1 = f1_score(ground_truths, predictions, zero_division=0)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"\nCount of 'yes' and 'no':")
print(f"Predictions - Yes: {yes_count_pred}, No: {no_count_pred}")
print(f"Ground Truths - Yes: {yes_count_gt}, No: {no_count_gt}")
