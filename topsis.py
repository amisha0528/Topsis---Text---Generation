import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("data.csv")

#Extract relevant data
model_size = data["Model_size_GB"].values
inference_time = data["Inference_Time_ms"].values
bleu_score = data["BLEU_Score"].values
fact_checking_score = data["Fact_Checking_Score_(0-100)"].values

# Normalize the data matrix
normalized_matrix = np.column_stack(
    [
        np.max(model_size) / model_size,                 # Minimize (smaller model size is better)
        np.max(inference_time) / inference_time,         # Minimize (lower inference time is better)
        bleu_score / np.max(bleu_score),                 # Maximize (higher BLEU score is better)
        fact_checking_score / np.max(fact_checking_score) # Maximize (higher fact-checking score is better)
    ]
)

# Define weights
weights = np.array([0.3, 0.3, 0.2, 0.2])

weighted_normalized_matrix = normalized_matrix * weights
# Ideal and Negative-Ideal Solutions
ideal_solution = np.max(weighted_normalized_matrix, axis=0)
negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)

# Calculate the distance to ideal and negative-ideal solutions
distance_to_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution) ** 2, axis=1))
distance_to_negative_ideal = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution) ** 2, axis=1))

# Calculate the relative closeness to the ideal solution(Topsis score)
topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

# Rank the models based on their Topsis score
data["TOPSIS_Score"] = topsis_score
data["Rank"] = data["TOPSIS_Score"].rank(ascending=False)

# Display the results table
print("Model Ranking:")
print(data[["Model", "TOPSIS_Score", "Rank"]])

# Save the results to a CSV file
data.to_csv("result.csv", index=False)