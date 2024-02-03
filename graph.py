import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import result
data = pd.read_csv("result.csv")


# Display the table
print("Model Ranking Table:")
print(
    data[["Model", "Model_size_GB", "Inference_Time_ms", "BLEU_Score", "Fact_Checking_Score_(0-100)", "Rank"]].sort_values(
        by="Rank"
    )
)

# Bar chart
labels = data["Model"]
num_models = len(labels)

# Parameters for bar chart
model_size = data["Model_size_GB"]
inference_time = data["Inference_Time_ms"]
BLEU_score = data["BLEU_Score"]
fact_checking_score = data["Fact_Checking_Score_(0-100)"]
ranks = data["Rank"]

# Normalize ranks to a scale of 0 to 1 for better comparison
normalized_ranks = ranks / np.max(ranks)

# Define colors for each metric
colors = ['yellow', 'green', 'orange', 'purple', 'black']

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
index = np.arange(num_models)

ax.bar(index,model_size,width=bar_width,label="Model_size_GB", color=colors[0])
ax.bar(index,inference_time,width=bar_width,label="Inference_Time_ms",bottom=model_size, color=colors[1])
ax.bar(index, BLEU_score, width=bar_width, label="BLEU_Score",bottom=model_size + inference_time, color=colors[2])
ax.bar(index,fact_checking_score,width=bar_width,label="Fact_Checking_Score_(0-100)",bottom=model_size + inference_time + BLEU_score, color=colors[3])
ax.bar(
    index,
    normalized_ranks,
    width=bar_width,
    label="Normalized Rank",
    color=colors[4],
    alpha=0.5,
)

ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.set_ylabel("Metrics")
ax.set_title("Text Generation Model Comparison Through Topsis")
ax.legend()
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("BarChart.png")
plt.show()