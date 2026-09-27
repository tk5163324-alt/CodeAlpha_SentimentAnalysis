import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Paths
DATA_PATH = Path("data/reviews.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate sentiment score
df["Sentiment_Score"] = df["Review"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

# Classify sentiment
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Predicted_Sentiment"] = df["Sentiment_Score"].apply(classify_sentiment)

# Display results
print("\nSentiment Analysis Results")
print("=" * 40)
print(df[["Review", "Predicted_Sentiment", "Sentiment_Score"]])

# Sentiment count
sentiment_counts = df["Predicted_Sentiment"].value_counts()

print("\nSentiment Summary")
print("=" * 40)
print(sentiment_counts)

# Save analyzed dataset
df.to_csv(OUTPUT_DIR / "sentiment_results.csv", index=False)

# -----------------------------
# 1. Sentiment Distribution
# -----------------------------
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Predicted_Sentiment"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentiment_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------
# 2. Sentiment Pie Chart
# -----------------------------
plt.figure(figsize=(7, 7))

sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage of Customer Sentiments")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentiment_percentage.png",
    dpi=300
)

plt.close()

print("\nAnalysis completed successfully!")
print("Results saved in the outputs folder.")