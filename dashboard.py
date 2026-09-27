import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Paths
DATA_PATH = Path("outputs/sentiment_results.csv")

# Title
st.title("📊 Customer Review Sentiment Analysis")
st.write("Analysis of customer reviews using VADER Sentiment Analysis.")

# Load data
df = pd.read_csv(DATA_PATH)

# Summary
positive = (df["Predicted_Sentiment"] == "Positive").sum()
negative = (df["Predicted_Sentiment"] == "Negative").sum()
neutral = (df["Predicted_Sentiment"] == "Neutral").sum()
total = len(df)

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", total)
col2.metric("Positive", positive)
col3.metric("Negative", negative)
col4.metric("Neutral", neutral)

st.divider()

# Sentiment distribution
sentiment_counts = (
    df["Predicted_Sentiment"]
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = ["Sentiment", "Count"]

fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    title="Sentiment Distribution",
    text="Count"
)

st.plotly_chart(fig, use_container_width=True)

# Pie chart
fig2 = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Sentiment Percentage",
    hole=0.3
)

st.plotly_chart(fig2, use_container_width=True)

# Reviews table
st.subheader("📝 Analyzed Reviews")

st.dataframe(
    df[[
        "Review",
        "Predicted_Sentiment",
        "Sentiment_Score"
    ]],
    use_container_width=True
)