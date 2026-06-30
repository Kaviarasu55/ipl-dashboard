import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Fantasy Leaderboard", page_icon="🏆", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("ipl_prepared.csv")


df = load_data()
st.title("🏆 Fantasy Leaderboard")

years_options = ["Overall"] + sorted(df["Year"].unique().tolist())
selected_year = st.selectbox("Select Year", years_options)
filtered = df if selected_year == "Overall" else df[df["Year"] == selected_year]


def explain_leaderboard(category, leaderboard_df):
    client = Groq(api_key=groq_key)
    top10 = leaderboard_df.head(10).to_string()
    prompt = f"""You are a cricket analyst. Explain this IPL Fantasy {category} Leaderboard for {selected_year} in 4-5 lines.
Highlight top performers, what makes them stand out, and any interesting patterns.

Leaderboard:
{top10}"""
    with st.spinner("Analysing..."):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        st.info(response.choices[0].message.content)


tab1, tab2, tab3, tab4 = st.tabs(
    ["⚔️ Batting", "🎳 Bowling", "🌟 Allrounder", "🏅 Overall"]
)

with tab1:
    batting_lb = (
        filtered.groupby("Player")["Batting_Fantasy_Points"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .assign(Rank=lambda x: range(1, len(x) + 1))
        .set_index("Rank")
    )
    st.dataframe(batting_lb, width='stretch')
    if st.button("🤖 Explain Batting Leaderboard"):
        explain_leaderboard("Batting", batting_lb)

with tab2:
    bowling_lb = (
        filtered.groupby("Player")["Bowling_Fantasy_Points"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .assign(Rank=lambda x: range(1, len(x) + 1))
        .set_index("Rank")
    )
    st.dataframe(bowling_lb, width='stretch')
    if st.button("🤖 Explain Bowling Leaderboard"):
        explain_leaderboard("Bowling", bowling_lb)

with tab3:
    allrounder_df = (
        filtered.groupby("Player")
        .agg(
            Total_Runs=("Runs", "sum"),
            Total_Wickets=("Wickets", "sum"),
            Allrounder_Fantasy_Points=("Allrounder_Fantasy_Points", "sum"),
        )
        .reset_index()
    )
    min_runs = 1000 if selected_year == "Overall" else 150
    min_wickets = 50 if selected_year == "Overall" else 8
    allrounder_df = allrounder_df[
        (allrounder_df["Total_Runs"] >= min_runs)
        & (allrounder_df["Total_Wickets"] >= min_wickets)
    ].sort_values("Allrounder_Fantasy_Points", ascending=False)
    allrounder_df.index = range(1, len(allrounder_df) + 1)
    allrounder_df.index.name = "Rank"
    st.dataframe(allrounder_df, width='stretch')
    if st.button("🤖 Explain Allrounder Leaderboard"):
        explain_leaderboard("Allrounder", allrounder_df)

with tab4:
    overall_lb = (
        filtered.groupby("Player")["Fantasy_Points"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .assign(Rank=lambda x: range(1, len(x) + 1))
        .set_index("Rank")
    )
    st.dataframe(overall_lb, width='stretch')
    if st.button("🤖 Explain Overall Leaderboard"):
        explain_leaderboard("Overall", overall_lb)
