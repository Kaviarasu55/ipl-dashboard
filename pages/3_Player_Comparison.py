import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Player Comparison", page_icon="⚔️", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("ipl_prepared.csv")


df = load_data()
st.title("⚔️ Player Comparison")

col1, col2 = st.columns(2)
player1 = col1.selectbox("Player 1", sorted(df["Player"].unique()), index=0)
player2 = col2.selectbox("Player 2", sorted(df["Player"].unique()), index=1)

stat = st.selectbox(
    "Compare by",
    ["Runs", "Wickets", "Fours", "Sixes", "Strike_Rate", "Economy", "Fantasy_Points"],
)

p1 = df[df["Player"] == player1][["Year", stat]].copy()
p2 = df[df["Player"] == player2][["Year", stat]].copy()
p1["Player"] = player1
p2["Player"] = player2

combined = pd.concat([p1, p2])
fig = px.line(
    combined,
    x="Year",
    y=stat,
    color="Player",
    markers=True,
    title=f"{stat} — {player1} vs {player2}",
)
st.plotly_chart(fig, width='stretch')

st.subheader("Career Summary")
summary = (
    df[df["Player"].isin([player1, player2])]
    .groupby("Player")
    .agg(
        Matches=("Matches", "sum"),
        Runs=("Runs", "sum"),
        Wickets=("Wickets", "sum"),
        Highest_Score=("Highest_Score", "max"),
        Avg_Strike_Rate=("Strike_Rate", "mean"),
        Fantasy_Points=("Fantasy_Points", "sum"),
    )
    .round(2)
    .T
)

st.dataframe(summary, width='stretch')

if st.button("🤖 Explain Comparison"):
    client = Groq(api_key=groq_key)
    prompt = f"""You are a cricket analyst. Compare these two IPL players based on their career stats and explain who is better and why in 4-5 lines.

Player 1: {player1}
Player 2: {player2}
Stat being compared: {stat}

Career Summary:
{summary.to_string()}"""

    with st.spinner("Analysing..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        st.info(response.choices[0].message.content)
