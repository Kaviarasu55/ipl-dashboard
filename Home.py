import streamlit as st

st.set_page_config(page_title="IPL Dashboard", page_icon="🏏", layout="wide")

st.title("🏏 IPL Player Performance Dashboard")
st.markdown("#### Your all-in-one IPL analytics platform powered by AI")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 **Player Stats**\n\nInteractive heatmap, radar chart, year-by-year stats with AI-powered chart explanations.")
    st.info("⚔️ **Player Comparison**\n\nCompare any two players side by side with career summary and AI analysis.")

with col2:
    st.success("🏆 **Fantasy Leaderboard**\n\nBatting, Bowling, Allrounder and Overall fantasy rankings with year filter and AI insights.")
    st.success("🌟 **Prediction**\n\nML-powered performance prediction using Linear Regression for runs, wickets and more.")

with col3:
    st.warning("🤖 **RAG Assistant**\n\nAsk anything about IPL players in plain English — powered by Groq LLM + Text-to-Pandas RAG.")
    st.warning("📁 **Dataset**\n\n283K+ ball-by-ball IPL records from 2008–2025, aggregated into player-year stats.")

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏏 Total Players", "500+")
col2.metric("📅 Years Covered", "2008 – 2025")
col3.metric("🎯 Total Records", "3,284")
col4.metric("🤖 AI Features", "3")