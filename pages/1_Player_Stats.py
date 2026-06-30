import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Player Stats", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("ipl_prepared.csv")


df = load_data()


st.title("🏏 IPL Player Performance Dashboard")

st.sidebar.header("Filters")

# Player dropdown — sorted alphabetically
all_players = sorted(df["Player"].unique())
selected_player = st.sidebar.selectbox("Select Player", all_players)

# Year range slider
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),  # default = full range
)


# Filter by selected player and year range
player_df = df[
    (df["Player"] == selected_player)
    & (df["Year"] >= year_range[0])
    & (df["Year"] <= year_range[1])
]


st.subheader(f"📊 {selected_player} — Stats Overview")

# If no data found for selected filters
if player_df.empty:
    st.warning("No data found for this player in the selected year range.")
else:
    # st.metric shows a nice card with label + value
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Runs", int(player_df["Runs"].sum()))
    col2.metric("Total Wickets", int(player_df["Wickets"].sum()))
    col3.metric("Matches", int(player_df["Matches"].sum()))
    col4.metric("Highest Score", int(player_df["Highest_Score"].max()))
    col5.metric("Fantasy Points", int(player_df["Fantasy_Points"].sum()))

    # Show raw year-by-year table
    display_df = player_df.reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df)
    st.subheader("🔥 Performance Heatmap")

    # Columns we want to show in heatmap
    heatmap_cols = [
        "Runs",
        "Wickets",
        "Fours",
        "Sixes",
        "Strike_Rate",
        "Economy",
        "Fantasy_Points",
    ]

    # Rows = Years, Columns = Stats
    heatmap_data = player_df.set_index("Year")[heatmap_cols]

    # Normalize each column 0 to 1 independently
    # # So Columns compares with itself
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (
        heatmap_data.max() - heatmap_data.min()
    )

    fig = px.imshow(
        heatmap_normalized,  # colors based on normalized values
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=False,  # we'll add original values manually
    )

    # Show original values as text, not normalized ones
    fig.update_traces(text=heatmap_data.values, texttemplate="%{text}")
    fig.update_layout(
        xaxis_title="Stats",
        yaxis_title="Year",
        coloraxis_showscale=False,  # hides the color bar on the side
    )

    st.plotly_chart(fig, width='stretch')

    if st.button("🤖 Explain Heatmap"):
        client = Groq(api_key=groq_key)
        heatmap_text = heatmap_data.to_string()
    
        prompt = f"You are a cricket analyst. Explain this IPL player's heatmap stats in 4-5 lines. Highlight peak years, decline, and consistency.\n\nPlayer: {selected_player}\n\nStats:\n{heatmap_text}"
    
        with st.spinner("Analysing..."):
           response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(response.choices[0].message.content)


    st.subheader("🕸️ Player Radar Chart")

    available_years = sorted(player_df["Year"].unique())
    selected_year = st.selectbox(
        "Select Year for Radar", available_years, index=len(available_years) - 1
    )

    year_data = player_df[player_df["Year"] == selected_year].iloc[0]
    radar_cols = ["Runs", "Wickets", "Fours", "Sixes", "Strike_Rate", "Fantasy_Points"]

    # Normalize against player's own career range
    norm = lambda col: (year_data[col] - player_df[col].min()) / (
        player_df[col].max() - player_df[col].min() or 1
    )
    normalized_vals = [norm(col) for col in radar_cols]

    # Show actual values as labels
    labels = [f"{col}<br>{year_data[col]:.1f}" for col in radar_cols]

    fig2 = go.Figure(
        go.Scatterpolar(
            r=normalized_vals,
            theta=labels,
            fill="toself",
            name=f"{selected_player} ({selected_year})",
        )
    )
    fig2.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=True)

    st.plotly_chart(fig2, width='stretch')

    if st.button("🤖 Explain Radar"):
     client = Groq(api_key=groq_key)
     radar_text = "\n".join([f"{col}: {year_data[col]:.1f}" for col in radar_cols])
    
     prompt = f"You are a cricket analyst. Explain this player's radar chart in 4-5 lines. Highlight strengths, weaknesses and overall role.\n\nPlayer: {selected_player} ({selected_year})\n\nStats:\n{radar_text}"
    
     with st.spinner("Analysing..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(response.choices[0].message.content)