import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Performance Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Player Performance Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("ipl_prepared.csv")

df = load_data()

player = st.selectbox("Select Player", sorted(df['Player'].unique()))
stat   = st.selectbox("Predict", ['Runs', 'Wickets', 'Fantasy_Points', 'Strike_Rate'])

player_df = df[df['Player'] == player].sort_values('Year')

if len(player_df) < 3:
    st.warning("Not enough data to predict. Need at least 3 years of data.")
else:
    X = player_df['Year'].values.reshape(-1, 1)
    y = player_df[stat].values

    model = LinearRegression()
    model.fit(X, y)

    next_year = player_df['Year'].max() + 1
    prediction = model.predict([[next_year]])[0]

    st.metric(f"Predicted {stat} for {int(next_year)}", f"{prediction:.1f}")

    # Show historical + prediction in chart
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=player_df['Year'], y=y, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=[next_year], y=[prediction], mode='markers',
                             marker=dict(size=12, color='red', symbol='star'), name='Predicted'))

    fig.update_layout(title=f"{player} — {stat} Prediction", xaxis_title="Year", yaxis_title=stat)
    st.plotly_chart(fig, width='stretch')