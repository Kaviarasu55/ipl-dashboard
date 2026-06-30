import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="IPL Assistant", page_icon="🤖", layout="wide")
st.title("🤖 IPL RAG Assistant")


@st.cache_data
def load_data():
    return pd.read_csv("ipl_prepared.csv")


df = load_data()

# Show available columns so user knows what to ask
with st.expander("📋 Available Data Columns"):
    st.write(list(df.columns))

question = st.text_input(
    "Ask anything about IPL players:", placeholder="Who scored most runs in 2016?"
)

if st.button("Ask") and question:
    client = Groq(api_key=groq_key)

    # Step 1 — Ask Groq to generate pandas code
    code_prompt = f"""
You are a Python/pandas expert. 
I have a dataframe called `df` with these columns:
{list(df.columns)}

Sample data:
{df.head(3).to_string()}

Write ONLY a single Python pandas expression to answer this question:
"{question}"

Rules:
- Use only `df` variable
- Return only the final expression, no explanation, no markdown, no print()
- The expression must return a value (string, number, or dataframe)
- Example for single year: df[df['Year']==2016].sort_values('Runs', ascending=False).head(1)[['Player','Runs']]
- Example for overall/career: df.groupby('Player')['Wickets'].sum().sort_values(ascending=False).head(1)
- Use groupby+sum for any question asking about overall/career/total/all time stats
"""

    with st.spinner("Thinking..."):
        code_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": code_prompt}],
        )
        generated_code = code_response.choices[0].message.content.strip()
        # Clean any markdown backticks if model adds them
        generated_code = (
            generated_code.replace("```python", "").replace("```", "").strip()
        )

        st.caption("🔍 Generated Code:")
        st.code(generated_code)

        # Step 2 — Run the generated code on df
        try:
            result = eval(generated_code)

            # Step 3 — Ask Groq to explain the result in plain English
            explain_prompt = f"""
Question: {question}
Data result: {str(result)}

This is the complete and correct answer from the dataset.Answer the question confidently in 2-3 clear sentences based on the data result.
"""
            explain_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": explain_prompt}],
            )
            st.success(explain_response.choices[0].message.content)

        except Exception as e:
            st.error(f"Could not process: {e}")
            st.info("Try rephrasing your question.")
