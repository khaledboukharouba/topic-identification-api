import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint URL
API_URL = "http://localhost:8000/get_category"

# Custom CSS for slider and button


# Streamlit app title
st.title("Text Categorization App")
st.write("Enter text below to categorize it based on predefined topics.")

# Form to input text and specify probability threshold
with st.form(key="input_form"):
    text_input = st.text_area("Enter text to categorize", height=150)
    threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2)
    submit_button = st.form_submit_button(label="Submit", type="primary")

if submit_button:
    if not text_input.strip():
        st.warning("Please enter some text to categorize.")
    else:
        # Prepare request payload
        request_data = {"text": text_input}
        params = {"threshold": threshold}

        # Make request to FastAPI
        response = requests.post(API_URL, json=request_data, params=params)
        
        if response.status_code == 200:
            result = response.json()
            # st.subheader("Input Text")
            # st.write(result["description"])

            st.subheader("Relevant Topics with Confidence Scores")
            if result["top_topics"]:
                for idx, topic in enumerate(result["top_topics"], start=1):
                    st.write(f"{idx}. **{topic['topic']}** - Confidence: {topic['confidence']:.2%}")
                # Convert topics to DataFrame for tabular display
                topics_df = pd.DataFrame(result["top_topics"])
                # st.table(topics_df)
                topics_df['confidence'] = topics_df['confidence'].apply(lambda x: f'<span class="badge badge-success">{x:.0%}</span>')
                topics_df.columns = ['Google topic', 'Confidence']
                st.markdown(topics_df.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.write("No topics found with confidence above the threshold.")
        else:
            st.error("Failed to get category. Please try again.")
