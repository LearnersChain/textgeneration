#importing libraries

import os
import streamlit as st
from langchain.llms.bedrock import Bedrock

#function for text analysis
def get_text_analysis(input_content, analysis_type):
    llm = Bedrock(
        model_id = "cohere.command-text-v14",
        model_kwargs = {
            "max_tokens": 512,
            "temperature": 0,
            "p":0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods":"NONE"
        }
    )
    
    prompt = f"Provide the {analysis_type}: {input_content}"
    return llm.predict(prompt)
    
#Setting up the streamlit interface
st.set_page_config(page_title="Text Analysis", page_icon=":robot:")
st.title("Text Analysis helper")

input_text = st.text_area("Input text", label_visibility="collapsed")

# user options for analysis
analysis_options = st.multiselect("Select analysis type", ["Sentiment", "Keywords", "Entities"],["Sentiment"])

# analyze button and displaying results

go_button = st.button("Analyze", type="primary")
if go_button:
    with st.spinner("Analyzing......"):
        for analysis_type in analysis_options:
            response_content = get_text_analysis(input_text, analysis_type)
            st.subheader(analysis_type)
            st.write(response_content)
            
            
            result = get_text_analysis(input_text, analysis_type)
            st.write(f"{analysis_type}: {result}")
    