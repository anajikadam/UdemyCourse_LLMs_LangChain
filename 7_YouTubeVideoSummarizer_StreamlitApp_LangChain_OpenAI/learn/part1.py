import streamlit as st
from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"]=  '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'

#App Framework
st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt=st.text_input("Plug in your prompt here")

# LLMs
llm=OpenAI(temperature=0.9)

#Show stuff to the screen if there is a prompt
if prompt:
    response = llm(prompt)
    st.write(response)
