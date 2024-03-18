import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

import os
os.environ["OPENAI_API_KEY"]=  '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'

#App Framework
st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt=st.text_input("Plug in your prompt here")

#Prompt Templates
title_template=PromptTemplate(
    input_variables=['topic'],
    template='Write me Youtube video title about {topic}'

)

script_template=PromptTemplate(
    input_variables=['title'],
    template='Write me Youtube video script based on this TITLE {title}'

)
# LLMs
llm=OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True)

sequential_chain=SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)
#Show stuff to the screen if there is a prompt
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)
