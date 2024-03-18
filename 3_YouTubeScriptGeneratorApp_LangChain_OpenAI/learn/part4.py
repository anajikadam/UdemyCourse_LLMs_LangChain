import streamlit as st
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, SequentialChain

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import os

from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"]=  OPENAI_API_KEY

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
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')

sequential_chain=SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
                                 output_variables=['title', 'script'], verbose=True)
#Show stuff to the screen if there is a prompt
if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

