import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
# LLMs
llm=OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True)

#Show stuff to the screen if there is a prompt
if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)
