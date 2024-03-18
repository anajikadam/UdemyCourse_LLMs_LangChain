import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
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

#Memory
memory=ConversationBufferMemory(input_key='topic', memory_key='chat_history')


# LLMs
llm=OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script',memory=memory)

sequential_chain=SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
                                 output_variables=['title', 'script'], verbose=True)
#Show stuff to the screen if there is a prompt
if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])


    with st.expander('Message History'):
        st.info(memory.buffer)