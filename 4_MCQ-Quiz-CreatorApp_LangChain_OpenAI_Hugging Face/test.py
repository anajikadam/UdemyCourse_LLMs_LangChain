import json
import streamlit as st
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
import pandas as pd
import traceback
from utils import parse_file, RESPONSE_JSON, get_table_data
load_dotenv()

# llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0)

