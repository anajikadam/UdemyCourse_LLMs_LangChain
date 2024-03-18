import streamlit as st
import pickle
import os

from PyPDF2 import PdfReader

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores.elastic_vector_search import ElasticVectorSearch
# from langchain_community.vectorstores import Pinecone
# from langchain_community.vectorstores import Weaviate
from langchain_community.vectorstores import FAISS


from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

def main():
    st.header('Chat with  PDF 💬')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown('''
    This is an LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model 
    ''')

    # Upload a PDF File
    pdf = st.file_uploader("Upload your PDF File", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
                                                        chunk_size=1000,
                                                        chunk_overlap=200,
                                                        length_function=len
                                                    )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks[0])
        store_name = pdf.name[:-4] # remove .pdf extension
        st.write(store_name)
        embeddings = OpenAIEmbeddings()
        if os.path.exists(f"{store_name}"):
            VectorStore = FAISS.load_local(f"{store_name}", embeddings)
            # with open(f"{store_name}.pkl", "rb") as f:
            #     VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            VectorStore = FAISS.from_texts(chunks, embeddings)
            # with open(f"{store_name}.pkl", "wb") as f:
                # pickle.dump(VectorStore, f)
            VectorStore.save_local(f"{store_name}")
            st.write('Embeddings Created')

        query = st.text_input("Ask Question from your PDF File")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()

