# from langchain.document_loaders import UnstructuredURLLoader

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# urls = [
#     # 'https://www.mosaicml.com/blog/mpt-7b',
#     'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
#     'https://lmsys.org/blog/2023-03-30-vicuna/'
#     ]

urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()


def load_urls(urls):
    loaders=UnstructuredURLLoader(urls=urls)
    data=loaders.load()
    return data

def get_text_chunks(data):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks=text_splitter.split_documents(data)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    #embeddings=HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


# data=load_urls(urls)
# print(data)
# #Split the Text into Chunks
# text_chunks = get_text_chunks(data)
# print(len(text_chunks))










