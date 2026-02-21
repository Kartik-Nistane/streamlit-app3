# import os
# import streamlit as st
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama

# st.set_page_config(page_title="C++ Rag Chatbot",layout="wide")
# st.title("🤖 C++ Rag Chatbot")
# st.write("Ask any question related to c++ introduction")
# load_dotenv()

# llm = OllamaLLM(model="gemma2:2b")

# @st.cache_resource

# def load_vector_store():
#     loader =TextLoader("C++_Introduction.txt",encoding="utf-8")
#     documents=loader.load()

#     text_splitter=RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=20
#     )
#     final_documents=text_splitter.split_documents(documents)
    
#     embedding=HuggingFaceEmbeddings(
#         model_name="all-miniLM-L6-v2"

#     )
#     db=FAISS.from_documents(final_documents,embedding)
#     return db

# db=load_vector_store()
# query=st.text_input("Enter your question here")
# if query:
#     document=db.similarity_search(query,k=3)
#     st.subheader("Retrieved Context")
#     for i,doc in enumerate(document):
#         st.markdown(f"**Chunk {i+1}:**")
#         st.write(doc.page_content)



import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="C++ Rag Chatbot",layout="wide")
st.title("🤖 C++ Rag Chatbot")


@st.cache_resource

def load_vector_store():
    loader =TextLoader("C++_Introduction.txt",encoding="utf-8")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents=text_splitter.split_documents(documents)
    
    embedding=HuggingFaceEmbeddings(
        model_name="all-miniLM-L6-v2"

    )
    db=FAISS.from_documents(final_documents,embedding)
    return db

db=load_vector_store()
llm=Ollama(model="gemma2:2b")

#chat interface
text_input = st.chat_input("Type your message here...")
if text_input:
    with st.spinner("Generating response..."):
        docs=db.similarity_search(text_input)
        context="\n".join([doc.page_content for doc in docs])
    prompt=f"Answer the question based on the following context:\n{context}\nQuestion: {text_input}"
    response=llm.invoke(prompt)
    st.subheader("Response:")
    st.write(response)
    
