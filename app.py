import os
import streamlit as st
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="C++ Rag Chatbot",page_icon=":robot_face:")
st.title("🤖 C++ Rag Chatbot")
st.write("Ask any question related to c++ introduction")

#load environement variables
load_dotenv()

#3.chache document loading
@st.cache_resource

#4.load documents
def load_vector_store():
    loader =TextLoader("C++_Introduction.txt",encoding="utf-8")
    documents=loader.load()

    #5. split text

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents=text_splitter.split_documents(documents)
    
    # 6.embeddings
    embedding=HuggingFaceEmbeddings(
        model_name="all-miniLM-L6-v2"

    )
    # 7.Crate FAISS vector store
    db=FAISS.from_documents(final_documents,embedding)
    return db
db=load_vector_store()
query=st.text_input("Enter your question here")
if query:
    #converts user questions to embeddings
    #searches FAISS database
    #return top 3 relevant chunks
    document=db.similarity_search(query,k=3)
    st.subheader("Retrieved Context")
    for i,doc in enumerate(document):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(doc.page_content)

