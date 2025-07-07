import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

st.set_page_config(page_title="Chatbot Atlas Cloud", layout="wide")
st.title("Chatbot Atlas Cloud Services")

# URL à scraper (tu peux étendre à plusieurs)
URLS = ["https://atlascloudservices.com/produits-et-solutions/"]

@st.cache_resource(show_spinner=False)
def init_bot():
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = init_bot()

if prompt := st.text_input("Posez votre question sur Atlas Cloud Services :"):
    with st.spinner("Réflexion en cours…"):
        answer = qa.run(prompt)
    st.markdown(f"**Réponse :** {answer}")
