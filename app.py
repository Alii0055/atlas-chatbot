import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Chatbot Atlas Cloud", layout="wide")
st.title("🤖 Chatbot Atlas Cloud Services")

@st.cache_resource(show_spinner=False)
def init_bot():
    # 1. Charger le texte local
    loader = TextLoader("atlas_services.txt", encoding="utf-8")
    docs = loader.load()

    # 2. Split du contenu
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    # 3. Embeddings
    embeddings = OpenAIEmbeddings()

    # 4. Si index FAISS existe, on le recharge — sinon on le crée
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embeddings)
    else:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")

    # 5. Création du chatbot
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = init_bot()

# Interface utilisateur
prompt = st.text_input("💬 Posez votre question sur les services Atlas Cloud :")

if prompt:
    with st.spinner("🔎 Recherche de la réponse..."):
        try:
            reponse = qa.run(prompt)
            st.success(reponse)
        except Exception as e:
            st.error(f"Erreur lors de la réponse : {str(e)}")
