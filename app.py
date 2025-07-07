from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

st.title("Chatbot Atlas Cloud Services")

@st.cache_resource(show_spinner=False)
def init_bot():
    loader = TextLoader("atlas_services.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = init_bot()

prompt = st.text_input("Posez votre question :")

if prompt:
    with st.spinner("Recherche de la réponse…"):
        response = qa.run(prompt)
    st.markdown(f"**Réponse :** {response}")
