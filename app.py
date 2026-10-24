import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.title("Annual Report AI Extractor")
st.write("Upload a financial report (e.g., Hindustan Unilever Annual Report) to extract metrics securely.")

# Securely load the API key from Streamlit's server
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Cache the vector database so it doesn't reload on every single question
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(pages, embeddings)
    return vector_store.as_retriever()

uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily so the AI can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("Processing document..."):
        retriever = process_pdf(temp_path)
    
    # Initialize the AI Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Define the FP&A instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an FP&A analyst. Answer the question based ONLY on the context below. If you don't know, say so.\n\nContext: {context}"),
        ("human", "{input}")
    ])
    
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    # Chat Input
    query = st.text_input("Ask a question (e.g., 'What is the operating margin?'):")
    if query:
        response = rag_chain.invoke({"input": query})
        st.success(response["answer"])
