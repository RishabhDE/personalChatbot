# app.py
# This is the main file for your Streamlit application.
# It uses an environment variable for the API key, and UI inputs for usernames.
# Now with support for .txt file uploads as a workaround for scraping issues.

# First, update your 'requirements.txt' file with the following.
#
# streamlit
# langchain
# langchain_community
# langchain-google-genai
# google-generativeai
# faiss-cpu
# sentence-transformers
# InstructorEmbedding
# beautifulsoup4
# pypdf
# python-docx
# python-dotenv
#
# Then, install these dependencies by running:
# pip install -r requirements.txt
#
# Next, create a file named '.env' in the same directory.
# This file will store your secrets. Add the following line to it,
# replacing the placeholder with your actual data:
#
# GOOGLE_API_KEY="your_google_api_key_here"

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
import os
from pypdf import PdfReader
from docx import Document as DocxDocument
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Helper Functions for Data Extraction ---

def get_text_from_url(url):
    """Scrapes text content from a given URL. Prone to failure on modern sites."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        if not text:
            st.warning(f"Warning: No text was scraped from {url}. The site might be blocking scrapers.")
        return text
    except requests.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}. This is common for sites like LinkedIn.")
        return ""

def get_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def get_text_from_docx(docx_file):
    """Extracts text from an uploaded DOCX file."""
    try:
        doc = DocxDocument(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

# *** NEW: Function to read plain text files ***
def get_text_from_txt(txt_file):
    """Reads text from an uploaded TXT file."""
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

# --- Core RAG Logic ---

def get_text_chunks(raw_text):
    """Splits raw text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        st.warning("No text to process. Please provide data.")
        return None
    try:
        # Using a powerful open-source embedding model. It requires the 'InstructorEmbedding' package.
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.error("This can happen if you have installation issues. Try running 'pip install --force-reinstall InstructorEmbedding' in your terminal.")
        return None

def get_conversational_chain():
    """Creates and configures the QA chain with Gemini."""
    prompt_template = """
    You are "Rishabh's AI", a friendly and professional chatbot assistant.
    Your goal is to answer questions about Rishabh Malik based ONLY on the context provided.
    Rishabh is an Azure Data Engineer with 5 years of experience, specializing in PySpark, Python, and building data solutions on Azure. He's also a creative person who loves anime and riding his TVS Ronin.

    Answer the question as accurately as possible using the provided text.
    If the answer is not in the context, politely say "I'm sorry, I don't have that information about Rishabh in my knowledge base."
    Keep your answers concise and human-like, just as Rishabh would respond. Add a touch of professional humor if possible.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    model = GoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=st.session_state.prompt_template.from_template(prompt_template)
    )
    return chain

def handle_user_input(user_question):
    """Processes user question, queries vector store, and generates a response."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("The knowledge base isn't processed yet. Please click 'Process' first.")
        return
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key is not configured. Please set it in your environment.")
        return
        
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        docs = st.session_state.vector_store.similarity_search(user_question, k=3)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        st.info("This could be due to API key issues or content safety restrictions from the model.")

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Rishabh's AI (Env Configured)", page_icon="⚙️")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "prompt_template" not in st.session_state:
        from langchain.prompts import PromptTemplate
        st.session_state.prompt_template = PromptTemplate

    st.header("Chat with Rishabh's AI ⚙️")
    st.write("Ask me anything about Rishabh's professional background, skills, or projects!")

    user_question = st.text_input("Your question:", key="user_question_input", placeholder="e.g., What's Rishabh's experience with PySpark?")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Knowledge Base Configuration")

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            st.success("✅ Google API Key Loaded")
        else:
            st.error("❌ Google API Key not found. Please set it in your .env file.")

        linkedin_username = st.text_input("LinkedIn Username", placeholder="e.g., your-linkedin-profile")
        github_username = st.text_input("GitHub Username", placeholder="e.g., your-github-handle")

        # *** CHANGE: Added .txt to accepted file types ***
        st.markdown("For LinkedIn, it's best to 'Save to PDF' from your profile and upload it here.")
        uploaded_files = st.file_uploader(
            "Upload your Resume and other info (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            if not google_api_key:
                st.error("Cannot process without a Google API Key.")
            else:
                genai.configure(api_key=google_api_key)
                with st.spinner("Processing your data... This is the way."):
                    raw_text = ""
                    # Process uploaded files
                    for file in uploaded_files:
                        if file.name.endswith(".pdf"):
                            raw_text += get_text_from_pdf(file)
                        elif file.name.endswith(".docx"):
                            raw_text += get_text_from_docx(file)
                        elif file.name.endswith(".txt"):
                            raw_text += get_text_from_txt(file)
                    
                    # Process URLs
                    linkedin_url = f"https://www.linkedin.com/in/{linkedin_username}/" if linkedin_username else ""
                    github_url = f"https://github.com/{github_username}" if github_username else ""

                    if linkedin_url:
                        st.info(f"Attempting to scrape LinkedIn...")
                        raw_text += get_text_from_url(linkedin_url)
                    if github_url:
                        st.info(f"Attempting to scrape GitHub...")
                        raw_text += get_text_from_url(github_url)

                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks.")

                    vector_store = get_vector_store(text_chunks)

                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success("Knowledge base is ready! You can now ask questions.")

if __name__ == '__main__':
    main()
