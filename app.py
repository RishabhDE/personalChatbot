
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
import os
from pypdf import PdfReader
from docx import Document as DocxDocument
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file for local development
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
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversational_chain():
    """Creates and configures the QA chain with a dynamic prompt."""
    person_name = st.session_state.get("person_name", "the user")
    
    # Generic prompt template that uses the extracted name
    prompt_template_str = f"""
    You are "{person_name}'s AI", a friendly and professional chatbot assistant.
    Your goal is to answer questions about {person_name} based ONLY on the context provided.

    Answer the question as accurately as possible using the provided text.
    If the answer is not in the context, politely say "I'm sorry, I don't have that information about {person_name} in my knowledge base."
    Keep your answers concise and human-like.

    Context:
    {{context}}

    Question:
    {{question}}

    Helpful Answer:
    """
    
    model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=st.session_state.prompt_template.from_template(prompt_template_str)
    )
    return chain

# *** NEW: Function to extract the person's name from the text ***
def get_person_name(text_chunks):
    """Extracts the subject's name from the document chunks using the LLM."""
    if not text_chunks:
        return "ResumAI"

    # Create a small, targeted vector store for name extraction
    try:
        temp_vector_store = get_vector_store(text_chunks)
        if temp_vector_store is None:
             return "ResumAI" # Default name
        
        # Query for the name
        name_query = "What is the full name of the person these documents are about? Respond with only the name."
        docs = temp_vector_store.similarity_search(name_query, k=1)
        
        # Use a simple LLM call to get the name
        model = GoogleGenerativeAI(model="gemini-1.5-flash")
        chain = load_qa_chain(llm=model, chain_type="stuff")
        response = chain.invoke({"input_documents": docs, "question": name_query})
        
        # Clean up the name
        name = response.get("output_text", "ResumAI").strip()
        return name
    except Exception as e:
        st.warning(f"Could not automatically determine the name. Using default. Error: {e}")
        return "ResumAI" # Default name


def handle_chat_input(user_question):
    """Handles user input, gets response, and updates chat history."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("The knowledge base isn't processed yet. Please click 'Process' first.")
        return

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                chain = get_conversational_chain()
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                response_text = response["output_text"]
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "person_name" not in st.session_state:
        st.session_state.person_name = "Your" # Default
    if "prompt_template" not in st.session_state:
        from langchain.prompts import PromptTemplate
        st.session_state.prompt_template = PromptTemplate

    st.header(f"Chat with {st.session_state.person_name}'s AI 🤖")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar configuration
    with st.sidebar:
        st.subheader("Knowledge Base Configuration")

        if "GOOGLE_API_KEY" in st.secrets:
            st.success("✅ Google API Key found.")
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            st.error("❌ Google API Key not found. Go to Settings > Secrets to add it.")

        linkedin_username = st.text_input("LinkedIn Username (Optional)")
        github_username = st.text_input("GitHub Username (Optional)")
        
        st.markdown("Upload documents (PDF, DOCX, TXT) to build the knowledge base.")
        uploaded_files = st.file_uploader(
            "Upload your Resume and other info",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if "GOOGLE_API_KEY" not in st.secrets:
                st.error("Cannot process without a Google API Key in Secrets.")
            elif not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Processing documents... This might take a moment."):
                    raw_text = ""
                    for file in uploaded_files:
                        if file.name.endswith(".pdf"):
                            raw_text += get_text_from_pdf(file)
                        elif file.name.endswith(".docx"):
                            raw_text += get_text_from_docx(file)
                        elif file.name.endswith(".txt"):
                            raw_text += get_text_from_txt(file)
                    
                    linkedin_url = f"https://www.linkedin.com/in/{linkedin_username}/" if linkedin_username else ""
                    github_url = f"https://github.com/{github_username}" if github_username else ""

                    if linkedin_url:
                        raw_text += get_text_from_url(linkedin_url)
                    if github_url:
                        raw_text += get_text_from_url(github_url)

                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    
                    if st.session_state.vector_store:
                        # Extract the name and update the session
                        st.session_state.person_name = get_person_name(text_chunks)
                        st.session_state.messages = [] # Clear previous chat
                        st.success(f"Knowledge base ready for {st.session_state.person_name}!")
                        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question about the provided documents..."):
        handle_chat_input(prompt)


if __name__ == '__main__':
    main()
