
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
from pypdf import PdfReader
from docx import Document as DocxDocument
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file for local development
load_dotenv()

# --- Helper Functions for Data Extraction ---

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
    """Creates and configures the QA chain with a dynamic and more natural prompt."""
    person_name = st.session_state.get("person_name", "the user")
    
    # *** CHANGE: Updated prompt for more direct and natural responses ***
    prompt_template_str = f"""
    You are "{person_name}'s AI", a friendly and professional chatbot assistant that has deep knowledge about {person_name}.
    Your goal is to answer questions about {person_name} directly and naturally, using the context provided as your source of truth.
    Do not mention that you are basing your answer on the provided text. Act as if you know this information innately.

    For example, if the user asks "What is their experience?", you should respond "{person_name} has 5 years of experience..." instead of "Based on the text, they have 5 years of experience...".

    If the answer is not in the context, politely say "I'm sorry, I don't have specific information about that regarding {person_name}."
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

def get_person_name(text_chunks):
    """Extracts the subject's name from the document chunks using the LLM."""
    if not text_chunks:
        return "ResumAI"

    try:
        temp_vector_store = get_vector_store(text_chunks)
        if temp_vector_store is None:
             return "ResumAI" 
        
        name_query = "What is the full name of the person these documents are about? Respond with only the name."
        docs = temp_vector_store.similarity_search(name_query, k=1)
        
        model = GoogleGenerativeAI(model="gemini-1.5-flash")
        chain = load_qa_chain(llm=model, chain_type="stuff")
        response = chain.invoke({"input_documents": docs, "question": name_query})
        
        name = response.get("output_text", "ResumAI").strip().split()[0] # Take the first name for brevity
        return name
    except Exception as e:
        st.warning(f"Could not automatically determine the name. Using default. Error: {e}")
        return "ResumAI"

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "person_name" not in st.session_state:
        st.session_state.person_name = "Your" 
    if "prompt_template" not in st.session_state:
        from langchain.prompts import PromptTemplate
        st.session_state.prompt_template = PromptTemplate

    # Sidebar for configuration
    with st.sidebar:
        st.subheader("Knowledge Base Configuration")
        
        # *** CHANGE: Removed username inputs and added clearer instructions ***
        st.markdown(
            """
            **How to use this app:**
            1.  **Provide Documents:** Upload a resume, cover letter, or any other documents about a person.
            2.  **For Best Results:** For profiles like LinkedIn, go to the profile, click `More` > `Save to PDF`, and upload that file.
            3.  **Process:** Click the "Process Documents" button.
            """
        )

        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if st.button("Process Documents"):
            if "GOOGLE_API_KEY" not in st.secrets:
                st.error("Google API Key not found. Go to Settings > Secrets to add it.")
            elif not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Processing documents... This might take a moment."):
                    # Configure API key
                    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

                    raw_text = ""
                    for file in uploaded_files:
                        if file.name.endswith(".pdf"):
                            raw_text += get_text_from_pdf(file)
                        elif file.name.endswith(".docx"):
                            raw_text += get_text_from_docx(file)
                        elif file.name.endswith(".txt"):
                            raw_text += get_text_from_txt(file)
                    
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    
                    if st.session_state.vector_store:
                        st.session_state.person_name = get_person_name(text_chunks)
                        st.session_state.messages = [] 
                        st.success(f"Ready to chat about {st.session_state.person_name}!")
                        st.rerun()

    # Main chat interface
    st.header(f"Chat with {st.session_state.person_name}'s AI 🤖")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        if st.session_state.vector_store is None:
            st.warning("Please process some documents first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                        chain = get_conversational_chain()
                        response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                        response_text = response["output_text"]
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == '__main__':
    main()
