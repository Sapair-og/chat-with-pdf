import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        # Using a reliable, lightweight local embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def get_conversation_chain(vectorstore, provider, api_key):
    try:
        if provider == "OpenAI":
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.3
            )
        elif provider == "Google Gemini":
            llm = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-1.5-flash",
                temperature=0.3
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing LLM model: {str(e)}")
        return None

def main():
    load_dotenv()
    
    # Page Configuration with modern style setting
    st.set_page_config(
        page_title='Chat with PDF',
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom Styling
    st.markdown("""
        <style>
        .main {
            background-color: #0f1115;
            color: #e2e8f0;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #2563eb;
            color: white;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
            transform: translateY(-1px);
        }
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }
        .sidebar .sidebar-content {
            background-color: #1e293b;
        }
        </style>
    """, unsafe_allow_html=True)

    # Session State Initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Sidebar Options
    with st.sidebar:
        st.title("📚 Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "Select LLM Provider",
            ["OpenAI", "Google Gemini"]
        )
        
        # Retrieve key from environment or user input
        env_key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
        if env_key_name not in os.environ and "GOOGLE_API_KEY" in os.environ and provider == "Google Gemini":
            env_key_name = "GOOGLE_API_KEY"
            
        env_key = os.getenv(env_key_name, "")
        
        # User input for key if not in env or to override it
        api_key = st.text_input(
            f"Enter {provider} API Key",
            type="password",
            value=env_key,
            placeholder=f"Loaded from .env" if env_key else f"Enter key to start..."
        )
        
        st.markdown("---")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        # Process button
        if st.button('Process Documents', use_container_width=True):
            if not api_key:
                st.warning(f"Please provide an API key for {provider} before processing documents.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Processing documents..."):
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text could be extracted. The PDFs might be empty or scanned images.")
                        st.stop()
                    
                    # Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore:
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, provider, api_key)
                        if st.session_state.conversation:
                            st.session_state.processed_files = [pdf.name for pdf in pdf_docs]
                            st.success("Documents successfully processed! You can now start chatting.")
        
        # Status/Processed files list
        if st.session_state.processed_files:
            st.markdown("### Processed Files:")
            for file_name in st.session_state.processed_files:
                st.markdown(f"- 📄 `{file_name}`")
                
        st.markdown("---")
        
        # Clear Chat Button
        if st.button("Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            if "conversation" in st.session_state and st.session_state.conversation is not None:
                # Reset conversation memory buffer
                if hasattr(st.session_state.conversation, 'memory') and st.session_state.conversation.memory:
                    st.session_state.conversation.memory.clear()
            st.rerun()

    # Main Area Layout
    st.title("Chat with multiple PDFs :books:")
    st.markdown("Interact and ask questions about your documents in real-time. Upload documents via the sidebar to get started.")

    # Render Chat History
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])

    # Chat Input Box at the bottom
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if not st.session_state.conversation:
            st.info("Please upload and process PDF documents first to start chatting.")
        else:
            # Append and render user question
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate and render response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation({'question': user_question})
                        answer = response['answer']
                        st.markdown(answer)
                        # Append assistant response
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

if __name__ == '__main__':
    main()