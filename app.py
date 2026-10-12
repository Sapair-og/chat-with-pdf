import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_pdf_pages(pdf_docs):
    """One Document per PDF page, tagged with file name and page number for citations."""
    pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_no, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append(Document(page_content=page_text,
                                      metadata={"source": pdf.name, "page": page_no}))
    return pages

def get_text_chunks(pages):
    # Splits on paragraphs first, then sentences, then words, so chunks stay coherent.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(pages)

def format_sources(docs):
    """Unique (file, page, snippet) entries for the chunks the answer was based on."""
    seen, sources = set(), []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        snippet = " ".join(doc.page_content.split())[:220]
        sources.append({"source": key[0], "page": key[1], "snippet": snippet})
    return sources

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']}  \n> {s['snippet']}...")

def get_vectorstore(text_chunks):
    try:
        # Using a reliable, lightweight local embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
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
                model="gemini-2.5-flash",
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
                    # Extract text page by page
                    pages = get_pdf_pages(pdf_docs)
                    if not pages:
                        st.error("No readable text could be extracted. The PDFs might be empty or scanned images.")
                        st.stop()
                    
                    # Split into chunks
                    text_chunks = get_text_chunks(pages)
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore:
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, provider, api_key)
                        if st.session_state.conversation:
                            st.session_state.processed_files = [pdf.name for pdf in pdf_docs]
                            st.success(f"Processed {len(pages)} pages into {len(text_chunks)} chunks. You can start chatting.")
        
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
            render_sources(message.get("sources", []))

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
                        response = st.session_state.conversation.invoke({'question': user_question})
                        answer = response['answer']
                        sources = format_sources(response.get('source_documents', []))
                        st.markdown(answer)
                        render_sources(sources)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

if __name__ == '__main__':
    main()