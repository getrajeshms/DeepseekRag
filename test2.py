import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Using FAISS for persistent vector store
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import asyncio
import os

# Constants and configurations
PDF_STORAGE_PATH = 'document_store/pdfs/'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Cached model initialization
@st.cache_resource
def init_models():
    return {
        'embeddings': OllamaEmbeddings(model="deepseek-r1:1.5b"),
        'llm': OllamaLLM(model="deepseek-r1:1.5b", temperature=0)
    }

# Cached prompt template
@st.cache_resource
def get_prompt_template():
    return ChatPromptTemplate.from_template("""
    You are a highly intelligent and precise document assistant. Your task is to answer user queries **ONLY** using the provided context.  
    Follow these rules:
    - **Do not use external knowledge** beyond the given context.  
    - **If the answer is not in the context, say: "I don’t know based on the given document."**  
    - **Keep the answer clear and concise** (maximum 5-6 sentences).  
    - **Maintain accuracy** and do not speculate.

    ### **Context:**  
    {document_context}  

    ### **User Query:**  
    {user_query}  

    ### **Assistant Response:**  
    """)

# Function to process PDF and split it into chunks
def process_document(file_content):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    temp_path = os.path.join(PDF_STORAGE_PATH, "temp.pdf")
    
    with open(temp_path, "wb") as f:
        f.write(file_content)
    
    loader = PDFPlumberLoader(temp_path)
    documents = loader.load()
    
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    return splitter.split_documents(documents)

# Properly handling async functions in Streamlit
def run_async_task(async_func, *args):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(async_func(*args), loop).result()
    else:
        return asyncio.new_event_loop().run_until_complete(async_func(*args))

# Asynchronous chunk processing with FAISS
async def process_chunks_async(chunks, embeddings):
    return await FAISS.afrom_documents(chunks, embeddings)

# Asynchronous response generation
async def generate_response_async(query, context_docs, models, prompt_template):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    response_chain = prompt_template | models['llm']
    
    response = await response_chain.ainvoke({
        "user_query": query,
        "document_context": context_text
    })
    
    return response.strip()

# Main UI
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Initialize models
models = init_models()
prompt_template = get_prompt_template()

# File upload
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    with st.spinner("Processing document..."):
        # **RESET VECTOR STORE ON NEW UPLOAD**
        st.session_state.vector_store = None  

        # Process the new document
        raw_docs = process_document(uploaded_pdf.getvalue())
        chunks = chunk_documents(raw_docs)
        
        # **Ensure a new vector store is created**
        st.session_state.vector_store = run_async_task(process_chunks_async, chunks, models['embeddings'])
        
        st.success("✅ New document processed successfully! Ask your questions below.")

# Chat interface
if st.session_state.vector_store is not None:
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = st.session_state.vector_store.similarity_search(user_input, k=3)
            
            response = run_async_task(generate_response_async, user_input, relevant_docs, models, prompt_template)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(response)
