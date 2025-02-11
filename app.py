import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

load_dotenv()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text):
    """Summarize the extracted PDF text using the summarization pipeline."""
    if len(text) > 1000:
        summary = summarizer(text[:3000], max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return text 

def extract_topic(text):
    """Extract the topic of the document using the first two sentences."""
    sentences = text.split(".")
    if len(sentences) > 2:
        return ". ".join(sentences[:2]).strip() + "."
    else:
        return text

def get_text_chunks(text):
    """Split text into smaller chunks for better embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Generate vector embeddings and store them using FAISS."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, model_name):
    """Create a conversation chain using the selected model."""
    # The selected model is passed here (e.g., "gemma:2b", "llama2", or "mistral")
    llm = ChatOllama(model=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Process user input and display chat history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**You:** {message.content}")
        else:
            st.write(f"**Chatbot:** {message.content}")

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_topic" not in st.session_state:
        st.session_state.pdf_topic = ""
    if "pdf_summary" not in st.session_state:
        st.session_state.pdf_summary = ""
    
    st.header("Chat with Your PDF 📚")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Settings")
        # Dropdown for model selection
        model_name = st.selectbox("Choose a model:", ["gemma:2b", "llama2", "mistral"])
        
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                # Create the conversation chain using the selected model
                st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
                
                # Extract and store document topic and summary
                st.session_state.pdf_topic = extract_topic(raw_text)
                st.session_state.pdf_summary = summarize_text(raw_text)
    
    # Display document topic and summary if available
    if st.session_state.pdf_topic:
        st.subheader("📌 Document Topic")
        st.write(st.session_state.pdf_topic)
    
    if st.session_state.pdf_summary:
        st.subheader("📝 Summary")
        st.write(st.session_state.pdf_summary)

if __name__ == '__main__':
    main()
