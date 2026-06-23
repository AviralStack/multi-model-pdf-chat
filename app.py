import json
import os
import sys
import boto3
import streamlit as st
import numpy as np

from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

# Groq integration
from langchain_groq import ChatGroq

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Fetch credentials securely from environment variables (Safe for Hugging Face)
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="AWS_Bedrock_doc_bot"

aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

## Bedrock Runtime Client configuration
# It will use environment variables on Hugging Face, or fall back to your local AWS CLI config
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

# Ensure the local data directory exists to store user files dynamically
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def data_ingestion():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitters.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_free_llm():
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b", 
        groq_api_key=groq_api_key, # Uses the environment variable safely
        temperature=0.3
    )
    return llm

def get_llama_llm():
    llm = BedrockLLM(
        model="meta.llama3-70b-instruct-v1:0",
        bedrock_client=bedrock_runtime,
        max_tokens=512
    )
    return llm

prompt_template = """
Human:Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with
250 words with detailed explanation. If you don't know the answer,
just say that you don't know, don't try to make up an answer. 
<context>
{context}
</context>

Question:{question}

Assistance:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Multi-Model Chat PDF", layout="wide")
    st.header("Chat with Your Uploaded PDFs")
    
    # Initialize session state keys
    if "groq_output" not in st.session_state:
        st.session_state.groq_output = ""
    if "llama_output" not in st.session_state:
        st.session_state.llama_output = ""

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    with st.sidebar:
        st.title("Document Ingestion")
        
        uploaded_files = st.file_uploader(
            "Upload your PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("Vectors Update"):
            if not uploaded_files:
                st.error("Please upload at least one PDF file first!")
            else:
                with st.spinner("Saving files and processing embeddings..."):
                    # Clear out old files from previous runs to avoid mixing datasets
                    for existing_file in os.listdir(DATA_DIR):
                        os.remove(os.path.join(DATA_DIR, existing_file))
                        
                    # Save the new user uploaded files locally
                    for uploaded_file in uploaded_files:
                        filepath = os.path.join(DATA_DIR, uploaded_file.name)
                        with open(filepath, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                    # Run standard ingestion and vector storage pipeline
                    docs = data_ingestion()
                    if not docs:
                        st.error("No text could be extracted from the uploaded documents.")
                    else:
                        get_vector_store(docs=docs)
                        st.success("Vector Store Updated Successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate with Groq"):
            if not user_question:
                st.warning("Please enter a question first.")
            elif not os.path.exists("faiss_index"):
                st.warning("No vector store found. Please upload a PDF and click 'Vectors Update' in the sidebar first.")
            else:
                with st.spinner("Processing via Groq Cloud..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = get_free_llm()
                        st.session_state.groq_output = get_response_llm(llm, faiss_index, user_question)
                        st.success("Groq Done!")
                    except Exception as e:
                        st.error(f"Error processing request: {e}")
                    
    with col2:
        if st.button("Generate with AWS Llama"):
            if not user_question:
                st.warning("Please enter a question first.")
            elif not os.path.exists("faiss_index"):
                st.warning("No vector store found. Please upload a PDF and click 'Vectors Update' in the sidebar first.")
            else:
                with st.spinner("Processing via AWS Bedrock..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = get_llama_llm()
                        st.session_state.llama_output = get_response_llm(llm, faiss_index, user_question)
                        st.success("AWS Llama Done!")
                    except Exception as e:
                        st.error(f"Error processing request: {e}")

    st.markdown("---")
    
    display_col1, display_col2 = st.columns(2)
    
    with display_col1:
        if st.session_state.groq_output:
            st.write("### Groq Output:")
            st.write(st.session_state.groq_output)
            
    with display_col2:
        if st.session_state.llama_output:
            st.write("### AWS Llama 3 Output:")
            st.write(st.session_state.llama_output)

if __name__ == "__main__":
    main()

    ## Working??