import os
from typing import List
from langchain.document_loaders import (
    #CSVLoader,
    #PyPDFLoader,
    TextLoader,
    #UnstructuredFileLoader,
    #WebBaseLoader,
    #YoutubeLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.llms import Ollama # import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings 
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = "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"
QDRANT_URL = "https://c4419e74-7ef2-4ed8-a87f-7a4deab5ae86.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "zogQ_POA6M43vw51Lw0ADW7JdJL-VsMYFroWD5KKd9cg75fSbZ-IYw"
COLLECTION_NAME = "my_documents"
GOOGLE_API_KEY = "AIzaSyBWiOOTwv1xF7N1jt2U8GP74aPTN6jQ4ZY"
# Global variables to hold initialized components
embeddings = None
vectorstore = None
qa_chain = None


# Document loading functions
def load_documents(file_paths: dict) -> List:
    documents = []
    try:
        if 'txt' in file_paths:
            documents.extend(TextLoader(file_path=file_paths['txt']).load())
        if 'txt2' in file_paths:
            documents.extend(TextLoader(file_path=file_paths['txt2']).load())
        # ... (rest of your loading logic for other file types)
        return documents
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 50) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


"""

# NVIDIA Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

def get_embeddings(api_key: str, model: str = "snowflake/arctic-embed-l"):
    return NVIDIAEmbeddings(
        nvidia_api_key=api_key,
        model=model
    )
"""

# Ollama Embeddings
def get_embeddings():  # Removed api_key and model parameters
    return OllamaEmbeddings(
        model="snowflake-arctic-embed2"  # Specify the model you pulled
    )

# Qdrant vector store setup
from langchain.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

def setup_qdrant_store(
    url: str,
    api_key: str,
    collection_name: str,
    embedding_size: int,
    force_recreate: bool = False
) -> qdrant_client.QdrantClient:
    client = qdrant_client.QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=True
    )
    
    # Check if collection exists
    if force_recreate or not client.collection_exists(collection_name):
        # Delete if exists and force_recreate is True
        if force_recreate and client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            )
        )
    
    return client

# Initialize Qdrant vectorstore
def initialize_vectorstore(
    client,
    collection_name: str,
    embeddings,
    texts,
    metadatas=None
):
    return Qdrant.from_texts(
        texts=[doc.page_content for doc in texts],
        embedding=embeddings,
        url="https://c4419e74-7ef2-4ed8-a87f-7a4deab5ae86.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="zogQ_POA6M43vw51Lw0ADW7JdJL-VsMYFroWD5KKd9cg75fSbZ-IYw",
        collection_name=collection_name,
        metadatas=metadatas
    )


"""
# ChatNVIDIA LLM setup
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA

def setup_qa_chain(api_key: str, retriever) -> RetrievalQA:
    llm = ChatNVIDIA(
        api_key=api_key,
        model="meta/llama3-8b-instruct",
        temperature=0.5
    )

    # Define the custom prompt template
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are only required to answer to the {question} dont deviate from the context nor the question."
            "You are a very helpful, clever and friendly assistant for College queries purposes."
            "You ONLY answer questions related to the college based on the provided context."
            "If a question is outside of the scope of the college information in the context, you do not provide an answer."
            "You avoid answering any queries that are not in the {context}."
        ),
        HumanMessagePromptTemplate.from_template(
            "Here is the context:\n{context}\n\nQuestion: {question}"
        )
    ])

    # Create the LLMChain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use "stuff" to pass all documents at once
        retriever=retriever,
        chain_type_kwargs={"prompt": template}
    )

    return qa_chain

"""

# Gemini LLM setup using ChatGoogleGenerativeAI
def setup_qa_chain(retriever) -> RetrievalQA:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Or another Gemini model
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        convert_system_message_to_human=True
        
    )

    # Define the custom prompt template (no changes required)
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are only required to answer to the {question} in a very friendly and welcoming manner dont deviate from the context nor the question."
            "You are a very helpful, clever and friendly assistant for College queries purposes."
            "You ONLY answer questions related to the college based on the provided context."
            "If a question is outside of the scope of the college information in the context, you do not provide an answer."
            "You avoid answering any queries that are not in the {context}."
        ),
        HumanMessagePromptTemplate.from_template(
            "Here is the context:\n{context}\n\nQuestion: {question}"
        )
    ])

    # Create the LLMChain with the custom prompt (no changes required)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": template},
        
    )

    return qa_chain

def initialize_rag(file_paths: dict):
    global embeddings, vectorstore, qa_chain

    print("Initializing RAG...")

    # 1. Load documents
    print("Loading documents...")
    documents = load_documents(file_paths)
    if not documents:
        raise Exception("No documents loaded")

    # 2. Split documents
    print("Splitting documents...")
    text_chunks = split_documents(documents)

    # 3. Initialize embeddings
    print("Initializing embeddings...")
    # If you're using NVIDIA embeddings, keep the next line. Otherwise, you'll need an alternative.
    embeddings = get_embeddings() # You will remove this line if you are not using NVIDIA API

    # 4. Setup Qdrant
    print("Setting up Qdrant...")
    client = setup_qdrant_store(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding_size=1024,  # Update this to match your embedding model's output size
        force_recreate=False
    )

    # 5. Initialize vector store
    print("Initializing vector store...")
    vectorstore = initialize_vectorstore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        texts=text_chunks
    )

    # 6. Setup retriever
    print("Setting up retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 7. Setup QA chain
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(retriever) # Removed api_key

    print("RAG initialization complete.")

def query_rag(query: str):
    global qa_chain
    if qa_chain is None:
        raise Exception("RAG system not initialized")

    try:
        response = qa_chain.invoke(query)
        return response["result"]
    except Exception as e:
        print(f"Error running query: {str(e)}")
        return "Error processing your query."