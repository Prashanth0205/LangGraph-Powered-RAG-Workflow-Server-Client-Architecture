from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
# from langchain.vectorstores import FAISS 
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store(docs, store_path: Optional[str] = None) -> FAISS:
    "Create a FAISS vector store from a list of documents"
    # Creating text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)

    # Embedding model and vector store 
    embedding_model = OllamaEmbeddings(model='llama3.1')
    store = FAISS.from_documents(texts, embedding_model)

    if store_path:
        store.save_local(store_path)
    
    return store

def get_local_store(store_path: str) -> FAISS:
    "Loads a localy stores FAISS vector store"
    # Load the embedding model
    embedding_model = OllamaEmbeddings(model="llama3.1")
    
    # Load the vector store from the local path
    store = FAISS.load_local(store_path, 
                             embedding_model,
                             allow_dangerous_deserialization=True
                             )

    return store