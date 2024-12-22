import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "content_for_rag", "google_ai_engineer_notes.txt")

persistent_directory = os.path.join(current_dir, "db", "faiss")

# Check if the faiss vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    #Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    #Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(len(docs))
