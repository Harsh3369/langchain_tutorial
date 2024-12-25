import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

#define the directory
current_directory = os.path.dirname(os.path.abspath(__file__))  
content_directory = os.path.join(current_directory, "google_whitepaper_rag_content")
db_dir = os.path.join(current_directory, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Content directory: {content_directory}")
print(f"Persistent directory: {persistent_directory}")

# Check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the content files exist
    if not os.path.exists(content_directory):
        raise FileNotFoundError(
            f"The content directory {content_directory} does not exist. Please check the path."
        )
    
    #list all the files in the content directory
    content_files = [f for f in os.listdir(content_directory) if f.endswith(".pdf")]

    # Read the text content from each file and store it with metadata
    documents = []

    # Loop through content files and process them
    for content_file in content_files:
        # Construct the full file path
        file_path = os.path.join(content_directory, content_file)
        
        # Use PyPDFLoader for PDF files
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        
        # Add metadata and store documents
        for doc in book_docs:
            doc.metadata = {"source": content_file}  # Add source metadata
            documents.append(doc)
    
    #Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size = 300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #Display informations about the the split documents
    print(f"Number of documents: {len(docs)}")
    print(f"Sample document: {docs[0].page_content}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically using CPU as device
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialize.")