import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory= persistent_directory,
            embedding_function= embeddings)

#define the query
query = "What are embeddings models?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(search_type = "similarity_score_threshold",
                            search_kwargs = {"k": 3, "score_threshold": 0.3},
                            )
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
print(f"Number of documents: {len(relevant_docs)}")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")