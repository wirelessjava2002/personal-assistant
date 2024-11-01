import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# Load Markdown documents
def load_documents():
    loader = DirectoryLoader("documents", glob="*.md")
    documents = loader.load()
    return documents

# Split documents into manageable chunks
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")



from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize the embedding model
embeddings = OpenAIEmbeddings()

# Create a vector store for your document chunks
def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store



