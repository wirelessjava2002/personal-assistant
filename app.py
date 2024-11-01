import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

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


# Load API key from environment variables
load_dotenv(dotenv_path=".env")  # Explicitly load .env

# Initialize the embedding model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a vector store for your document chunks
def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store



# Create a retrieval QA chain
def create_qa_chain(vector_store):
    llm = OpenAI(temperature=0)  # Adjust temperature for randomness
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return qa_chain


# User interaction
def main():
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = create_qa_chain(vector_store)

    print("Welcome to your personal assistant! Ask me anything about your documents.")
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(question)
        print(f"Assistant: {answer}")

if __name__ == "__main__":
    main()
