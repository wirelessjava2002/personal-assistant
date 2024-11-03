import os
import pyttsx3
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic

# Load environment variables
load_dotenv(dotenv_path=".env")

def load_documents():
    loader = DirectoryLoader("documents", glob="*.md")
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Initialize Hugging Face embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def initialize_speech_engine():
    engine = pyttsx3.init()
    return engine

def speak_text(engine, text):
    engine.say(text)
    engine.runAndWait()

def create_qa_chain(vector_store):
    # Initialize Claude 3.5 Sonnet
    llm = ChatAnthropic(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-sonnet-20240229",
        temperature=0,
        max_tokens=1000
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}  # Number of relevant chunks to retrieve
        )
    )
    return qa_chain

def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set your ANTHROPIC_API_KEY in the .env file")
    
    # Initialize text-to-speech engine
    speech_engine = initialize_speech_engine()
    
    print("Loading and processing documents...")
    speak_text(speech_engine, "Loading and processing documents...")
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")
    
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    
    print("Initializing QA chain...")
    qa_chain = create_qa_chain(vector_store)
    
    print("\nWelcome to your personal assistant! Ask me anything about your documents.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break

        try:
            # Use the correct method to retrieve relevant documents
            #relevant_docs = vector_store.as_retriever().get_relevant_documents(question)
            relevant_docs = vector_store.as_retriever().invoke(question)


            if relevant_docs:  # Check if any documents were retrieved
                # Construct a response based on the retrieved documents
                response = qa_chain.invoke(question)
                response_text = response['result']
                print(f"\nAssistant: {response_text}")
                speak_text(speech_engine, response_text)
            else:
                # If no documents are found, fall back to the LLM's response
                fallback_response = llm.invoke(question)  # Use your ChatAnthropic instance
                fallback_text = fallback_response['result']
                print(f"\nAssistant (fallback): {fallback_text}")
                speak_text(speech_engine, fallback_text)
        except Exception as e:
            print(f"\nError: {str(e)}")



if __name__ == "__main__":
    main()
