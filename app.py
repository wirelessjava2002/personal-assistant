import os
import pyttsx3
import subprocess
import shutil
import platform
import logging
import threading
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from rich.console import Console
from rich.logging import RichHandler

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("assistant")
console = Console()

class LLMType(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"

@dataclass
class AssistantConfig:
    """Configuration for the personal assistant."""
    llm_type: LLMType
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    speech_enabled: bool = True
    documents_path: str = "documents"
    documents_glob: str = "*.md"

class PersonalAssistant:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self._setup_linux_audio()
        self.speech_engine = self._initialize_speech_engine() if config.speech_enabled else None
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.qa_chain = None
        self.llm = None

    def _setup_linux_audio(self) -> None:
        """Set up audio system for Linux/GitPod environment."""
        if platform.system() == "Linux":
            try:
                commands = [
                    "sudo apt-get update",
                    "sudo apt-get install -y alsa-utils pulseaudio espeak sox",
                    "pulseaudio --start",
                    "sudo alsa force-reload",
                ]
                
                for cmd in commands:
                    process = subprocess.Popen(
                        cmd.split(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logger.warning(f"Command '{cmd}' failed: {stderr.decode()}")
                    
                subprocess.run(["amixer", "sset", "Master", "unmute"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                subprocess.run(["amixer", "sset", "Master", "100%"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                
                logger.info("Audio system configured for Linux environment")
            except Exception as e:
                logger.error(f"Failed to configure audio system: {e}")

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize Hugging Face embeddings."""
        try:
            return HuggingFaceEmbeddings(model_name=self.config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _check_system_audio(self) -> bool:
        """Check system audio configuration."""
        if platform.system() == "Linux":
            try:
                test_file = "/tmp/test.wav"
                subprocess.run([
                    "sox", "-n", test_file, 
                    "synth", "0.1", "sine", "1000"
                ], check=True)
                
                process = subprocess.run(
                    ["aplay", test_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                
                if process.returncode == 0:
                    logger.info("Audio system check passed")
                    return True
                else:
                    logger.warning(f"Audio check failed: {process.stderr.decode()}")
                    return False
                    
            except Exception as e:
                logger.warning(f"Audio system check failed: {e}")
                return False
        return True

    def _initialize_speech_engine(self) -> Optional[pyttsx3.Engine]:
        """Initialize text-to-speech engine."""
        try:
            if platform.system() == "Linux":
                engine = pyttsx3.init('espeak')
            else:
                engine = pyttsx3.init()
                
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            def test_speech():
                try:
                    engine.say("Test")
                    engine.runAndWait()
                    logger.info("Speech engine test successful")
                except Exception as e:
                    logger.error(f"Speech engine test failed: {e}")
            
            test_thread = threading.Thread(target=test_speech)
            test_thread.start()
            test_thread.join(timeout=5)
            
            return engine
                
        except Exception as e:
            logger.error(f"Failed to initialize speech engine: {e}")
            return None

    def speak_text(self, text: str) -> None:
        """Speak text with improved error handling."""
        if not self.speech_engine:
            return

        try:
            def speak_with_timeout():
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()

            speech_thread = threading.Thread(target=speak_with_timeout)
            speech_thread.start()
            speech_thread.join(timeout=30)

            if speech_thread.is_alive():
                logger.warning("Speech synthesis timed out")
                self.speech_engine.stop()
                speech_thread.join(timeout=1)

        except Exception as e:
            logger.error(f"Speech failed: {e}")
            self._setup_linux_audio()
            self.speech_engine = self._initialize_speech_engine()

    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory."""
        try:
            os.makedirs(self.config.documents_path, exist_ok=True)
            
            if not os.listdir(self.config.documents_path):
                sample_path = os.path.join(self.config.documents_path, "sample.md")
                with open(sample_path, "w") as f:
                    f.write("# Sample Document\nThis is a sample document for testing.")
            
            loader = DirectoryLoader(
                self.config.documents_path, 
                glob=self.config.documents_glob
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            splitter = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise

    def initialize_llm(self) -> Tuple[RetrievalQA, any]:
        """Initialize the LLM and QA chain."""
        try:
            if self.config.llm_type == LLMType.CLAUDE:
                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise ValueError("Missing ANTHROPIC_API_KEY in .env file")
                llm = ChatAnthropic(
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model="claude-3-sonnet-20240229",
                    temperature=0,
                    max_tokens=1000
                )
            else:  # GEMINI
                if not os.getenv("GEMINI_API_KEY"):
                    raise ValueError("Missing GEMINI_API_KEY in .env file")
                llm = ChatGoogleGenerativeAI(
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    model="gemini-1.5-flash-latest",
                    temperature=0,
                    max_output_tokens=1000
                )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            return qa_chain, llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def setup(self) -> None:
        """Set up the assistant."""
        try:
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.qa_chain, self.llm = self.initialize_llm()
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def switch_llm(self) -> None:
        """Switch between Claude and Gemini."""
        self.config.llm_type = (
            LLMType.GEMINI if self.config.llm_type == LLMType.CLAUDE 
            else LLMType.CLAUDE
        )
        self.qa_chain, self.llm = self.initialize_llm()
        logger.info(f"Switched to {self.config.llm_type.value.title()}")

    def get_model_response(self) -> str:
        """Get a standardized model response."""
        if self.config.llm_type == LLMType.CLAUDE:
            return "I am Claude 3.5 Sonnet, an AI assistant created by Anthropic."
        return "I am Gemini 1.5 Flash, an AI assistant created by Google."

    def process_query(self, question: str) -> Optional[str]:
        """Process a user query and return the response."""
        try:
            if not question.strip():
                return "Please enter a question."

            if question.lower() in ['who are you', 'what are you', 'what model are you']:
                return self.get_model_response()

            response = self.qa_chain.invoke({"query": question})
            return response.get('result', "I couldn't generate a response.")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"

def main():
    # Load environment variables
    load_dotenv(dotenv_path=".env")

    # Initialize with Gemini as default
    config = AssistantConfig(
        llm_type=LLMType.GEMINI,
        speech_enabled=True
    )
    
    try:
        assistant = PersonalAssistant(config)
        console.print("\n[bold green]Setting up your personal assistant...[/bold green]")
        assistant.setup()
        
        console.print(f"\n[bold]Welcome to your personal assistant! ({assistant.config.llm_type.value.title()})[/bold]")
        console.print("- Type 'exit' or 'quit' to end the session")
        console.print("- Type 'switch' to change between Gemini and Claude")
        
        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                
                if question.lower() in ["exit", "quit"]:
                    break
                elif question.lower() == "switch":
                    assistant.switch_llm()
                    continue
                elif not question:
                    continue

                response = assistant.process_query(question)
                if response:
                    console.print(f"\n[bold green]Assistant ({assistant.config.llm_type.value.title()}):[/bold green] {response}")
                    if assistant.speech_engine:
                        assistant.speak_text(response)

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Shutting down...[/bold yellow]")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                console.print("\n[bold red]An error occurred. Please try again.[/bold red]")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print("\n[bold red]The assistant encountered a fatal error and must shut down.[/bold red]")

if __name__ == "__main__":
    main()