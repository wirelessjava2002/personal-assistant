import os
import pyttsx3
import subprocess
import shutil
import platform
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import threading
from rich.console import Console
from rich.logging import RichHandler

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
        self.speech_engine = self._initialize_speech_engine() if config.speech_enabled else None
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize Hugging Face embeddings."""
        try:
            return HuggingFaceEmbeddings(model_name=self.config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _check_system_audio(self) -> bool:
        """
        Check system audio configuration with detailed diagnostics.
        Returns True if the system has working audio capabilities.
        """
        system = platform.system()
        
        if system == "Linux":
            # Check if speaker-test command exists
            speaker_test = shutil.which('speaker-test')
            if not speaker_test:
                logger.warning("speaker-test not found. Installing alsa-utils might help.")
                return False

            # Test if audio device is available without producing output
            try:
                with open(os.devnull, 'w') as devnull:
                    result = subprocess.run(
                        ['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1', '-D', 'default'],
                        stdout=devnull,
                        stderr=devnull,
                        timeout=1
                    )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                logger.warning("""
                Audio system check failed. To fix this:
                1. Install ALSA utilities:
                   sudo apt-get install alsa-utils
                2. Configure your sound card:
                   sudo alsactl init
                3. Unmute audio:
                   amixer sset Master unmute
                   amixer sset Speaker unmute
                   amixer sset Headphone unmute
                """)
                return False
        
        return True

    def _initialize_speech_engine(self) -> Optional[pyttsx3.Engine]:
        """Initialize text-to-speech engine with improved error handling."""
        if not self._check_system_audio():
            logger.warning("Text-to-speech disabled due to audio system configuration issues.")
            return None

        try:
            # Initialize based on platform
            system = platform.system()
            engine = None
            
            if system == "Windows":
                engine = pyttsx3.init('sapi5')
            elif system == "Darwin":  # macOS
                engine = pyttsx3.init('nsss')
            elif system == "Linux":
                # Try espeak specifically first
                if shutil.which('espeak'):
                    try:
                        engine = pyttsx3.init('espeak')
                    except:
                        pass
                
                # Fall back to default if espeak fails
                if not engine:
                    try:
                        engine = pyttsx3.init()
                    except:
                        logger.warning("""
                        Failed to initialize espeak. To fix this:
                        1. Install espeak:
                           sudo apt-get install espeak
                        2. Make sure ALSA is properly configured:
                           sudo apt-get install alsa-utils
                           sudo alsactl init
                        """)
                        return None

            # Final fallback
            if not engine:
                engine = pyttsx3.init()

            # Test and configure the engine
            if engine:
                try:
                    voices = engine.getProperty('voices')
                    if len(voices) == 0:
                        logger.warning("No voices found for text-to-speech")
                        return None
                    
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 0.9)
                    
                    # Test the engine silently
                    engine.setProperty('volume', 0)
                    engine.say("test")
                    engine.runAndWait()
                    engine.setProperty('volume', 0.9)
                    
                    return engine
                except:
                    logger.warning("Failed to configure text-to-speech engine")
                    return None

            return None

        except Exception as e:
            logger.warning(f"Text-to-speech initialization failed: {e}")
            return None

    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory."""
        try:
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

    def get_model_response(self) -> str:
        """Get a standardized model response."""
        if self.config.llm_type == LLMType.CLAUDE:
            return "I am Claude 3.5 Sonnet, an AI assistant created by Anthropic."
        return "I am Gemini 1.5 Flash, an AI assistant created by Google."

    def get_enhanced_prompt(self, question: str) -> str:
        """Get model-specific enhanced prompt."""
        if question.lower() in ['who are you', 'what are you', 'what model are you']:
            return self.get_model_response()
        
        base_prompt = (
            "You are a helpful and friendly assistant. Provide a natural, "
            f"engaging response. Remember: {self.get_model_response()} "
            f"Question: {question}"
        )
        
        if self.config.llm_type == LLMType.GEMINI:
            return f"Context: {self.get_model_response()}\nQuestion: {question}"
        
        return base_prompt

    def speak_text(self, text: str) -> None:
        """Speak text with improved error handling."""
        if self.speech_engine:
            try:
                # Set a reasonable timeout for speech
                def speak_with_timeout():
                    self.speech_engine.say(text)
                    self.speech_engine.runAndWait()
                
                speech_thread = threading.Thread(target=speak_with_timeout)
                speech_thread.start()
                speech_thread.join(timeout=10)  # 10 second timeout
                
                if speech_thread.is_alive():
                    logger.warning("Speech synthesis timed out")
                    # Force stop the speech
                    self.speech_engine.stop()
                    speech_thread.join(timeout=1)
            
            except Exception as e:
                logger.warning(f"Failed to speak text: {e}")
                # Try to reinitialize the speech engine
                self.speech_engine = self._initialize_speech_engine()

    def setup(self) -> None:
        """Set up the assistant."""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.qa_chain, self.llm = self.initialize_llm()

    def switch_llm(self) -> None:
        """Switch between Claude and Gemini."""
        self.config.llm_type = (
            LLMType.GEMINI if self.config.llm_type == LLMType.CLAUDE 
            else LLMType.CLAUDE
        )
        self.qa_chain, self.llm = self.initialize_llm()
        logger.info(f"Switched to {self.config.llm_type.value.title()}")

    def process_query(self, question: str) -> Optional[str]:
        """Process a user query and return the response."""
        try:
            if question.lower() in ['who are you', 'what are you', 'what model are you']:
                return self.get_model_response()

            relevant_docs = self.vector_store.as_retriever().invoke(question)
            
            if relevant_docs:
                response = self.qa_chain.invoke(question)
                return response['result']
            else:
                enhanced_prompt = self.get_enhanced_prompt(question)
                response = self.llm.invoke(enhanced_prompt)
                return response.content

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"

def main():
    # Load environment variables
    load_dotenv(dotenv_path=".env")

    # Initialize with Gemini as default
    config = AssistantConfig(llm_type=LLMType.GEMINI)
    
    try:
        assistant = PersonalAssistant(config)
        console.print("\n[bold green]Setting up your personal assistant...[/bold green]")
        assistant.setup()
        
        console.print(f"\n[bold]Welcome to your personal assistant! ({assistant.config.llm_type.value.title()})[/bold]")
        console.print("- Type 'exit' or 'quit' to end the session")
        console.print("- Type 'switch' to change between Gemini and Claude")
   
        
        while True:
            question = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if question.lower() in ["exit", "quit"]:
                break
            elif question.lower() == "switch":
                assistant.switch_llm()
                continue

            response = assistant.process_query(question)
            if response:
                console.print(f"\n[bold green]Assistant ({assistant.config.llm_type.value.title()}):[/bold green] {response}")
                assistant.speak_text(response)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print("\n[bold red]The assistant encountered a fatal error and must shut down.[/bold red]")

if __name__ == "__main__":
    main()