import pyttsx3
import threading
import platform
import subprocess
import logging

logger = logging.getLogger("assistant")

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self._setup_linux_audio()

    def _setup_linux_audio(self):
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

    def speak_text(self, text: str):
        """Speak text using the speech engine."""
        if not self.engine:
            return

        try:
            def speak_with_timeout():
                self.engine.say(text)
                self.engine.runAndWait()

            speech_thread = threading.Thread(target=speak_with_timeout)
            speech_thread.start()
            speech_thread.join(timeout=30)

            if speech_thread.is_alive():
                logger.warning("Speech synthesis timed out")
                self.engine.stop()
                speech_thread.join(timeout=1)

        except Exception as e:
            logger.error(f"Speech failed: {e}")

    def _initialize_speech_engine(self):
        """Initialize text-to-speech engine."""
        try:
            if platform.system() == "Linux":
                engine = pyttsx3.init('espeak')
            else:
                engine = pyttsx3.init()
                
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            return engine
            
        except Exception as e:
            logger.error(f"Failed to initialize speech engine: {e}")
            return None