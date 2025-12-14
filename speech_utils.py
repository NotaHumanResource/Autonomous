# speech_utils.py
import logging
import threading
import time
import os

class SpeechUtils:
    """Simplified wrapper that uses dependency injection to avoid circular imports"""
    
    def __init__(self):
        self.listening = False
        self._status_handlers = []
        self.current_status = "inactive"
        self.recognition_data = {}
        self._whisper_utils = None  # Will be injected by main.py

    def set_whisper_utils(self, whisper_utils):
        """Inject whisper_speech_utils dependency from main.py"""
        self._whisper_utils = whisper_utils
        logging.info("Whisper utils injected into SpeechUtils")

    def text_to_speech(self, text):
        """Use injected whisper_speech_utils"""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected for TTS")
            return False
        
        try:
            return self._whisper_utils.text_to_speech(text)
        except Exception as e:
            logging.error(f"TTS Error: {e}")
            return False

    
    def speech_to_text(self, max_duration=30):
        """Use injected whisper_speech_utils - SIMPLIFIED"""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected for STT")
            return None
        
        try:
            return self._whisper_utils.speech_to_text(max_duration=max_duration)
        except Exception as e:
            logging.error(f"STT Error: {e}")
            return None

    # Remove speech_to_text_fast - no longer needed  
    
    
    def _handle_status_update(self, status, data):
        """Internal handler for status updates from the speech backend."""
        try:
            self.current_status = status
            self.recognition_data = data
            
            # Log the status update
            logging.info(f"Speech utils status update: {status}")
            
            # Notify any registered handlers
            for handler in self._status_handlers:
                try:
                    handler(status, data)
                except Exception as handler_error:
                    logging.error(f"Error in individual status handler: {str(handler_error)}", exc_info=True)
                    
        except Exception as e:
            logging.error(f"Error in _handle_status_update: {str(e)}", exc_info=True)
    
    def register_status_handler(self, handler):
        """Register a function to receive status updates."""
        if callable(handler) and handler not in self._status_handlers:
            self._status_handlers.append(handler)
            return True
        return False
    
    def unregister_status_handler(self, handler):
        """Remove a previously registered status handler."""
        if handler in self._status_handlers:
            self._status_handlers.remove(handler)
            return True
        return False
    
    def start_continuous_listening(self, callback=None):
        """Use injected whisper_speech_utils for wake word detection"""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected for continuous listening")
            return False
        
        try:
            return self._whisper_utils.start_continuous_listening(callback)
        except Exception as e:
            logging.error(f"Error starting continuous listening: {e}")
            return False
        
    def stop_continuous_listening(self):
        """Use injected whisper_speech_utils"""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected")
            return False
        
        try:
            self._whisper_utils.stop_continuous_listening()
            return True
        except Exception as e:
            logging.error(f"Error stopping continuous listening: {e}")
            return False
    
    def is_listening_active(self):
        """Check if continuous listening is active"""
        if self._whisper_utils is None:
            return False
        
        try:
            return self._whisper_utils.is_listening_active()
        except Exception as e:
            logging.error(f"Error checking listening status: {e}")
            return False
    
    def pause_listening(self, should_pause=True):
        """Use injected whisper_speech_utils"""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected")
            return False
        
        try:
            self._whisper_utils.pause_listening(should_pause)
            return True
        except Exception as e:
            logging.error(f"Error pausing listening: {e}")
            return False
    
    def send_to_gemma(self, text):
        """Send text to Gemma3 model and get response."""
        if self._whisper_utils is None:
            return "Sorry, speech service is not available."
        
        try:
            return self._whisper_utils.send_to_gemma(text)
        except Exception as e:
            logging.error(f"Error sending to Gemma: {e}")
            return "Sorry, I encountered an error."
    
    def chat_with_voice(self, text):
        """Send text to Gemma3 and speak the response."""
        try:
            # Get response from Gemma3
            response = self.send_to_gemma(text)
            
            if response and "not available" not in response and "error" not in response:
                # Speak the response
                self.text_to_speech(response)
                return response
            else:
                self.text_to_speech("Sorry, I didn't get a response.")
                return None
                
        except Exception as e:
            logging.error(f"Error in chat_with_voice: {e}")
            self.text_to_speech("Sorry, I encountered an error.")
            return None
    
    def test_speech_components(self):
        """Test if speech components are working correctly and return status."""
        if self._whisper_utils is None:
            return {
                "whisper_available": False,
                "tts_available": False,
                "audio_available": False,
                "error": "whisper_speech_utils not injected - check initialization"
            }
        
        try:
            # Use the injected Whisper test function
            result = self._whisper_utils.test_whisper_integration()
            return result
        except Exception as e:
            return {
                "whisper_available": False,
                "tts_available": False,
                "audio_available": False,
                "error": str(e)
            }
    
    def get_current_status(self):
        """Get the current recognition status and data."""
        return (self.current_status, self.recognition_data)
    
    def get_wake_words(self):
        """Get the current wake words."""
        if self._whisper_utils is None:
            return ["whisper not injected"]
        
        try:
            return self._whisper_utils.get_wake_words()
        except Exception as e:
            logging.error(f"Error getting wake words: {e}")
            return ["error getting wake words"]
    
    def set_wake_words(self, wake_words):
        """Set custom wake words."""
        if self._whisper_utils is None:
            logging.error("whisper_speech_utils not injected")
            return False
        
        try:
            return self._whisper_utils.set_wake_words(wake_words)
        except Exception as e:
            logging.error(f"Error setting wake words: {e}")
            return False

# Create a singleton instance
speech_utils = SpeechUtils()