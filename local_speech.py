# Modified version of local_speech.py - Windows SAPI implementation
import win32com.client
import pythoncom
import logging
import threading
import os
import time
import queue

class LocalSpeechUtils:
    def __init__(self):
        self.engine = None
        self.recognizer = None
        self._tts_thread = None
        self._stt_thread = None
        self._continuous_listening = False
        self._speech_queue = queue.Queue()
        self._is_speaking = False
        self._listening_callback = None
        self._wake_words = ["hey gemma", "ok gemma", "hello gemma"]
        self._pause_listening = False
        self._wake_word_detected = False
        self._last_wake_time = 0
        self._wake_word_timeout = 10  # seconds after wake word is valid
        # New properties for status tracking
        self._recognition_status = "inactive"  # inactive, listening, processing
        self._status_callback = None
        self._temp_recognized_text = ""  # For showing recognition in progress
        
    
    def _init_tts(self):
        """Initialize text-to-speech using Windows SAPI default voice"""
        if self.engine is None:
            try:
                pythoncom.CoInitialize()
                self.engine = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Just use whatever is the system default - don't try to change it
                voice_name = self.engine.Voice.GetDescription()
                logging.info(f"✅ TTS ready with voice: {voice_name}")
                
                return True
                
            except Exception as e:
                logging.error(f"❌ TTS initialization failed: {e}")
                return False
        return True

    def text_to_speech(self, text):
        """Simple, reliable text-to-speech"""
        if not self._init_tts() or not text.strip():
            return False
            
        try:
            self._is_speaking = True
            
            def speak_text():
                try:
                    pythoncom.CoInitialize()
                    logging.info(f"🔊 Speaking: '{text[:50]}...'")
                    self.engine.Speak(text)
                    logging.info("✅ Speech completed")
                except Exception as e:
                    logging.error(f"❌ Speech error: {e}")
                finally:
                    self._is_speaking = False
                    pythoncom.CoUninitialize()
            
            thread = threading.Thread(target=speak_text)
            thread.daemon = True
            thread.start()
            return True
            
        except Exception as e:
            self._is_speaking = False
            logging.error(f"❌ TTS Error: {e}")
            return False
    
    def set_status_callback(self, callback):
        """Set a callback function to receive status updates."""
        self._status_callback = callback
        logging.info("Status callback set successfully")
            
    def _update_status(self, status, data=None):
        """Update recognition status and notify callback if set."""
        try:
            if data is None:
                data = {}
            
            self._recognition_status = status
            
            # Add timestamp to all status updates
            data['timestamp'] = time.time()
            
            # Log status changes (except for frequent interim results)
            if status != 'recognizing' or logging.getLogger().level <= logging.DEBUG:
                logging.info(f"Speech recognition status: {status} - {data}")
            
            # Call the status callback if set
            if self._status_callback is not None:
                try:
                    self._status_callback(status, data)
                except Exception as e:
                    logging.error(f"Error in status callback: {str(e)}", exc_info=True)
                    
        except Exception as e:
            logging.error(f"Error in _update_status: {str(e)}", exc_info=True)
    
    def text_to_speech(self, text):
        """Convert text to speech using modern Windows Speech Platform or SAPI fallback"""
        if not self._init_tts():
            return False
            
        try:
            if not text or not text.strip():
                return False
            
            # Mark that we are speaking
            self._is_speaking = True
            
            def speak_text():
                try:
                    if self.engine == "modern_speech_platform":
                        # Use modern Speech Platform via PowerShell
                        escaped_text = text.replace('"', '""')  # Escape quotes for PowerShell
                        ps_command = f"""
                        Add-Type -AssemblyName System.Speech
                        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                        $voices = $synth.GetInstalledVoices()
                        $ryanVoice = $voices | Where-Object {{ $_.VoiceInfo.Name -like "*Ryan*" }}
                        if ($ryanVoice) {{
                            $synth.SelectVoice($ryanVoice.VoiceInfo.Name)
                            Write-Output "Using voice: " + $ryanVoice.VoiceInfo.Name
                        }}
                        $synth.Speak("{escaped_text}")
                        """
                        
                        logging.info(f"Speaking with modern platform: '{text[:50]}...'")
                        subprocess.run(
                            ["powershell", "-Command", ps_command],
                            capture_output=True,
                            timeout=30
                        )
                        
                    else:
                        # Use traditional SAPI
                        pythoncom.CoInitialize()
                        logging.info(f"Speaking with SAPI: '{text[:50]}...'")
                        self.engine.Speak(text)
                        pythoncom.CoUninitialize()
                    
                    logging.info("TTS: Speech completed")
                    self._is_speaking = False
                    
                except Exception as e:
                    self._is_speaking = False
                    logging.error(f"TTS thread error: {e}")
                
            # Start speech in separate thread
            self._tts_thread = threading.Thread(target=speak_text)
            self._tts_thread.daemon = True
            self._tts_thread.start()
            return True
            
        except Exception as e:
            self._is_speaking = False
            logging.error(f"TTS Error: {e}")
            return False
    
    def speech_to_text(self, timeout=10):
        """One-time speech to text conversion using Windows SAPI"""
        if not self._init_stt():
            return None
            
        debug_info = {"stage": "initializing"}
        try:
            # Initialize COM in this thread
            pythoncom.CoInitialize()
            debug_info["stage"] = "com_initialized"
            
            # Create context
            context = self.recognizer.CreateRecoContext()
            debug_info["stage"] = "context_created"
            
            # Create grammar
            grammar = context.CreateGrammar()
            debug_info["stage"] = "grammar_created"
            
            # Set dictation mode
            try:
                grammar.DictationSetState(1)  # Enable dictation
                debug_info["stage"] = "dictation_enabled"
            except Exception as dict_err:
                logging.error(f"Failed to enable dictation: {dict_err}")
                debug_info["dictation_error"] = str(dict_err)
                pythoncom.CoUninitialize()
                return None
            
            # Set up result variable
            result = None
            
            # Set up event handler for recognition
            class RecoContextEvents:
                def OnRecognition(self, StreamNumber, StreamPosition, RecognitionType, Result):
                    nonlocal result
                    if Result:
                        result = Result.PhraseInfo.GetText()
                        debug_info["result"] = result
                        logging.info(f"Recognition result: {result}")
            
            # Initialize COM events
            connection = win32com.client.WithEvents(context, RecoContextEvents)
            debug_info["stage"] = "events_initialized"
            
            # Wait for recognition
            start_time = time.time()
            logging.info(f"Waiting for speech input (timeout: {timeout}s)...")
            while result is None and time.time() - start_time < timeout:
                pythoncom.PumpWaitingMessages()
                time.sleep(0.1)
            
            if result:
                logging.info(f"Recognized text: {result}")
                debug_info["stage"] = "success"
                pythoncom.CoUninitialize()
                return result
            else:
                logging.warning("No speech detected within timeout")
                debug_info["stage"] = "timeout"
                pythoncom.CoUninitialize()
                return None
                
        except Exception as e:
            logging.error(f"Speech-to-text error: {str(e)}", exc_info=True)
            debug_info["error"] = str(e)
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            return None
            
        finally:
            logging.info(f"Speech recognition debug info: {debug_info}")
    
    def start_continuous_listening(self, callback=None):
        """Start continuously listening for speech in a background thread
        
        Args:
            callback: Function to call when speech is recognized (text: str) -> None
        """
        if self._continuous_listening:
            logging.info("Continuous listening already active")
            return False
            
        self._continuous_listening = True
        self._listening_callback = callback
        self._wake_word_detected = False

        
        
        # Empty the speech queue
        while not self._speech_queue.empty():
            self._speech_queue.get()
            
        # Start the listening thread
        self._stt_thread = threading.Thread(target=self._continuous_listen_loop)
        self._stt_thread.daemon = True
        self._stt_thread.start()
        
        
        logging.info("Started continuous listening thread")
        return True
    
    def stop_continuous_listening(self):
        """Stop the continuous listening thread"""
        if not self._continuous_listening:
            return
            
        self._continuous_listening = False
        if self._stt_thread and self._stt_thread.is_alive():
            # Give it a moment to stop gracefully
            time.sleep(0.5)
        
        logging.info("Stopped continuous listening")
        
    def _is_wake_word(self, text):
        """Check if the recognized text contains a wake word"""
        if not text:
            return False
            
        text = text.lower().strip()
        for wake_word in self._wake_words:
            if wake_word in text:
                logging.info(f"Wake word detected: '{wake_word}' in '{text}'")
                return True
        return False
        
    def _process_speech(self, text):
        """Process recognized speech, handling wake words and callbacks"""
        if not text or len(text) < 2:  # Avoid processing noise
            return False
            
        # Log the recognized text
        logging.info(f"Processing speech: '{text}'")
            
        # Add to the queue
        self._speech_queue.put(text)
        
        # Check for wake words
        if self._is_wake_word(text):
            # Find which wake word was detected
            detected_wake_word = None
            for wake_word in self._wake_words:
                if wake_word in text.lower():
                    detected_wake_word = wake_word
                    break
            
            logging.info(f"Wake word activated: '{detected_wake_word}'")
            self._wake_word_detected = True
            self._last_wake_time = time.time()
            
            # Update status for wake word detection
            self._update_status('wake_word', {
                'wake_word': detected_wake_word,
                'timeout': self._wake_word_timeout
            })
            
            # Give audio confirmation for wake word detection
            self.text_to_speech("Yes?")
            return True
            
        # If wake word was recently detected or we're in always listening mode,
        # call the callback with the recognized text
        current_time = time.time()
        wake_word_active = (current_time - self._last_wake_time) < self._wake_word_timeout
        
        if (self._wake_word_detected and wake_word_active) or not self._wake_words:
            if self._listening_callback:
                logging.info(f"Sending recognized text to callback: '{text}'")
                self._listening_callback(text)
                
                # Reset wake word state after processing
                if self._wake_word_detected:
                    self._wake_word_detected = False
                    # Update status back to listening for wake words
                    self._update_status('listening', {'wake_words': self._wake_words})
                    
            return True
            
        return False
        
    def _continuous_listen_loop(self):
        """Background thread function for continuous listening using Windows SAPI"""
        if not self._init_stt():
            logging.error("Failed to initialize speech recognition for continuous listening")
            self._continuous_listening = False
            self._update_status('inactive', {'error': 'Failed to initialize STT'})
            return
            
        try:
            # Initialize COM for this thread
            pythoncom.CoInitialize()
            logging.info("Continuous listening: COM initialized")
            
            # Create context
            context = self.recognizer.CreateRecoContext()
            logging.info("Continuous listening: Created recognition context")
            
            # Create grammar
            grammar = context.CreateGrammar()
            logging.info("Continuous listening: Created grammar")
            
            # Enable dictation
            grammar.DictationSetState(1)
            logging.info("Continuous listening: Dictation enabled")
            
            # Set up event handling
            recognized_text = None
            interim_text = None
            
            # Define local function to process when speech is recognized
            def process_recognition(text):
                if text and not self._is_speaking and not self._pause_listening:
                    logging.info(f"Continuous listening: Recognized text: '{text}'")
                    # Update status to recognized with final text
                    self._update_status('recognized', {'text': text})
                    # Process the speech (wake word detection, callbacks, etc.)
                    self._process_speech(text)
            
            # Set up result variable and event handling for RecoContext
            class RecoContextEvents:
                def OnRecognition(self, StreamNumber, StreamPosition, RecognitionType, Result):
                    nonlocal recognized_text
                    if Result:
                        text = Result.PhraseInfo.GetText()
                        recognized_text = text
                        logging.info(f"Speech event received: '{text}'")
                        # Update status with final recognition result
                        self._update_status('recognized', {'text': text})
                        process_recognition(text)
                
                def OnHypothesis(self, StreamNumber, StreamPosition, Result):
                    nonlocal interim_text
                    if Result:
                        # Get interim results for feedback
                        text = Result.PhraseInfo.GetText()
                        interim_text = text
                        # Update status with interim recognition
                        self._update_status('recognizing', {'partial_text': text})
                        logging.debug(f"Speech hypothesis: '{text}'")
                
                def OnFalseRecognition(self, StreamNumber, StreamPosition, Result):
                    logging.debug("False recognition detected")
            
            # Connect events
            connection = win32com.client.WithEvents(context, RecoContextEvents)
            logging.info("Continuous listening: Event handlers connected")
            
            # Update status to indicate we're ready and listening for wake words
            self._update_status('listening', {'wake_words': self._wake_words})
            
            # Visual feedback that we're ready
            logging.info("✅ Continuous listening started with Windows SAPI - Ready for wake words")
            
            # Main listening loop
            last_status_time = time.time()
            failure_count = 0
            
            while self._continuous_listening:
                try:
                    # Process Windows messages to receive events
                    pythoncom.PumpWaitingMessages()
                    
                    # Don't listen while speaking to avoid feedback
                    if self._is_speaking:
                        time.sleep(0.1)
                        if self._wake_word_detected:
                            # If was in wake word mode, update status to show we're paused
                            self._update_status('paused', {'reason': 'speaking'})
                        continue
                        
                    # Skip listening if paused
                    if self._pause_listening:
                        time.sleep(0.1)
                        if self._wake_word_detected:
                            # If was in wake word mode, update status to show we're paused
                            self._update_status('paused', {'reason': 'manual_pause'})
                        continue
                    
                    # Check if wake word mode is active and if it's still valid
                    current_time = time.time()
                    if self._wake_word_detected:
                        time_since_wake = current_time - self._last_wake_time
                        # If wake word timeout has expired, go back to passive listening
                        if time_since_wake > self._wake_word_timeout:
                            logging.info(f"Wake word timeout expired after {time_since_wake:.1f} seconds")
                            self._wake_word_detected = False
                            # Update status to show we're back to passive listening
                            self._update_status('listening', {'wake_words': self._wake_words})
                    
                    # Periodically log status to confirm the loop is still running
                    if current_time - last_status_time > 30:  # Every 30 seconds
                        logging.info("Continuous listening is active and waiting for speech...")
                        # Refresh the listening status
                        if self._wake_word_detected:
                            time_left = self._wake_word_timeout - (current_time - self._last_wake_time)
                            self._update_status('wake_word', {
                                'timeout_remaining': time_left,
                                'active': True
                            })
                        else:
                            self._update_status('listening', {'wake_words': self._wake_words})
                        last_status_time = current_time
                    
                    # Check if we received recognized text
                    if recognized_text:
                        logging.info(f"Processing recognized text: '{recognized_text}'")
                        recognized_text = None  # Reset for next recognition
                    
                    # Check if interim text has been updated
                    if interim_text and self._wake_word_detected:
                        # Only log detailed interim results when in wake word mode
                        # to reduce log spam during passive listening
                        logging.debug(f"Interim text: '{interim_text}'")
                    
                    # Brief sleep to avoid high CPU usage
                    time.sleep(0.05)
                    
                except Exception as e:
                    logging.error(f"Error in continuous listening loop: {e}")
                    failure_count += 1
                    self._update_status('error', {'error': str(e), 'count': failure_count})
                    
                    if failure_count > 5:
                        logging.error("Too many failures in listening loop, restarting recognition...")
                        # Try to recreate context and grammar
                        try:
                            context = self.recognizer.CreateRecoContext()
                            grammar = context.CreateGrammar()
                            grammar.DictationSetState(1)
                            connection = win32com.client.WithEvents(context, RecoContextEvents)
                            logging.info("Successfully reset recognition components")
                            failure_count = 0
                            self._update_status('listening', {'wake_words': self._wake_words, 'reset': True})
                        except Exception as reset_error:
                            logging.error(f"Failed to reset recognition: {reset_error}")
                            self._update_status('error', {'error': str(reset_error), 'terminal': True})
                    
                    time.sleep(0.5)  # Longer delay on error
                    
        except Exception as e:
            logging.error(f"Fatal error in continuous listening: {e}")
            self._continuous_listening = False
            self._update_status('inactive', {'error': str(e)})
            
        finally:
            logging.info("Continuous listening thread ending")
            self._update_status('inactive', {'reason': 'thread_ended'})
            try:
                pythoncom.CoUninitialize()
            except:
                pass
    
    def get_speech_if_available(self):
        """Get recognized speech if available, non-blocking
        
        Returns:
            str or None: Recognized speech text if available, None otherwise
        """
        try:
            if not self._speech_queue.empty():
                return self._speech_queue.get_nowait()
            return None
        except queue.Empty:
            return None
            
    def pause_listening(self, should_pause=True):
        """Temporarily pause or resume listening
        
        Args:
            should_pause (bool): True to pause, False to resume
        """
        self._pause_listening = should_pause
        logging.info(f"{'Paused' if should_pause else 'Resumed'} continuous listening")
    
    def is_listening_active(self):
        """Check if continuous listening is active
        
        Returns:
            bool: True if continuous listening is active
        """
        return self._continuous_listening
    
    def set_wake_words(self, wake_words):
        """Set custom wake words
        
        Args:
            wake_words (list): List of strings to use as wake words
        """
        if wake_words and isinstance(wake_words, list):
            self._wake_words = [w.lower() for w in wake_words]
            logging.info(f"Set wake words to: {self._wake_words}")
            return True
        return False
    
    def get_wake_words(self):
        """Get current wake words
        
        Returns:
            list: List of wake word strings
        """
        return self._wake_words.copy()
    
    
    
    def test_windows_sapi(self):
        """Test if Windows SAPI components are working correctly
        
        Returns:
            dict: Test results
        """
        results = {
            "tts_available": False,
            "stt_available": False,
            "error": None
        }
        
        try:
            # Initialize COM
            pythoncom.CoInitialize()
            
            # Test TTS
            try:
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                results["tts_available"] = True
                results["tts_voices"] = [voice.GetDescription() for voice in speaker.GetVoices()]
            except Exception as tts_err:
                results["tts_error"] = str(tts_err)
            
            # Test STT
            try:
                recognizer = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
                context = recognizer.CreateRecoContext()
                grammar = context.CreateGrammar()
                results["stt_available"] = True
            except Exception as stt_err:
                results["stt_error"] = str(stt_err)
                
            return results
            
        except Exception as e:
            results["error"] = str(e)
            return results
            
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass

# Create an instance
local_speech_utils = LocalSpeechUtils()