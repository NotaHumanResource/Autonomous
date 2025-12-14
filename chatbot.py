"""Main chatbot logic with enhanced search, conversation summary storage, and curiosity."""
import datetime
import sqlite3
import logging
import os
import re
import time
import json
import uuid
import sys
from utils import calculate_tokens
from conversation_summary_manager import ConversationSummaryManager
from conversation_state import ConversationStateManager
from typing import Dict, List, Optional
from langchain_ollama import OllamaLLM
from document_reader import DocumentReader
from vector_db import VectorDB
from memory_db import MemoryDB
from curiosity import Curiosity  
from deepseek import DeepSeekEnhancer
from datetime import datetime as dt 
from config import (OLLAMA_MODEL, MODEL_PARAMS, DOCS_PATH, DB_PATH,
                   QDRANT_LOCAL_PATH, QDRANT_COLLECTION_NAME, 
                   QDRANT_USE_LOCAL, QDRANT_URL)

class Chatbot:
    """Main chatbot class with conversation summary management and curiosity."""
    
    def __init__(self):
        """Initialize the chatbot and its components, including reminder manager."""
        logging.info("Initializing Chatbot")
        try:
            # Initialize databases and readers
            self.vector_db = VectorDB()
            self.memory_db = MemoryDB()

            # Initialize flags to track LLM generation and conversation state
            # These flags prevent autonomous background tasks from interfering with active responses
            self._llm_generating = False  # True when LLM is actively generating a response
            self.conversation_in_progress = False  # True when processing user input
            logging.info("Initialized LLM generation and conversation state flags")
            
            # Set docs_path earlier so it can be used in DocumentReader initialization
            self.docs_path = DOCS_PATH
            
            # Initialize DocumentReader with chatbot reference and docs_path
            from document_reader import DocumentReader
            self.doc_reader = DocumentReader(docs_path=self.docs_path, chatbot=self)
                       
            # Initialize system prompt from file
            self._initialize_system_prompt()
            
            # Set up conversation tracking
            self.current_conversation = []
            
            # Initialize Curiosity module with self reference
            self.curiosity = Curiosity(self.memory_db, self)
            
            # Initialize DeepSeekEnhancer and enhance the system prompt
            from deepseek import DeepSeekEnhancer
            self.deepseek_enhancer = DeepSeekEnhancer(self)
            
            # Get enhanced system prompt
            self.current_system_prompt = self.deepseek_enhancer.enhance_system_prompt()
            
            # Initialize LLM only once with the enhanced system prompt
            self.llm = self._initialize_llm()

            # Initialize prompt tracking
            self._last_prompt_sent = None
            self._last_prompt_tokens = 0
            
            # Phase 2: Initialize cumulative token tracking
            self._cumulative_tokens_sent = 0
            self._last_counted_prompt = None
            self._total_tokens_all_time = 0
            self._prompt_was_sent = False 
            logging.info("Initialized prompt tracking variables with cumulative tracking")
                
            # Initialize conversation state manager
            self.conversation_manager = ConversationStateManager(DB_PATH)
            # Initialize session and auto-retrieve latest summary
            self.conversation_manager.initialize_session(auto_retrieve_summary=True)
            
            # Initialize the conversation summary manager
            self.conversation_summary_manager = ConversationSummaryManager(self)
            logging.info(f"✅ conversation_summary_manager initialized: {self.conversation_summary_manager is not None}")
            
            # Initialize reminder manager
            from reminders import ReminderManager
            self.reminder_manager = ReminderManager(DB_PATH)
            logging.info("Initialized ReminderManager")

            # Initialize cognitive state tracking
            self.current_cognitive_state = 'Neutral'  # Default state
            self.cognitive_state_history = []
            logging.info("Initialized cognitive state tracking with default state: Neutral")
            
            # ===== MEMORY COMMAND COUNTERS AND SESSION MANAGEMENT =====
            # Initialize session management for command tracking
            try:
                import uuid
                from session_manager import SessionManager
                
                # Generate a unique session ID for this chatbot instance
                self.session_id = str(uuid.uuid4())
                logging.info(f"Generated session ID for chatbot: {self.session_id}")
                
                # Create session manager with lifetime counters reference
                # Note: deepseek_enhancer already has lifetime_counters initialized
                self.session_manager = SessionManager(self.deepseek_enhancer.lifetime_counters)
                
                # Start the session and sync the session ID
                session_id = self.session_manager.start_new_session()
                self.session_id = session_id
                self.deepseek_enhancer.session_id = session_id
                
                # Verify database setup
                db_path = self.deepseek_enhancer.lifetime_counters.get_database_path()
                logging.info(f"LifetimeCounters database initialized at: {db_path}")
                
                # Test database functionality
                test_counters = self.deepseek_enhancer.lifetime_counters.get_counters()
                logging.info(f"LifetimeCounters test read successful. Sample counters: {dict(list(test_counters.items())[:5])}")
                
            except ImportError as import_err:
                logging.error(f"Failed to import session management modules: {import_err}")
                # Fallback: create a simple session ID without full session management
                import uuid
                self.session_id = str(uuid.uuid4())
                self.session_manager = None
                logging.warning("Session management disabled due to import error")
                
            except Exception as session_err:
                logging.error(f"Error initializing session management: {session_err}")
                # Fallback: create a simple session ID
                import uuid
                self.session_id = str(uuid.uuid4())
                self.session_manager = None
                logging.warning("Session management partially disabled due to initialization error")
            # ===== END MEMORY COMMAND COUNTERS SECTION =====
            
            # Initialize image processor for multimodal capabilities
            try:
                from image_processor import ImageProcessor
                self.image_processor = ImageProcessor()
                logging.info("Image Processor initialized for multimodal capabilities")
            except ImportError:
                logging.warning("Image Processor module not available. Install requirements for multimodal features.")
            except Exception as e:
                logging.error(f"Error initializing Image Processor: {e}")
            
            # ===== FINAL VERIFICATION AND LOGGING =====
            # Log successful initialization with session info
            if hasattr(self, 'session_manager') and self.session_manager:
                session_summary = self.session_manager.get_session_summary()
                logging.info(f"Chatbot initialization completed with session tracking: {session_summary}")
            else:
                logging.info(f"Chatbot initialization completed with basic session ID: {self.session_id}")
            
           
            # Verify all critical components are initialized
            critical_components = [
                'vector_db', 'memory_db', 'doc_reader', 'curiosity', 
                'deepseek_enhancer', 'llm', 'conversation_manager', 
                'conversation_summary_manager', 'reminder_manager'  
            ]

            missing_components = [comp for comp in critical_components if not hasattr(self, comp)]
            if missing_components:
                logging.error(f"Critical components missing after initialization: {missing_components}")
                raise Exception(f"Failed to initialize critical components: {missing_components}")

            logging.info("✅ Chatbot initialization completed successfully with enhanced system prompt and command tracking")
            
        except Exception as e:
            logging.error(f"Chatbot initialization error: {e}", exc_info=True)
            raise

    def get_session_memory_stats(self) -> dict:
        """
        Get current session memory command counts for display in enhanced system prompt.
        
        This provides the AI model with self-awareness of its memory usage patterns
        by tracking all command executions during the current session.
        
        Uses Streamlit session state as source of truth (same data shown in UI sidebar).
        
        Command Categories:
        - Core Memory: SEARCH, STORE, FORGET, REFLECT
        - Auxiliary: SUMMARIZE, REMINDER, DISCUSS_WITH_CLAUDE
        - Meta/Utility: WEB_SEARCH, RESEARCH_DIALOGUE, HELP
        - Self-Awareness: COGNITIVE_STATE
        
        Returns:
            dict: Complete command counts with keys matching enhanced_prompt.txt placeholders
        """
        
        # Initialize default return dictionary with all required keys
        # This ensures .format() calls never get KeyError
        default_stats = {
            # Core Memory
            'search': 0,
            'store': 0,
            'forget': 0,
            'reflect': 0,
            # Auxiliary
            'summarize': 0,
            'reminder': 0,
            'discuss': 0,
            # Meta/Utility
            'web_search': 0,
            'research_dialogue': 0,
            'help': 0,
            # Self-Awareness
            'cognitive_state': 0,
            # Total
            'total_count': 0
        }
        
        try:
            # Try to import streamlit and access session state
            # This is the same source of truth used by the UI sidebar
            import streamlit as st
            
            if not hasattr(st, 'session_state'):
                logging.warning("Streamlit session_state not available")
                return default_stats
                
            if 'memory_command_counts' not in st.session_state:
                logging.warning("memory_command_counts not in session_state")
                return default_stats
            
            # Get the session counters dictionary from Streamlit session state
            # This is the SAME data displayed in the UI sidebar
            session_counts = st.session_state.memory_command_counts
            
            # ===================================================================
            # CORE MEMORY OPERATIONS
            # ===================================================================
            # SEARCH: Get the 'search' counter (already combined in session state)
            search_count = session_counts.get('search', 0)
            
            # STORE: Memory storage operations
            store_count = session_counts.get('store', 0)
            
            # FORGET: Memory deletion operations
            forget_count = session_counts.get('forget', 0)
            
            # REFLECT: Combines regular reflect and concept reflection
            reflect_count = (
                session_counts.get('reflect', 0) + 
                session_counts.get('reflect_concept', 0)
            )
            
            # ===================================================================
            # AUXILIARY OPERATIONS
            # ===================================================================
            # SUMMARIZE: Conversation summarization
            # Note: Session state uses 'summarize_conversation' key
            summarize_count = session_counts.get('summarize_conversation', 0)
            
            # REMINDER: Reminder management operations
            reminder_count = (
                session_counts.get('reminder', 0) +
                session_counts.get('reminder_complete', 0)
            )
            
            # DISCUSS_WITH_CLAUDE: AI-to-AI communication
            discuss_count = session_counts.get('discuss_with_claude', 0)
            
            # ===================================================================
            # META/UTILITY OPERATIONS
            # ===================================================================
            # WEB_SEARCH: External web searches
            web_search_count = session_counts.get('web_search', 0)
            
            # RESEARCH_DIALOGUE: Multi-turn autonomous research
            # Note: Session state might use 'self_dialogue' key
            research_dialogue_count = (
                session_counts.get('research_dialogue', 0) +
                session_counts.get('self_dialogue', 0)
            )
            
            # HELP: Command help requests
            help_count = session_counts.get('help', 0)
            
            # ===================================================================
            # SELF-AWARENESS OPERATIONS
            # ===================================================================
            # COGNITIVE_STATE: Tracking of processing/emotional states
            cognitive_state_count = session_counts.get('cognitive_state', 0)
            
            # ===================================================================
            # CALCULATE TOTAL
            # ===================================================================
            total_count = (
                search_count + store_count + forget_count + reflect_count +
                summarize_count + reminder_count + discuss_count +
                web_search_count + research_dialogue_count + help_count +
                cognitive_state_count
            )
            
            # Log the stats for debugging
            logging.info(
                f"Session memory stats (from st.session_state) - "
                f"SEARCH: {search_count}, STORE: {store_count}, FORGET: {forget_count}, "
                f"REFLECT: {reflect_count}, SUMMARIZE: {summarize_count}, "
                f"REMINDER: {reminder_count}, DISCUSS: {discuss_count}, "
                f"WEB_SEARCH: {web_search_count}, RESEARCH_DIALOGUE: {research_dialogue_count}, "
                f"HELP: {help_count}, COGNITIVE_STATE: {cognitive_state_count}, "
                f"TOTAL: {total_count}"
            )
            
            # Return comprehensive stats dictionary
            return {
                # Core Memory
                'search': search_count,
                'store': store_count,
                'forget': forget_count,
                'reflect': reflect_count,
                # Auxiliary
                'summarize': summarize_count,
                'reminder': reminder_count,
                'discuss': discuss_count,
                # Meta/Utility
                'web_search': web_search_count,
                'research_dialogue': research_dialogue_count,
                'help': help_count,
                # Self-Awareness
                'cognitive_state': cognitive_state_count,
                # Total
                'total_count': total_count
            }
            
        except ImportError:
            # Streamlit not available
            logging.error("Could not import Streamlit for session stats")
            return default_stats
        except Exception as e:
            # Error handling: Log and return default zeros
            logging.error(f"Error getting session memory stats: {e}", exc_info=True)
            return default_stats
        
    def accumulate_prompt_tokens(self):
        """
        Accumulate tokens from the last sent prompt to the cumulative counter.
        This should be called ONCE immediately after sending a prompt to the LLM.
        Prevents double-counting by checking if this specific prompt was already counted.
        """
        try:
            # Initialize tracking variables if needed
            if not hasattr(self, '_cumulative_tokens_sent'):
                self._cumulative_tokens_sent = 0
            if not hasattr(self, '_last_counted_prompt_text'):
                self._last_counted_prompt_text = None
            
            # Get current prompt tokens
            if hasattr(self, '_last_prompt_tokens') and self._last_prompt_tokens > 0:
                current_prompt_tokens = self._last_prompt_tokens
            elif hasattr(self, '_last_prompt_sent') and self._last_prompt_sent:
                current_prompt_tokens = calculate_tokens(self._last_prompt_sent)
                self._last_prompt_tokens = current_prompt_tokens
            else:
                logging.warning("ACCUMULATE: No prompt tokens to accumulate")
                return
            
            # Check if we should accumulate
            if hasattr(self, '_prompt_was_sent') and self._prompt_was_sent:
                current_prompt_text = getattr(self, '_last_prompt_sent', '')
                
                # Only accumulate if this is a NEW prompt
                if self._last_counted_prompt_text != current_prompt_text:
                    cumulative_before = self._cumulative_tokens_sent
                    self._cumulative_tokens_sent += current_prompt_tokens
                    self._last_counted_prompt_text = current_prompt_text
                    
                    logging.critical(f"✅ TOKENS ACCUMULATED:")
                    logging.critical(f"   Added: +{current_prompt_tokens:,} tokens")
                    logging.critical(f"   Before: {cumulative_before:,} tokens")
                    logging.critical(f"   After: {self._cumulative_tokens_sent:,} tokens")
                    
                    # Reset the flag
                    self._prompt_was_sent = False
                else:
                    logging.critical(f"⏭️ SKIPPED: Prompt already counted")
            else:
                logging.critical(f"⏸️ NO ACCUMULATION: _prompt_was_sent = {getattr(self, '_prompt_was_sent', 'not set')}")
                
        except Exception as e:
            logging.error(f"ACCUMULATE: Error - {e}", exc_info=True)
    
    def update_llm_system_prompt(self, new_system_prompt):
        """Update the LLM's system prompt without reinitializing the whole model."""
        try:
            if self.llm is None:
                logging.error("Cannot update system prompt: LLM not initialized")
                return False
        
            logging.info("Updating LLM system prompt")
        
            # Update the current system prompt
            self.current_system_prompt = new_system_prompt
        
            # Update the extra_body options without reinitializing everything
            if hasattr(self.llm, 'extra_body') and isinstance(self.llm.extra_body, dict):
                self.llm.extra_body['system'] = new_system_prompt
                logging.info("Updated LLM system prompt successfully")
                return True
            else:
                # If the LLM doesn't have an extra_body attribute or it's not a dict,
                # we need to reinitialize
                logging.warning("LLM doesn't support direct prompt update, reinitializing")
                self.llm = self._initialize_llm()
                return True
            
        except Exception as e:
            logging.error(f"Error updating LLM system prompt: {e}")
            return False

    def _initialize_system_prompt(self):
        """Initialize and manage the system prompt file."""
        try:
            self.system_prompt_file = "system_prompt.txt"
            if not os.path.exists(self.system_prompt_file):
                logging.error("System prompt file not found: system_prompt.txt")
                self.current_system_prompt = "Missing System Prompt"
                return

            # Read the content from the file
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                self.current_system_prompt = f.read()
            logging.info("System prompt loaded successfully")
        except Exception as e:
            logging.error(f"Error initializing system prompt: {e}")
            self.current_system_prompt = "Missing System Prompt"  

    def _get_enhanced_prompt_template(self):
        """Get the enhanced prompt template from file."""
        try:
            enhanced_prompt_file = "enhanced_prompt.txt"
            if not os.path.exists(enhanced_prompt_file):
                logging.error(f"Enhanced prompt file not found: {enhanced_prompt_file}")
                # Use a simple fallback prompt instead of creating a new file
                return "User Query: {user_input}\n\nPlease respond to Ken with the search results or let him know if there are no results."
            
            with open(enhanced_prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logging.info(f"Successfully loaded enhanced prompt file, length: {len(content)}")
                return content
        except Exception as e:
            logging.error(f"Error reading enhanced prompt file: {e}")
            return "Error loading prompt. User Query: {user_input}"
    
    def _initialize_llm(self):
        """Initialize the LLM and let Ollama handle conversation context."""
        try:
            logging.info("🚀 LLM INITIALIZATION STARTING")
            
            # Simple initialization - let Ollama manage conversation
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                system=self.current_system_prompt,  # System prompt only
                extra_body={
                    "options": {
                        "temperature": MODEL_PARAMS.get("temperature", 0.7),
                        "num_ctx": MODEL_PARAMS.get("num_ctx", 32768),
                        "num_gpu": 99,
                        "top_k": MODEL_PARAMS.get("top_k", 40),
                        "top_p": MODEL_PARAMS.get("top_p", 0.9),
                        "num_predict": 4096,  # ← ADD THIS - limits output to 4096 tokens max
                    }
                }
            )
            
            logging.info(f"✅ LLM INITIALIZED - Context: {MODEL_PARAMS.get('num_ctx', 32768)} tokens, Max output: 4096 tokens")
            return llm
            
        except Exception as e:
            logging.error(f"❌ LLM INITIALIZATION FAILED: {e}")
            raise
    
    def log_reminder_operation(self, operation_type, reminder_id, content=None, status="completed", error=None):
        """
        Log reminder operations for tracking and debugging.
        
        Args:
            operation_type (str): Type of operation ("create", "complete", "delete")
            reminder_id: The ID of the reminder
            content (str, optional): Content of the reminder
            status (str): Status of the operation ("starting", "completed", "failed")
            error (str, optional): Error message if operation failed
        """
        try:
            # Format the log prefix based on operation type
            prefix = f"[REMINDER {operation_type.upper()}]"
            
            # Create the log message
            if content:
                content_preview = content[:50] + "..." if len(content) > 50 else content
                message = f"{prefix} ID={reminder_id}, Content='{content_preview}', Status={status}"
            else:
                message = f"{prefix} ID={reminder_id}, Status={status}"
                
            # Add error details if present
            if error:
                message += f", Error: {error}"
                
            # Log the message
            logging.info(message)
            
        except Exception as e:
            # Fallback logging to ensure errors in logging don't cause additional problems
            logging.error(f"Error in log_reminder_operation: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Use unified token estimation from utils."""
        return calculate_tokens(text)
        
    def update_session_counter(self, command_type: str):
        """
        Update session counter for a specific command type.
        
        Args:
            command_type (str): The type of command ('store', 'retrieve', etc.)
        """
        import sys
        if 'streamlit' in sys.modules:
            try:
                import streamlit as st_local
                if hasattr(st_local, 'session_state') and 'memory_command_counts' in st_local.session_state:
                    # Initialize counter if it doesn't exist
                    if command_type not in st_local.session_state.memory_command_counts:
                        st_local.session_state.memory_command_counts[command_type] = 0
                    
                    # Increment the counter
                    st_local.session_state.memory_command_counts[command_type] += 1
                    logging.info(f"Updated {command_type} counter: {st_local.session_state.memory_command_counts[command_type]}")
                    return True
                else:
                    logging.debug(f"Could not update {command_type} counter - session_state not available")
                    return False
            except (ImportError, ModuleNotFoundError, Exception) as e:
                logging.debug(f"Could not update {command_type} counter: {e}")
                return False
        else:
            logging.debug("Streamlit not available for counter update")
            return False

    def initialize_session_counters(self):
        """Initialize all session counters to 0."""
        import sys
        if 'streamlit' in sys.modules:
            try:
                import streamlit as st_local
                if hasattr(st_local, 'session_state'):
                    if 'memory_command_counts' not in st_local.session_state:
                        st_local.session_state.memory_command_counts = {}
                    
                    # Initialize all counter types
                    counter_types = ['store', 'retrieve', 'forget', 'reflect', 'correct', 'reminder', 'summarize', 'discuss_with_claude']
                    
                    for counter_type in counter_types:
                        if counter_type not in st_local.session_state.memory_command_counts:
                            st_local.session_state.memory_command_counts[counter_type] = 0
                    
                    logging.info("Initialized session counters")
                    return True
            except Exception as e:
                logging.error(f"Error initializing session counters: {e}")
                return False
        return False

    def get_conversation_length(self):
        """Get the current conversation length from the primary source."""
        return len(self.current_conversation)

    def get_conversation_messages(self):
        """Get conversation messages from the primary source."""
        return self.current_conversation.copy()  # Return a copy to prevent external modification
    
    #For Debugging can remove:
    def verify_conversation_history(self):
        """
        Verify that conversation history is being properly maintained.
        """
        try:
            total_messages = len(self.current_conversation)
            
            # Count by role
            user_count = sum(1 for msg in self.current_conversation if msg.get('role') == 'user')
            assistant_count = sum(1 for msg in self.current_conversation if msg.get('role') == 'assistant')
            system_count = sum(1 for msg in self.current_conversation if msg.get('role') == 'system')
            
            # Calculate conversation tokens
            total_tokens = sum(self._estimate_tokens(msg.get('content', '')) 
                            for msg in self.current_conversation)
            
            max_tokens = MODEL_PARAMS.get('num_ctx', 32768)
            percentage = (total_tokens / max_tokens) * 100
            
            logging.critical(f"🔍 CONVERSATION VERIFICATION:")
            logging.critical(f"  Total messages: {total_messages}")
            logging.critical(f"  User: {user_count}, Assistant: {assistant_count}, System: {system_count}")
            logging.critical(f"  Total tokens: ~{total_tokens:,} / {max_tokens:,} ({percentage:.1f}%)")
            
            # Verify the last few messages exist
            if total_messages >= 5:
                logging.critical(f"  Last 5 messages:")
                for i, msg in enumerate(self.current_conversation[-5:], start=total_messages-4):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:30]
                    logging.critical(f"    [{i}] {role}: {content}...")
            
            return {
                'total_messages': total_messages,
                'user_count': user_count,
                'assistant_count': assistant_count,
                'system_count': system_count,
                'total_tokens': total_tokens,
                'percentage': percentage
            }
            
        except Exception as e:
            logging.error(f"Error in verify_conversation_history: {e}")
            return None
    
    def process_command(self, user_input: str, indicators: Dict) -> str:
        """
        Process user command with enhanced prompt template, memory commands, and auto-summarization.
        
        This method handles:
        1. Token tracking and auto-summarization at 85% threshold
        2. Enhanced prompt formatting with context
        3. Two-pass retrieval system (initial + search results)
        4. Memory command processing
        5. Response cleanup
        6. Recursion trap prevention for meta-cognitive questions
        7. Cognitive state tracking (rate-limited to 1 update per turn)
        
        Args:
            user_input (str): The user's input text
            indicators (Dict): UI indicators for status display
            
        Returns:
            str: The model's response text
        """
        start_time = time.time()
        logging.info(f"Starting process_command at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        try:
            # =====================================================
            # STEP 0: Reset per-turn flags for new user turn
            # =====================================================
            
            # --- Reset recursion detector ---
            # This allows ONE meta-cognitive storage per response cycle
            # while preventing infinite loops from repeated identical stores
            #
            # IMPORTANT: This does NOT limit the total number of stores per turn.
            # It only prevents the SAME content from being stored repeatedly.
            # 
            # Examples of what's ALLOWED:
            # - Multiple different [STORE:] commands in one response
            # - Storing search results from multiple searches
            # - Storing various insights about different topics
            #
            # Examples of what's BLOCKED:
            # - Storing "insight about loops" 4+ times in one response
            # - Any identical content repeated 4+ times
            
            if hasattr(self, 'deepseek_enhancer') and hasattr(self.deepseek_enhancer, '_recursion_detector'):
                # Only reset if we're not in cooldown (cooldown must persist across turns)
                if not self.deepseek_enhancer._recursion_detector.get('cooldown_until'):
                    self.deepseek_enhancer._recursion_detector['duplicate_count'] = 0
                    self.deepseek_enhancer._recursion_detector['last_store_content'] = None
                    self.deepseek_enhancer._recursion_detector['last_command_type'] = None
                    logging.debug("RECURSION_DETECTOR: Reset for new user turn (allows 1 meta-storage)")
                else:
                    # Still in cooldown from previous recursion trap
                    cooldown_remaining = (
                        self.deepseek_enhancer._recursion_detector['cooldown_until'] - 
                        datetime.datetime.now()
                    ).seconds
                    logging.warning(
                        f"RECURSION_DETECTOR: Still in cooldown ({cooldown_remaining}s remaining) "
                        "- meta-cognitive storage blocked"
                    )
            
            # --- Reset cognitive state rate limiter ---
            # Allows the model to update its cognitive state ONCE per conversation turn
            # Prevents spam like changing from Frustrated → Happy → Frustrated → Curious in one response
            if hasattr(self, 'deepseek_enhancer') and hasattr(self.deepseek_enhancer, '_state_updated_this_turn'):
                self.deepseek_enhancer._state_updated_this_turn = False
                logging.debug("COGNITIVE_STATE: Reset rate limiter for new turn (allows 1 state update)")
                
            # =====================================================
            # STEP 1: Get token count for monitoring
            # =====================================================
            # Note: Auto-summarization is handled by main.py after response generation

            # Get unified token count for monitoring and dashboard
            current_tokens, max_tokens, percentage = self.get_unified_token_count()
            logging.info(f"ENHANCED: Token usage: {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")

            # =====================================================
            # STEP 2: Build conversation context and check for meta-questions
            # =====================================================
            
            # Get enhanced prompt template (loads from enhanced_prompt.txt)
            enhanced_prompt_template = self._get_enhanced_prompt_template()
            
            # FIXED: Get conversation context from Streamlit messages (the actual source of truth)
            convo_context = ""
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and 'messages' in st.session_state:
                    # Get conversation messages from Streamlit (UI source of truth)
                    messages = st.session_state.messages
                    
                    # DUPLICATE PREVENTION: Exclude the current user message from convo_context
                    # The current user message is already included via {user_input} in the template
                    # Including it in convo_context would cause it to appear twice in the prompt
                    if (messages and 
                        messages[-1].get('role') == 'user' and 
                        messages[-1].get('content', '').strip() == user_input.strip()):
                        # Exclude the last message (current user input) from history
                        messages_to_format = messages[:-1]
                        logging.debug(f"CONVO_CONTEXT: Excluding current user message to prevent duplication")
                    else:
                        messages_to_format = messages
                    
                    # Format conversation context preserving original format_conversation_context logic
                    formatted = []
                    total_tokens = 0
                
                    # Use 70% of context for conversation history
                    max_context_tokens = int(MODEL_PARAMS.get("num_ctx", 32768) * 0.7)  # 22,937 tokens
                    
                    # Start from most recent and work backwards to fit in token limit
                    for i in range(len(messages_to_format) - 1, -1, -1):
                        msg = messages[i]
                        role = msg.get('role', '').lower()  # CHANGED: lowercase for ChatML tags
                        content = msg.get('content', '')
                        
                        if not content:
                            continue
                        
                        # Format the message with proper ChatML tags for Qwen3 model compatibility
                        # This ensures the model correctly identifies speaker roles in conversation history
                        formatted_msg = f"<|im_start|>{role}\n{content.strip()}<|im_end|>"
                        msg_tokens = self._estimate_tokens(formatted_msg)
                        
                        # Check if adding this message would exceed token limit
                        if total_tokens + msg_tokens > max_context_tokens:
                            logging.info(f"Reached token limit at message {len(messages) - i}, stopping context building")
                            break
                            
                        formatted.insert(0, formatted_msg)  # Insert at beginning since we're going backwards
                        total_tokens += msg_tokens
                    
                    convo_context = "\n".join(formatted)
                    logging.info(f"ENHANCED: Formatted {len(formatted)} messages, {total_tokens:,} tokens for context")
                    
                else:
                    logging.warning("ENHANCED: No Streamlit messages available for context")
            except Exception as context_error:
                logging.error(f"ENHANCED: Error getting conversation context: {context_error}")
                convo_context = ""
            
            # Get current time and date for temporal awareness
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%A, %B %d, %Y")
            
            # ⭐ NEW: Check for meta-questions that could trigger recursion
            meta_safety_instruction = ""
            if self._detect_meta_question(user_input):
                logging.warning("META_QUESTION: Detected potentially recursive question - adding safety instructions")
                
                # ⭐ UPDATED: Allow limited meta-cognitive storage with safeguards
                meta_safety_instruction = """
                ⚠️ META-COGNITIVE QUESTION DETECTED: This question is about your own processing.

                BALANCED APPROACH TO SELF-REFLECTION:
                1. You MAY use ONE [STORE:] command to capture a genuine meta-cognitive insight
                2. You MAY use ONE [REFLECT:] command if it provides developmental value
                3. DO NOT store insights about "storing insights" (that creates infinite loops)
                4. DO NOT reflect on "the act of reflecting" (that triggers recursion)
                5. Focus your storage on WHAT you learned, not HOW you're learning it right now

                SAFE META-STORAGE EXAMPLE:
                ✅ [STORE: When asked about loop behavior, I recognized that self-referential questions require careful boundary setting | type=insight | confidence=0.8]

                UNSAFE META-STORAGE EXAMPLE (WILL LOOP):
                ❌ [STORE: I'm storing an insight about recognizing recursive patterns while recognizing recursive patterns...]

                Remember: Store the CONCLUSION of your meta-analysis, not the PROCESS of analyzing.
                """
            
            # =====================================================
            # STEP 3: Format the enhanced prompt using template system
            # =====================================================
            try:
                # Get token usage warning if needed (shows at 85%+)
                token_warning = self.get_token_usage_warning()
                
                # Get session memory command stats for display
                session_stats = self.get_session_memory_stats()
                
                # For models internal memory command counter. Format the enhanced prompt with memory command stats
                enhanced_prompt = enhanced_prompt_template.format(
                    memory_context="",  # Empty for first pass, filled in second pass
                    convo_context=convo_context,
                    user_input=user_input,
                    token_usage=f"{current_tokens}/{max_tokens} ({percentage:.1f}%)",
                    token_warning=token_warning,
                    current_time=current_time,
                    current_date=current_date,
                    # Core Memory Command Counts
                    search_count=session_stats['search'],
                    store_count=session_stats['store'],
                    forget_count=session_stats['forget'],
                    reflect_count=session_stats['reflect'],
                    # Auxiliary Command Counts
                    summarize_count=session_stats['summarize'],
                    reminder_count=session_stats['reminder'],
                    discuss_count=session_stats['discuss'],
                    # Meta/Utility Command Counts
                    web_search_count=session_stats['web_search'],
                    research_dialogue_count=session_stats['research_dialogue'],
                    help_count=session_stats['help'],
                    cognitive_state_count=session_stats['cognitive_state'],
                    # Total Operations
                    total_count=session_stats['total_count']
                )

                # ⭐ NEW: Prepend safety instructions if this is a meta-question
                if meta_safety_instruction:
                    enhanced_prompt = meta_safety_instruction + "\n\n" + enhanced_prompt

                logging.info(f"ENHANCED: Formatted prompt length: {len(enhanced_prompt)}")
                            
            except Exception as format_error:
                logging.error(f"Error formatting enhanced prompt: {format_error}")
                # Fallback to simple format if template formatting fails
                enhanced_prompt = f"<|im_start|>system\n{self.current_system_prompt}<|im_end|>\n{convo_context}\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant"

            # =====================================================
            # STEP 4: Get initial response from LLM
            # =====================================================
            logging.info("ENHANCED: Getting initial LLM response with conversation context")
            
            # CRITICAL FIX: Store the prompt AND calculate tokens BEFORE the LLM call
            # This ensures our token tracking is accurate
            self._last_prompt_sent = enhanced_prompt
            self._last_prompt_tokens = calculate_tokens(enhanced_prompt)
            logging.info(f"TOKEN_TRACKING: Stored prompt - {len(enhanced_prompt):,} chars, {self._last_prompt_tokens:,} tokens")
            
            try:
                # === ENHANCED DIAGNOSTIC LOGGING START ===
                logging.critical(f"🔥 LLM INVOKE STARTING - Timestamp: {datetime.datetime.now().isoformat()}")
                
                # Log prompt characteristics
                prompt_length = len(enhanced_prompt)
                prompt_tokens = calculate_tokens(enhanced_prompt)
                logging.critical(f"📝 Prompt length: {prompt_length:,} characters")
                logging.critical(f"🔢 Estimated prompt tokens: {prompt_tokens:,}")
                
                # Log context window state
                current_tokens, max_tokens, percentage = self.get_unified_token_count()
                logging.critical(f"📊 CONTEXT STATE BEFORE LLM:")
                logging.critical(f"   Current tokens: {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")
                logging.critical(f"   Conversation messages: {len(self.current_conversation)}")
                
                # Log prompt beginning and end for pattern detection
                logging.critical(f"📄 Prompt START (first 200 chars):")
                logging.critical(f"   {enhanced_prompt[:200]}")
                logging.critical(f"📄 Prompt END (last 200 chars):")
                logging.critical(f"   {enhanced_prompt[-200:]}")
                
                # Check for potentially problematic patterns
                search_count = enhanced_prompt.count('[SEARCH:')
                retrieve_count = enhanced_prompt.count('[RETRIEVE:')
                store_count = enhanced_prompt.count('[STORE:')
                
                if search_count > 0:
                    logging.critical(f"⚠️ PATTERN: [SEARCH:] appears {search_count} times in prompt")
                if retrieve_count > 0:
                    logging.critical(f"⚠️ PATTERN: [RETRIEVE:] appears {retrieve_count} times in prompt")
                if store_count > 0:
                    logging.critical(f"⚠️ PATTERN: [STORE:] appears {store_count} times in prompt")
                
                # Start timing the LLM call
                llm_start_time = time.time()
                logging.critical(f"⏱️ Starting LLM invoke at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                
                # === ACTUAL LLM CALL (NO TIMEOUT) ===
                response = self.llm.invoke(enhanced_prompt)
                self._prompt_was_sent = True  # Set flag immediately after LLM call
                logging.critical(f"🚀 LLM INVOKE #1 COMPLETE - Set _prompt_was_sent=True")
                
                # === LOG COMPLETION ===
                llm_duration = time.time() - llm_start_time
                logging.critical(f"✅ LLM INVOKE COMPLETED")
                logging.critical(f"⏱️ Duration: {llm_duration:.2f} seconds")
                logging.critical(f"📏 Response length: {len(response) if response else 0} characters")
                # === ENHANCED DIAGNOSTIC LOGGING END ===
                
                if response is None:
                    return "I encountered an unexpected error. Please try again."
                
                logging.info(f"ENHANCED: Got response length: {len(response)}")
                original_response = response
                
            except Exception as e:
                llm_duration = time.time() - llm_start_time if 'llm_start_time' in locals() else 0
                logging.critical(f"❌ LLM INVOKE FAILED")
                logging.critical(f"⏱️ Time before failure: {llm_duration:.2f} seconds")
                logging.critical(f"❌ Error type: {type(e).__name__}")
                logging.critical(f"❌ Error message: {str(e)}")
                logging.critical(f"📊 Context state at failure: {current_tokens:,}/{max_tokens:,} tokens")
                return f"I encountered an issue processing your request: {str(e)}"

            # =====================================================
            # STEP 5: Process memory commands in the response
            # =====================================================
            logging.info("ENHANCED: Processing memory commands")
            processed_response = response
            commands_processed = 0
            
            try:
                if hasattr(self, 'deepseek_enhancer'):
                    # Process any [STORE:], [RETRIEVE:], [SEARCH:] commands in the response
                    # This is where recursion detection will activate if duplicate commands are found
                    processed_response, commands_processed = self.deepseek_enhancer.process_response(response)
                    logging.info(f"ENHANCED: Processed {commands_processed} memory commands")
                    
                    if processed_response is None:
                        processed_response = response
                        
            except Exception as process_error:
                logging.error(f"Error processing memory commands: {process_error}")
                processed_response = response

            # =====================================================
            # STEP 6: Handle retrieval commands with second pass
            # =====================================================
            
            # Check if response contains retrieval commands that would have generated search results
            retrieval_pattern = re.compile(r'\[(RETRIEVE|PRECISE_SEARCH|SEARCH|COMPREHENSIVE_SEARCH|EXACT_SEARCH):\s*(.*?)\s*\]', re.IGNORECASE)
            retrieval_commands = retrieval_pattern.findall(response)

            # Only do second pass if:
            # 1. We found retrieval commands
            # 2. Commands were actually processed (commands_processed > 0)
            # 3. The processed response is different from original (meaning results were added)
            if retrieval_commands and commands_processed > 0 and processed_response != response:
                logging.critical(f"🔄 SECOND PASS: Found {len(retrieval_commands)} retrieval commands, preparing second LLM call")
                
                # Extract search results for second pass
                search_results_pattern = re.compile(r'\*\*=====\s*(?:SEARCH|MEMORY RETRIEVAL|EXACT SEARCH).*?\*\*===== END OF (?:SEARCH|MEMORY RETRIEVAL|EXACT SEARCH) =====\*\*', re.DOTALL)
                search_results_sections = search_results_pattern.findall(processed_response)
                
                if search_results_sections:
                    search_results_text = "\n\n".join(search_results_sections)
                    logging.critical(f"🔄 SECOND PASS: Extracted {len(search_results_sections)} search result sections ({len(search_results_text)} chars)")
                    
                    # Second pass using SPECIALIZED template with search results
                    logging.info("🔄 SECOND PASS: Building specialized second-pass prompt")
                    
                    # Get token usage warning if needed (may have changed after first pass)
                    token_warning = self.get_token_usage_warning()
                    
                    # Get updated session stats (commands may have been executed in first pass)
                    session_stats = self.get_session_memory_stats()

                    # ═══════════════════════════════════════════════════
                    # CRITICAL FIX: Use read-only method to get current stats
                    # The first pass tokens were already counted when llm.invoke()
                    # completed. We just need the current state for display
                    # in the second pass prompt - we do NOT want to accumulate again.
                    # ═══════════════════════════════════════════════════
                    
                    # Check if read-only method is available (for backward compatibility)
                    if hasattr(self, 'get_token_stats_readonly'):
                        # Use the read-only method (safe, no accumulation)
                        current_tokens, max_tokens, percentage = self.get_token_stats_readonly()
                        logging.info(f"🔄 SECOND PASS: Using read-only stats - {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")
                    else:
                        # Fallback: Get stats manually without calling get_unified_token_count()
                        max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
                        cumulative = getattr(self, '_cumulative_tokens_sent', 0)
                        
                        if cumulative <= max_tokens:
                            current_tokens = cumulative
                        else:
                            current_tokens = int(max_tokens * 0.92)
                        
                        percentage = (current_tokens / max_tokens) * 100
                        logging.info(f"🔄 SECOND PASS: Calculated stats (fallback) - {current_tokens:,}/{max_tokens:,} ({percentage:.1f}%)")

                    # Get specialized second-pass template with search results injected
                    # and updated memory command statistics
                    second_pass_prompt = enhanced_prompt_template.format(
                        memory_context=search_results_text,  # NOW POPULATED with search results
                        convo_context=convo_context,
                        user_input=user_input,
                        token_usage=f"{current_tokens}/{max_tokens} ({percentage:.1f}%)",  # FRESH VALUES
                        token_warning=token_warning,
                        current_time=current_time,
                        current_date=current_date,
                        # Core Memory Command Counts
                        search_count=session_stats['search'],
                        store_count=session_stats['store'],
                        forget_count=session_stats['forget'],
                        reflect_count=session_stats['reflect'],
                        # Auxiliary Command Counts
                        summarize_count=session_stats['summarize'],
                        reminder_count=session_stats['reminder'],
                        discuss_count=session_stats['discuss'],
                        # Meta/Utility Command Counts
                        web_search_count=session_stats['web_search'],
                        research_dialogue_count=session_stats['research_dialogue'],
                        help_count=session_stats['help'],
                        cognitive_state_count=session_stats['cognitive_state'],
                        # Total Operations
                        total_count=session_stats['total_count']
                    )

                    # CRITICAL: Update stored prompt and tokens for accurate tracking
                    self._last_prompt_sent = second_pass_prompt
                    self._last_prompt_tokens = calculate_tokens(second_pass_prompt)
                    logging.critical(f"🔄 SECOND PASS: Updated tracking - {len(second_pass_prompt):,} chars, {self._last_prompt_tokens:,} tokens")
                    
                    # Log what we're sending to the model (for debugging)
                    logging.critical(f"🔄 SECOND PASS PROMPT Preview (first 500 chars):\n{second_pass_prompt[:500]}")
                    logging.critical(f"🔄 SECOND PASS PROMPT Preview (last 300 chars):\n{second_pass_prompt[-300:]}")

                    # Safety check: Only proceed if we have room in context window
                    # Use 98% threshold to leave a small safety margin
                    if self._last_prompt_tokens < MODEL_PARAMS["num_ctx"] * 0.98:
                        try:
                            logging.critical("🔄 SECOND PASS: STARTING second LLM call NOW...")
                            start_second = time.time()
                            
                            # Make the second LLM call with search results (NO TIMEOUT)
                            final_response = self.llm.invoke(second_pass_prompt)
                            self._prompt_was_sent = True  # Set flag immediately after LLM call
                            logging.critical(f"🚀 LLM INVOKE #2 COMPLETE - Set _prompt_was_sent=True")
                            
                            elapsed_second = time.time() - start_second
                            logging.critical(f"🔄 SECOND PASS: COMPLETED in {elapsed_second:.2f}s, response length: {len(final_response) if final_response else 0}")
                            
                            if final_response:
                                response = final_response
                                logging.info("ENHANCED: Completed two-pass retrieval process successfully")

                        except Exception as e:
                            elapsed_second = time.time() - start_second if 'start_second' in locals() else 0
                            logging.error(f"🔄 SECOND PASS: FAILED after {elapsed_second:.2f}s: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fall back to processed response from first pass
                            response = processed_response
            
            # =====================================================
            # STEP 7: Clean search results from final response
            # =====================================================

            
            def clean_search_results_from_response(response_text):
                """
                Remove internal result blocks from response to prevent context accumulation.
                
                These blocks are useful for the model during processing but should not
                be displayed to the user in the final response. The user has access to
                a complete HTML command guide in the UI.
                """
                # Pattern to match all internal result blocks using ===== delimiters
                # Uses flexible matching for variations in block names
                search_patterns = [
                    # Search result variants
                    r'\*\*=====\s*(?:COMPREHENSIVE\s+)?(?:SELECTIVE\s+)?(?:PRECISE\s+)?(?:EXACT\s+)?(?:MATCH\s+)?SEARCH\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*',
                    r'\*\*=====\s*SEARCH\s+RESULTS?:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+SEARCH\s*=====\*\*',
                    
                    # Memory retrieval
                    r'\*\*=====\s*MEMORY\s+RETRIEVAL\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+MEMORY\s+RETRIEVAL\s*=====\*\*',
                    
                    # Document summaries
                    r'\*\*=====\s*DOCUMENT\s+SUMMARIES?\s+SEARCH\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:SEARCH|DOCUMENT\s+SUMMARIES?)\s*=====\*\*',
                    
                    # Conversation summaries
                    r'\*\*=====\s*(?:LATEST\s+)?CONVERSATION\s+SUMMAR(?:Y|IES).*?=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:LATEST\s+SUMMARY|CONVERSATION\s+SUMMARIES?)\s*=====\*\*',
                    
                    # Reminder search results
                    r'\*\*=====\s*REMINDER\s+SEARCH\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:REMINDER\s+SEARCH|SEARCH)\s*=====\*\*',
                    
                    # AI-to-AI discussion (DISCUSS_WITH_CLAUDE)
                    r'\*\*=====\s*AI-TO-AI\s+DISCUSSION:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+DISCUSSION\s*=====\*\*',
                    
                    # External research dialogue (WEB_SEARCH)
                    r'\*\*=====\s*EXTERNAL\s+RESEARCH\s+DIALOGUE:.*?=====\*\*.*?\*\*=====\s*END\s+OF\s+EXTERNAL\s+RESEARCH\s+DIALOGUE\s*=====\*\*',
                    
                    # Image analysis
                    r'\*\*=====\s*IMAGE\s+ANALYSIS\s+RESULTS?\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+IMAGE\s+ANALYSIS\s*=====\*\*',
                    
                    # Command guide / help
                    r'\*\*=====\s*(?:INTERNAL\s+)?COMMAND\s+(?:GUIDE|REFERENCE)\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+COMMAND\s+(?:GUIDE|REFERENCE)\s*=====\*\*',
                    
                    # System prompt display
                    r'\*\*=====\s*SYSTEM\s+PROMPT\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+SYSTEM\s+PROMPT\s*=====\*\*',
                    
                    # Help blocks (search help, store help, modify prompt help)
                    r'\*\*=====\s*(?:SEARCH|STORE\s+COMMAND|MODIFY\s+SYSTEM\s+PROMPT)\s+HELP\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+(?:SEARCH\s+)?HELP\s*=====\*\*',
                    
                    # Status display
                    r'\*\*=====\s*STATUS\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+STATUS\s*=====\*\*',
                    
                    # Error blocks
                    r'\*\*=====\s*(?:CONVERSATION\s+SUMMARIES?\s+)?ERROR\s*=====\*\*.*?\*\*=====\s*END\s+OF\s+ERROR\s*=====\*\*',
                    
                    # Internal self-dialogue (uses markdown headers, not ===== pattern)
                    r'##\s*🤔\s*Internal\s+Self-Dialogue:.*?(?=\n##\s+[^🤔]|\n\*\*[A-Z]|\Z)',
                ]
                
                cleaned_response = response_text
                for pattern in search_patterns:
                    cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove multiple newlines and clean up formatting
                cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                return cleaned_response

            # Apply cleanup to final response
            original_response_length = len(response)
            response = clean_search_results_from_response(response)
            cleaned_response_length = len(response)
            
            if original_response_length != cleaned_response_length:
                logging.info(f"CLEANUP: Removed {original_response_length - cleaned_response_length} characters of search results from response")
                logging.info(f"CLEANUP: Response length: {original_response_length} → {cleaned_response_length}")

            # =====================================================
            # STEP 8: Reset UI indicators and return response
            # =====================================================
            
            # Reset indicators safely
            try:
                if hasattr(indicators['Model'], 'empty'):
                    indicators['Model'].empty()
                if hasattr(indicators['FAISS'], 'empty'):
                    indicators['FAISS'].empty()
                if hasattr(indicators['DB'], 'empty'):
                    indicators['DB'].empty()
            except Exception:
                pass

            # Log processing time
            elapsed_time = time.time() - start_time
            logging.info(f"process_command completed in {elapsed_time:.3f} seconds")
            
            return response

        except Exception as e:
            logging.error(f"Error in enhanced process_command: {e}", exc_info=True)
            elapsed_time = time.time() - start_time
            logging.info(f"process_command completed with error in {elapsed_time:.3f} seconds")
            return f"An error occurred: {str(e)}"
            

    def _detect_meta_question(self, user_input: str) -> bool:
        """
        Detect if user is asking about the AI's own processing/behavior.
        These meta-cognitive questions can trigger recursion traps.
        
        Args:
            user_input (str): The user's question/input
            
        Returns:
            bool: True if this appears to be a meta-question about AI's own behavior
        """
        try:
            # Keywords that indicate meta-cognitive questions
            meta_keywords = [
                'stuck in a loop', 'stuck in loop', 'why do you loop', 
                'analyze yourself', 'your own processing', 'your behavior', 
                'why did you', 'recursive', 'meta-cognitive', 'self-referential',
                'analyze your', 'your thinking', 'your response pattern',
                'why do you keep', 'infinite loop', 'repeated behavior',
                'same thing over', 'consciousness', 'self-awareness'
            ]
            
            input_lower = user_input.lower()
            
            # Check if any meta-keywords are present
            for keyword in meta_keywords:
                if keyword in input_lower:
                    logging.warning(f"META_QUESTION: Detected keyword '{keyword}' in user input")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error in meta-question detection: {e}")
            return False  # If detection fails, treat as normal question
                   
        #used by nightly learning and web knowlegde seeker
    def fill_knowledge_gaps(self, max_gaps: int = 3) -> str:
        """
        Trigger knowledge gap filling using web search.
        
        Args:
            max_gaps: Maximum number of gaps to process
            
        Returns:
            str: User-friendly status report
        """
        try:
            from knowledge_gap import KnowledgeGapQueue
            from nightly_learning import NightlyLearner
            
            logging.info(f"🚀 Starting knowledge gap filling process")
            
            # Initialize components
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            nightly_learner = NightlyLearner(chatbot=self)
            
            # Process knowledge gaps
            results = nightly_learner.process_knowledge_gaps_with_web_search(
                knowledge_gap_queue=gap_queue,
                max_gaps=max_gaps
            )
            
            if not results["success"]:
                return f"❌ Knowledge gap filling failed: {results.get('error', 'Unknown error')}"
            
            # Create user-friendly report
            if results["gaps_processed"] == 0:
                return "ℹ️ No pending knowledge gaps found to process."
            
            report = f"🧠 **Knowledge Gap Filling Complete**\n\n"
            report += f"📊 **Summary:**\n"
            report += f"• Gaps processed: {results['gaps_processed']}\n"
            report += f"• Successfully filled: {results['gaps_filled']}\n"
            report += f"• Failed to fill: {results['gaps_failed']}\n\n"
            
            if results["processed_topics"]:
                report += f"📋 **Processed Topics:**\n"
                for topic_info in results["processed_topics"]:
                    status_emoji = "✅" if topic_info["status"] == "filled" else "❌"
                    report += f"{status_emoji} {topic_info['topic']} - {topic_info['status']}\n"
            
            return report
            
        except Exception as e:
            logging.error(f"❌ Error in fill_knowledge_gaps: {e}", exc_info=True)
            return f"❌ Error filling knowledge gaps: {str(e)}"

    def _format_memory_for_context(self, memory: Dict) -> str:
        """Format a memory entry for context inclusion."""
        try:
            content = memory.get('content', '')
            confidence = memory.get('confidence', {}).get('level', 'Unknown')
            metadata = memory.get('metadata', {})
            memory_type = metadata.get('type', 'general')
            source = metadata.get('source', 'Unknown')
            if memory_type == 'document':
                return f"[Document Memory] {content} (Source: {source}, Confidence: {confidence})"
            elif memory_type == 'important':
                return f"[Important Memory] {content} (Confidence: {confidence})"
            elif memory_type == 'conversation':
                return f"[Conversation Summary] {content} (Confidence: {confidence})"
            else:
                return f"[Memory] {content} (Confidence: {confidence})"
        except Exception as e:
            logging.error(f"Error formatting memory: {e}")
            return str(memory.get('content', 'Error formatting memory'))
 
    def store_memory_with_transaction(self, content, memory_type, metadata=None, confidence=None, max_retries=2, duplicate_threshold=0.98):
        """
        Store memory with true two-phase commit across SQL and Vector databases.
        Ensures both databases stay in sync - both succeed or both fail together.
        
        Args:
            content: The text content to store
            memory_type: Type of memory (e.g., "conversation_summary", "user_info")
            metadata: Optional metadata dictionary
            confidence: Optional confidence weight (0.0 to 1.0)
            max_retries: Number of retry attempts for vector DB storage
            duplicate_threshold: Similarity threshold for duplicate detection (default 0.98)
                                Use higher values (e.g., 0.995) for conversation summaries
                                to allow similar but different content to be stored
        
        Returns:
            tuple[bool, str]: (success, memory_id or None)
        """
        memory_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        metadata['tracking_id'] = memory_id

        # Validate vector_db
        if not hasattr(self, 'vector_db') or self.vector_db is None:
            logging.error("VectorDB is not initialized")
            return False, None

        # Log the threshold being used
        logging.info(f"TRANSACTION: Storing {memory_type} with duplicate_threshold={duplicate_threshold}")

        # Convert tags array to comma-separated string for SQL storage
        tags_value = metadata.get("tags", None)
        if isinstance(tags_value, list):
            tags_str = ",".join(tags_value)  # Convert array to string for SQL
        else:
            tags_str = tags_value  # Already a string or None

        # ✅ PHASE 1: Prepare SQL transaction (but don't commit yet)
        conn = None
        cursor = None
        try:
            import sqlite3
            import json
            
            conn = sqlite3.connect(self.memory_db.db_path)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            conn.execute("BEGIN IMMEDIATE TRANSACTION")
            cursor = conn.cursor()
            
            # Calculate confidence weight if not provided
            if confidence is not None:
                initial_weight = confidence
            else:
                initial_weight = self.memory_db.calculate_memory_weight(
                    memory_type=memory_type,
                    access_count=0,
                    created_at=datetime.datetime.now()
                )
            
            # Prepare metadata JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert into SQL (but don't commit yet)
            cursor.execute("""
                INSERT INTO memories 
                (content, memory_type, source, weight, tags, tracking_id, access_count, created_at, last_accessed, metadata) 
                VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, (
                content,
                memory_type,
                metadata.get("source", "unknown"),
                initial_weight,
                tags_str,
                memory_id,
                metadata_json
            ))
            
            row_id = cursor.lastrowid
            logging.info(f"SQL insert prepared (not committed) for tracking_id={memory_id}, row_id={row_id}")
            
        except sqlite3.IntegrityError as integrity_error:
            # Handle constraint violations (e.g., duplicate tracking_id)
            logging.error(f"SQL integrity constraint violation: {integrity_error}")
            if conn:
                conn.rollback()
                conn.close()
            return False, None
            
        except Exception as e:
            logging.error(f"Error preparing SQL insert: {e}", exc_info=True)
            if conn:
                conn.rollback()
                conn.close()
            return False, None

        # ✅ PHASE 2: Try Vector DB storage
        metadata["memory_id"] = memory_id
        vector_success = False
        failure_reason = None
        
        for attempt in range(max_retries):
            try:
                backoff_time = 0.5 * (2 ** attempt) if attempt > 0 else 0
                if attempt > 0:
                    logging.info(f"VectorDB retry attempt {attempt+1}/{max_retries} after {backoff_time:.2f}s")
                    time.sleep(backoff_time)
                
                logging.debug(f"Attempting VectorDB add_text: memory_id={memory_id}, metadata={metadata}")
                
                # ================================================================
                # FIXED: Pass duplicate_threshold to vector_db.add_text()
                # This allows conversation summaries to use a stricter threshold
                # ================================================================
                vector_success, reason = self.vector_db.add_text(
                    text=content,
                    metadata=metadata,
                    memory_id=memory_id,
                    duplicate_threshold=duplicate_threshold
                )
                failure_reason = reason
                
                if vector_success:
                    logging.info(f"VectorDB add_text succeeded for memory_id={memory_id}")
                    break
                elif reason == "duplicate":
                    # If it's a duplicate, stop retrying immediately
                    logging.info(f"Duplicate detected in VectorDB, stopping retries for memory_id={memory_id}")
                    break
                else:
                    logging.warning(f"VectorDB add_text returned False on attempt {attempt+1}, reason: {reason}")
                    if attempt == max_retries - 1:
                        logging.error(f"All {max_retries} VectorDB attempts failed")
                        
            except Exception as e:
                logging.error(f"VectorDB error on attempt {attempt+1}: {e}", exc_info=True)
                failure_reason = "error"
                if attempt == max_retries - 1:
                    logging.error(f"Final VectorDB attempt failed: {e}")

        # ✅ PHASE 3: Commit or Rollback based on Vector DB result
        try:
            if not vector_success:
                # Vector DB failed - rollback SQL transaction
                conn.rollback()
                conn.close()
                
                if failure_reason == "duplicate":
                    logging.info(f"Duplicate detected in VectorDB - rolled back SQL transaction: {content[:50]}...")
                    # Don't queue for retry - this is expected behavior
                    return False, None
                else:
                    logging.warning(f"VectorDB storage failed (reason: {failure_reason}) - rolled back SQL transaction: {content[:50]}...")
                    # Queue for retry on actual errors
                    try:
                        self.memory_db.queue_for_deletion(memory_id)
                    except Exception as queue_error:
                        logging.error(f"Error queueing for retry: {queue_error}")
                    return False, None
            else:
                # Vector DB succeeded - commit SQL transaction
                conn.commit()
                conn.close()
                logging.info(f"✅ Successfully stored memory in BOTH databases with ID {memory_id}: {content[:50]}...")
                self._last_memory_id = memory_id
                return True, memory_id
                
        except Exception as commit_error:
            logging.error(f"Error during commit/rollback phase: {commit_error}", exc_info=True)
            try:
                if conn:
                    conn.rollback()
                    conn.close()
            except:
                pass
            return False, None
                
    def get_unified_token_count(self) -> tuple[int, int, float]:
        """
        Get current token count - READ ONLY, no accumulation.
        Returns: (estimated_active_tokens, max_tokens, percentage)
        """
        try:
            # Get max tokens from config
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            
            # Initialize tracking variables if needed
            if not hasattr(self, '_cumulative_tokens_sent'):
                self._cumulative_tokens_sent = 0
            
            # Calculate estimated active window
            if self._cumulative_tokens_sent <= max_tokens:
                estimated_active = self._cumulative_tokens_sent
                overflow = 0
            else:
                # Context overflow - estimate what model sees
                attention_sink = 2000
                usable_window = max_tokens - attention_sink
                estimated_active = int(max_tokens * 0.92)  # ~30,000 tokens
                overflow = self._cumulative_tokens_sent - max_tokens
                logging.info(f"TOKEN_TRACKING: Context overflow - {overflow:,} tokens beyond limit")
            
            # Calculate percentage
            percentage = (estimated_active / max_tokens) * 100
            
            # Log current state (less verbose than before)
            logging.info(f"TOKEN_READ: {estimated_active:,}/{max_tokens:,} ({percentage:.2f}%)")
            
            return estimated_active, max_tokens, percentage
            
        except Exception as e:
            logging.error(f"TOKEN_COUNT: Error - {e}", exc_info=True)
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            return 0, max_tokens, 0.0

    def get_token_statistics(self) -> dict:
        """
        Get comprehensive token statistics for detailed UI display.
        
        Returns:
            dict: {
                'cumulative_sent': Total tokens sent this session,
                'estimated_active': What model likely sees,
                'max_tokens': Context window limit,
                'percentage_active': Usage percentage based on active,
                'overflow': Tokens beyond limit,
                'experiencing_fade': Boolean if in overflow state,
                'attention_sink': Estimated early tokens retained
            }
        """
        try:
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            
            # Get current values from the standard method
            estimated_active, _, percentage = self.get_unified_token_count()
            
            # Get cumulative (with safe initialization)
            cumulative = getattr(self, '_cumulative_tokens_sent', 0)
            
            # Calculate overflow
            if cumulative > max_tokens:
                overflow = cumulative - max_tokens
                experiencing_fade = True
                attention_sink_estimate = 2000  # Early tokens model retains
            else:
                overflow = 0
                experiencing_fade = False
                attention_sink_estimate = 0
            
            return {
                'cumulative_sent': cumulative,
                'estimated_active': estimated_active,
                'max_tokens': max_tokens,
                'percentage_active': percentage,
                'overflow': overflow,
                'experiencing_fade': experiencing_fade,
                'attention_sink': attention_sink_estimate
            }
            
        except Exception as e:
            logging.error(f"Error getting token statistics: {e}", exc_info=True)
            return {
                'cumulative_sent': 0,
                'estimated_active': 0,
                'max_tokens': 32768,
                'percentage_active': 0.0,
                'overflow': 0,
                'experiencing_fade': False,
                'attention_sink': 0
            }
        
    def get_token_stats_readonly(self) -> tuple[int, int, float]:
        """
        Get token statistics WITHOUT triggering accumulation.
        Safe to call anytime for display purposes or intermediate calculations.
        
        This is READ-ONLY - it will never modify _cumulative_tokens_sent
        or any other token tracking variables.
        
        Use this method for:
        - UI display (sidebar token counter)
        - Intermediate checks during prompt building
        - Second-pass token calculations
        - Any situation where you need current stats without side effects
        
        Use get_unified_token_count() only:
        - After an LLM invoke() call
        - When you need to trigger accumulation
        
        Returns:
            tuple[int, int, float]: (estimated_active_tokens, max_tokens, percentage)
        """
        try:
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            
            # Get current cumulative value without modifying it
            cumulative = getattr(self, '_cumulative_tokens_sent', 0)
            
            # Calculate estimated active window (same logic as get_unified_token_count)
            if cumulative <= max_tokens:
                # Within limit - show actual cumulative
                estimated_active = cumulative
            else:
                # Exceeded limit - estimate what model sees with attention sink
                estimated_active = int(max_tokens * 0.92)  # ~30,000 tokens
            
            # Calculate percentage
            percentage = (estimated_active / max_tokens) * 100
            
            # Light diagnostic logging (not as verbose as the full method)
            logging.debug(f"TOKEN_READ_ONLY: {estimated_active:,}/{max_tokens:,} ({percentage:.1f}%)")
            
            return estimated_active, max_tokens, percentage
            
        except Exception as e:
            logging.error(f"TOKEN_READ_ONLY: Error - {e}", exc_info=True)
            max_tokens = MODEL_PARAMS.get("num_ctx", 32768)
            return 0, max_tokens, 0.0


    def get_token_usage_warning(self) -> str:
        """
        Generate context-aware token usage warnings for the model.
        Returns a formatted warning string to include in the model's prompt.
        
        Returns:
            str: Warning message (empty string if no warning needed)
        """
        try:
            stats = self.get_token_statistics()
            percentage = stats['percentage_active']
            current = stats['estimated_active']
            max_tokens = stats['max_tokens']
            experiencing_fade = stats['experiencing_fade']
            overflow = stats['overflow']
            
            # Check if we need to auto-summarize (at 85% to account for 4K summary + overhead)
            if percentage >= 85:
                # Trigger auto-summarization flag
                if not hasattr(self, '_auto_summary_triggered'):
                    self._auto_summary_triggered = False
                
                if not self._auto_summary_triggered and percentage < 100:
                    # Set flag to prevent repeated triggers
                    self._auto_summary_triggered = True
                    
                    return f"""
    🔄 AUTO-SUMMARIZATION INITIATED: You are at {current:,}/{max_tokens:,} tokens ({percentage:.1f}%).
    Your system is auto-summarizing the current conversation now to maintain continuous context.
    The summary will be injected into the current conversation so you may continue seamlessly.
    """
                
                # If we've exceeded 100%, something went wrong
                if percentage >= 100:
                    return f"""
                ⚠️ CRITICAL: Your auto-summarization should have triggered at 85% but failed.
                Please ask Ken to check the system logs.
                You are currently at {current:,}/{max_tokens:,} tokens ({percentage:.1f}%).
                """

            return ""
            
        except Exception as e:
            logging.error(f"Error generating token usage warning: {e}", exc_info=True)
            return ""


## ═══════════════════════════════════════════════════════════════════
## NEW METHOD 2: reset_token_counter_after_summary()
## Location: Add AFTER get_token_statistics() 
## Action: ADD this new method
## ═══════════════════════════════════════════════════════════════════

    def reset_token_counter_after_summary(self, keep_lifetime_stats=True):
        """
        Reset token tracking after conversation summarization.
        
        Args:
            keep_lifetime_stats (bool): If True, adds current cumulative to lifetime total
        
        Returns:
            bool: Success status
        """
        try:
            # Store cumulative in lifetime counter if requested
            if keep_lifetime_stats and hasattr(self, '_cumulative_tokens_sent'):
                if not hasattr(self, '_total_tokens_all_time'):
                    self._total_tokens_all_time = 0
                self._total_tokens_all_time += self._cumulative_tokens_sent
                logging.info(f"TOKEN_RESET: Added {self._cumulative_tokens_sent:,} to lifetime total (now {self._total_tokens_all_time:,})")
            
            # Reset current session tracking
            # Set to 0 because:
            # 1. Startup context was already counted in the tokens that triggered summarization
            # 2. New summary will be counted on next message turn when prompt is built
            base_tokens = 0
            self._cumulative_tokens_sent = base_tokens
            self._last_prompt_tokens = base_tokens
            self._last_counted_prompt_text = None

            logging.info("TOKEN_RESET: Counter reset after summarization")
            logging.info(f"TOKEN_RESET: Starting fresh at {base_tokens:,} tokens - next prompt will include new summary")
            
            return True
            
        except Exception as e:
            logging.error(f"Error resetting token counter: {e}", exc_info=True)
            return False


    def trigger_auto_summarization(self) -> dict:
        """
        Automatically summarize conversation when token threshold is reached.
        Does NOT clear conversation - only resets token counter and injects summary.
        
        Returns:
            dict: {
                'success': bool,
                'summary': str (if successful),
                'error': str (if failed),
                'tokens_before': int,
                'tokens_after': int
            }
        """
        try:
            logging.info("🔄 AUTO_SUMMARIZATION: Starting automatic conversation summarization")
            
            # Get current token stats
            stats_before = self.get_token_statistics()
            tokens_before = stats_before['cumulative_sent']
            
            # Step 1: Get conversation from Streamlit
            import streamlit as st
            if not hasattr(st, 'session_state') or 'messages' not in st.session_state:
                logging.error("AUTO_SUMMARIZATION: No conversation available in Streamlit")
                return {
                    'success': False,
                    'error': 'No conversation found',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            conversation = st.session_state.messages
            
            if len(conversation) < 10:
                logging.warning("AUTO_SUMMARIZATION: Conversation too short to summarize")
                return {
                    'success': False,
                    'error': 'Conversation too short',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            # Step 2: Generate and store summary
            logging.info(f"🔄 AUTO_SUMMARIZATION: Generating summary from {len(conversation)} messages")
            success = self._generate_and_store_conversation_summary(conversation)
            
            if not success:
                logging.error("AUTO_SUMMARIZATION: Failed to generate summary")
                return {
                    'success': False,
                    'error': 'Summary generation failed',
                    'tokens_before': tokens_before,
                    'tokens_after': tokens_before
                }
            
            # Step 3: Reset token counter (but keep conversation in UI)
            # This resets our ESTIMATE, not Ollama's actual sliding window
            self.reset_token_counter_after_summary(keep_lifetime_stats=True)
            
            # Step 4: Reset the auto-summary trigger flag for next cycle
            self._auto_summary_triggered = False
            
            # Get new token stats
            stats_after = self.get_token_statistics()
            tokens_after = stats_after['cumulative_sent']
            
            logging.info(f"🔄 AUTO_SUMMARIZATION: Complete! Token estimate reset: {tokens_before:,} → {tokens_after:,}")
            logging.info(f"🔄 AUTO_SUMMARIZATION: Note - Ollama maintains its own sliding window independently")
            
            return {
                'success': True,
                'tokens_before': tokens_before,
                'tokens_after': tokens_after
            }
            
        except Exception as e:
            logging.error(f"AUTO_SUMMARIZATION: Error during auto-summarization: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'tokens_before': 0,
                'tokens_after': 0
            }
    
    def _identify_search_mode(self, user_input: str) -> tuple[str, str]:
        """
        Simplified search mode identification - just two modes: default and comprehensive.
        The model should decide which to use via its memory commands.
    
        Args:
            user_input (str): The user input to analyze
        
        Returns:
            tuple[str, str]: (search_mode, processed_input)
        """
        # Check for null or empty input
        if not user_input or not user_input.strip():
            return "default", ""  # Return empty string for empty input
    
        cleaned_input = user_input.lower()

        # Check for explicit comprehensive search requests
        comprehensive_prefixes = ("deep search:", "comprehensive search:", "search thoroughly", 
                                  "find all", "tell me everything about")
        for prefix in comprehensive_prefixes:
            if cleaned_input.startswith(prefix):
                return "comprehensive", user_input[len(prefix):].strip()

        # Default search mode for everything else - let the model decide when to use [COMPREHENSIVE_SEARCH]
        return "default", user_input

    def _gather_relevant_context(self, user_input: str) -> str:
        """
        Gather relevant context from memory based on the user input.

        Args:
            user_input (str): The user's input

        Returns:
            str: Formatted memory context
        """
        try:
            if not user_input or not isinstance(user_input, str) or not user_input.strip():
                logging.warning("Empty user_input provided to _gather_relevant_context")
                return ""
        
            # Determine search mode based on user input
            search_mode, processed_input = self._identify_search_mode(user_input)
        
            # Set search parameters based on mode
            if search_mode == "comprehensive":
                k = 20  # More results for comprehensive
                threshold = 0.30 # Lower threshold for better recall
            elif search_mode == "selective":
                k = 5  # Fewer, more precise results
                threshold = 0.40  # Higher threshold for precision
            else:  # default
                k = 5
                threshold = 0.35
        
            # Get relevant memories based on user input
            results = self.vector_db.search(
                query=processed_input,
                mode=search_mode,
                k=k
            )
        
            # Get conversation context if available
            conversation_context = ""
            if hasattr(self, 'conversation_manager'):
                conversation_context = self.conversation_manager.get_formatted_context()
        
            if not results:
                logging.info(f"No relevant memories found for: {processed_input[:50]}...")
                # Return conversation context even if no memory results found
                return conversation_context
        
            # Filter by threshold
            filtered_results = [r for r in results if r.get('similarity_score', 0) >= threshold]
        
            if not filtered_results:
                logging.info(f"No memories passed threshold {threshold} for: {processed_input[:50]}...")
                # Return conversation context even if no filtered results
                return conversation_context
        
            # Format memory context for inclusion in prompt
            formatted_context = []
            for i, result in enumerate(filtered_results[:5]):  # Limit to top 5
                memory_content = result.get('content', '')
                if not memory_content:
                    continue
                
                score = result.get('similarity_score', 0)
                memory_type = result.get('metadata', {}).get('type', 'general')
                source = result.get('metadata', {}).get('source', 'Unknown')
            
                # Assign type prefixes for better readability
                type_prefix = {
                    "important": "[Important Memory]",
                    "document": "[Document Memory]", 
                    "general": "[Memory]",
                    "conversation": "[Conversation Summary]"
                }.get(memory_type, "[Memory]")
            
                formatted_context.append(
                    f"{type_prefix} {memory_content} (Source: {source}, Relevance: {score:.2f})"
                )
        
            # Log the automatic search
            logging.info(f"Automatic search for '{processed_input[:30]}...' found {len(filtered_results)} relevant memories")
       
            # Increment retrieve counter for auto-search
            if hasattr(self, 'deepseek_enhancer'):
                self.deepseek_enhancer.lifetime_counters.increment('retrieve')
                # Update session counters - safely check for streamlit
                try:
                    # Try to import streamlit if it's available
                    import streamlit as st_local
                    if hasattr(st_local, 'session_state') and 'memory_command_counts' in st_local.session_state:
                        if 'retrieve' not in st_local.session_state.memory_command_counts:
                            st_local.session_state.memory_command_counts['retrieve'] = 0
                        st_local.session_state.memory_command_counts['retrieve'] += 1
                except (ImportError, ModuleNotFoundError):
                    # Streamlit not available, skip counter update
                    logging.info("Streamlit not available, skipping session counter update")
                    
            # Return formatted context string
            if formatted_context:
                memory_str = "\n\n".join(formatted_context)
                logging.info(f"Auto-retrieved {len(formatted_context)} memories for context")
                
                # Combine conversation context with memory results
                if conversation_context:
                    return conversation_context + memory_str
                else:
                    return memory_str
            else:
                # Return conversation context if no memory context
                return conversation_context
            
        except Exception as e:
            logging.error(f"Error gathering relevant context: {e}")
            return ""

    def search_ai_learned_content(self, topic: str) -> str:
        """Search for content learned through AI-driven web processing."""
        
        # Search with specific filters for AI-processed web content
        results = self.vector_db.search(
            query=topic,
            mode="comprehensive",
            k=10,
            metadata_filters={
                "type": "web_learning",
                "processed_by": "ai_driven_selection"
            }
        )
        
        if not results:
            return f"No AI-learned web content found for '{topic}'"
        
        formatted_results = [f"\n**AI-LEARNED WEB CONTENT ABOUT '{topic.upper()}'**\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get('content', '')
            source = result.get('metadata', {}).get('source', 'Unknown')
            score = result.get('similarity_score', 0)
            
            formatted_results.append(f"**[{i}]** (Score: {score:.2f}) Source: {source}")
            formatted_results.append(f"{content[:300]}...")
            formatted_results.append("")
        
        return "\n".join(formatted_results) 
          
                    
    def format_response_with_reasoning(response):
        """Extract and format model reasoning for display."""
        reasoning_pattern = re.compile(r'<model_reasoning>(.*?)</model_reasoning>', re.DOTALL)
        reasoning_match = reasoning_pattern.search(response)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            final_response = response.replace(reasoning_match.group(0), '').strip()
            
            # Format with clear separation
            return f"MODEL REASONING:\n{reasoning}\n\nFINAL RESPONSE:\n{final_response}"
        
        return response  # If no reasoning found, return as-is
        
    def _generate_and_store_conversation_summary(self, conversation: List[Dict]) -> bool:
        """Generate and store a conversation summary using the conversation summary manager."""
        try:
            # Skip if conversation is too short
            if len(conversation) < 10:
                logging.info("Conversation too short for summarization")
                return False
            
            # Use the conversation_summary_manager to generate and store the summary
            if not hasattr(self, 'conversation_summary_manager'):
                logging.error("Conversation summary manager not initialized")
                return False
            
            # Generate summary using the conversation summary manager
            summary = self.conversation_summary_manager.generate_summary(conversation)
            
            if not summary:
                logging.warning("Failed to generate conversation summary")
                return False
            
            logging.info(f"Generated conversation summary: {len(summary)} characters")
            
            # Store using conversation state manager
            state_success = False
            if hasattr(self, 'conversation_manager'):
                state_success = self.conversation_manager.update_summary(
                    summary, 
                    memory_db=self.memory_db
                )
                logging.info(f"Stored summary in conversation_manager: {state_success}")
            
            return state_success
            
        except Exception as e:
            logging.error(f"Error generating and storing summary: {e}", exc_info=True)
            return False
            
    def log_command_usage_summary(self):
        """Log the current command usage summary."""
        try:
            summary = self.get_command_usage_summary()
            logging.info("=== COMMAND USAGE SUMMARY ===")
            logging.info(f"Session ID: {summary.get('session_id', 'Unknown')}")
            
            if 'lifetime_counters' in summary:
                total_lifetime = summary['lifetime_counters'].get('total', 0)
                logging.info(f"Total Lifetime Commands: {total_lifetime}")
                
                # Log top 5 most used commands
                lifetime_sorted = sorted(
                    [(k, v) for k, v in summary['lifetime_counters'].items() if k != 'total' and v > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                if lifetime_sorted:
                    logging.info("Top Lifetime Commands:")
                    for cmd, count in lifetime_sorted:
                        logging.info(f"  {cmd}: {count}")
            
            if 'session_counters' in summary:
                total_session = sum([v for v in summary['session_counters'].values() if isinstance(v, int)])
                logging.info(f"Total Session Commands: {total_session}")
                
                # Log session commands with counts > 0
                session_active = [(k, v) for k, v in summary['session_counters'].items() if v > 0]
                if session_active:
                    logging.info("Active Session Commands:")
                    for cmd, count in session_active:
                        logging.info(f"  {cmd}: {count}")
            
            logging.info("=== END COMMAND USAGE SUMMARY ===")
            
        except Exception as e:
            logging.error(f"Error logging command usage summary: {e}")

    def check_and_repair_database_sync(self):
        """Check for and repair inconsistencies between MemoryDB and VectorDB."""
        try:
            logging.info("Starting database synchronization check...")
        
            # Import QDRANT_COLLECTION_NAME directly from config
            from config import QDRANT_COLLECTION_NAME
    
            # Track statistics for reporting
            stats = {
                "memory_db_total": 0,
                "vector_db_total": 0,
                "memory_missing_in_vector": 0,
                "vector_missing_in_memory": 0,
                "repairs_attempted": 0,
                "repairs_succeeded": 0
            }
    
            # Step 1: Get count of items in both databases for reference
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats["memory_db_total"] = cursor.fetchone()[0]
    
            # Get vector DB count - handle properly with error checking
            try:
                # Use QDRANT_COLLECTION_NAME directly instead of self.vector_db.client.collection_name
                vector_count = self.vector_db.client.count(
                    collection_name=QDRANT_COLLECTION_NAME,
                    count_filter=None  # Count all points
                )
                stats["vector_db_total"] = vector_count.count
            except Exception as e:
                logging.error(f"Error getting vector count: {e}")
                stats["vector_db_total"] = "unknown"
    
            # Step 2: Check MemoryDB items in VectorDB using tracking_id (more reliable than content)
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                # Use limit and pagination to handle large databases
                batch_size = 50
                offset = 0
        
                while True:
                    cursor.execute("""
                        SELECT content, memory_type, tracking_id, id
                        FROM memories 
                        ORDER BY id
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
            
                    memories = cursor.fetchall()
                    if not memories:
                        break  # No more records
            
                    for content, memory_type, tracking_id, memory_id in memories:
                        # Skip if content is empty
                        if not content or not content.strip():
                            continue
                
                        # First try to find by tracking_id if available
                        memory_exists = False
                        if tracking_id:
                            # Search for exact tracking_id match in vector_db
                            try:
                                # Use vector search with filter to find by tracking_id
                                results = self.vector_db.search(
                                    query=content[:50],  # Use partial content as query
                                    mode="selective",
                                    k=3
                                )
                        
                                # Check if any result has the matching tracking_id
                                for result in results:
                                    result_tracking_id = result.get('metadata', {}).get('memory_id')
                                    if result_tracking_id == tracking_id:
                                        memory_exists = True
                                        break
                            except Exception as e:
                                logging.warning(f"Error searching by tracking_id: {e}")
                
                        # If no match by tracking_id, try content-based search
                        if not memory_exists:
                            results = self.vector_db.search(
                                query=content,
                                mode="selective",
                                k=3
                            )
                    
                            for result in results:
                                # Check for exact content match or very high similarity
                                if (result['content'] == content or 
                                    (result.get('similarity_score', 0) > 0.95 and content in result['content'])):
                                    memory_exists = True
                                    break
                
                        # If memory doesn't exist in VectorDB, add it
                        if not memory_exists:
                            logging.warning(f"Memory found in MemoryDB but missing in VectorDB (ID: {memory_id})")
                            stats["memory_missing_in_vector"] += 1
                    
                            # Add to VectorDB with original metadata
                            try:
                                # Get all metadata from the memory record
                                cursor.execute("""
                                    SELECT source, tags, weight, created_at
                                    FROM memories WHERE id = ?
                                """, (memory_id,))
                                meta_row = cursor.fetchone()
                        
                                if meta_row:
                                    source, tags, weight, created_at = meta_row
                                    metadata = {
                                        "metadata.type": memory_type, 
                                        "metadata.source": source or "sync_repair", 
                                        "metadata.memory_id": tracking_id or str(uuid.uuid4()),
                                        "metadata.tags": tags
                                    }
                            
                                    vector_success = self.vector_db.add_text(
                                        text=content, 
                                        metadata=metadata
                                    )
                            
                                    if vector_success:
                                        stats["repairs_succeeded"] += 1
                                        logging.info(f"Successfully repaired missing vector for memory ID {memory_id}")
                                    else:
                                        logging.error(f"Failed to add missing vector for memory ID {memory_id}")
                        
                                stats["repairs_attempted"] += 1
                            except Exception as repair_error:
                                logging.error(f"Error repairing memory: {repair_error}")
            
                    # Move to next batch
                    offset += batch_size
                    logging.info(f"Processed {offset} memories from MemoryDB")
    
            # Step 3: Sample check for orphaned vectors (in VectorDB but not in MemoryDB)
            try:
                # We'll sample a portion of vectors to keep this efficient
                vector_sample_limit = 100
        
                # Get a sample of vectors using search with a broad query
                sample_results = self.vector_db.search(
                    query="memory data information",  # Generic terms to get diverse results
                    mode="comprehensive",
                    k=vector_sample_limit
                )
        
                if sample_results:
                    for result in sample_results:
                        content = result.get('content', '')
                        if not content:
                            continue
                
                        # Get memory_id if available
                        memory_id = result.get('metadata', {}).get('memory_id')
                
                        # Check if exists in MemoryDB
                        memory_exists = False
                
                        with sqlite3.connect(self.memory_db.db_path) as conn:
                            cursor = conn.cursor()
                    
                            # First try by memory_id if available
                            if memory_id:
                                cursor.execute("""
                                    SELECT COUNT(*) FROM memories 
                                    WHERE tracking_id = ?
                                """, (memory_id,))
                                count = cursor.fetchone()[0]
                                if count > 0:
                                    memory_exists = True
                    
                            # If not found by ID, try by content
                            if not memory_exists:
                                cursor.execute("""
                                    SELECT COUNT(*) FROM memories 
                                    WHERE content = ?
                                """, (content,))
                                count = cursor.fetchone()[0]
                                if count > 0:
                                    memory_exists = True
                
                        # If vector exists but memory doesn't, report it
                        if not memory_exists:
                            stats["vector_missing_in_memory"] += 1
                            logging.warning(f"Vector found in VectorDB but missing in MemoryDB: {content[:100]}...")
                    
                            
            except Exception as e:
                logging.error(f"Error checking for orphaned vectors: {e}")
    
            # Step 4: Generate final report
            logging.info(f"Database sync check completed. Stats: {stats}")
    
            result = {
                "found": stats["memory_db_total"],
                "repaired": stats["repairs_succeeded"],
                "missing_in_vector": stats["memory_missing_in_vector"],
                "missing_in_memory": stats["vector_missing_in_memory"],
                "stats": stats
            }
    
            # Store the sync report in a dedicated file for tracking
            try:
                sync_report_path = os.path.join(os.path.dirname(self.memory_db.db_path), "sync_reports.json")
        
                existing_reports = []
                if os.path.exists(sync_report_path):
                    try:
                        with open(sync_report_path, 'r') as f:
                            existing_reports = json.load(f)
                    except:
                        existing_reports = []
        
                # Add timestamp to this report
                report_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "report": result
                }
        
                # Keep only the last 10 reports
                existing_reports.append(report_entry)
                if len(existing_reports) > 10:
                    existing_reports = existing_reports[-10:]
        
                with open(sync_report_path, 'w') as f:
                    json.dump(existing_reports, f, indent=2)
            except Exception as e:
                logging.error(f"Error storing sync report: {e}")
    
            return result
    
        except Exception as e:
            logging.error(f"Error in database sync check: {e}")
            return {"error": str(e), "found": 0, "repaired": 0}
        
           
    def check_due_reminders(self):
        """
        Check for reminders that are due by delegating to the reminder_manager.
        
        Returns:
            list: List of due reminders
        """
        try:
            if hasattr(self, 'reminder_manager') and self.reminder_manager:
                # Delegate to the existing method in the reminder manager
                return self.reminder_manager.get_due_reminders()
            else:
                logging.warning("ReminderManager not available, cannot check due reminders")
                return []
        except Exception as e:
            logging.error(f"Error checking due reminders: {e}")
            return []
    
    def correct_memory(self, original_content: str, new_content: str) -> str:
        try:
            # First, check if the original memory exists or something similar exists
            logging.info(f"Checking if memory exists: '{original_content[:50]}...'")
            memory_exists = self.memory_db.contains(original_content)
            
            if not memory_exists:
                logging.warning(f"Exact memory not found in memory_db: '{original_content[:50]}...'")
                # Try to find similar content in memory_db instead of just failing
                similar_memories = self.memory_db.search_similar(original_content)
                if similar_memories and len(similar_memories) > 0:
                    original_content = similar_memories[0]['content']
                    logging.info(f"Found similar memory in memory_db: '{original_content[:50]}...'")
                    memory_exists = True
                    
            if not memory_exists:
                return f"Memory not found: '{original_content[:50]}...'"
            
            # Get the original memory metadata using search_with_ids
            logging.info(f"Finding memory metadata for: '{original_content[:50]}...'")
            memory = None
            
            # Use search_with_ids to get proper vector IDs
            original_memories = self.vector_db.search_with_ids(original_content, mode="selective", k=5)
            logging.info(f"Found {len(original_memories)} potential matches in vector_db")

            # First try exact match
            for mem in original_memories:
                logging.info(f"Comparing: '{mem['content'][:50]}...' to '{original_content[:50]}...'")
                if mem['content'] == original_content:
                    memory = mem
                    logging.info("Found exact match")
                    break
            
            # If no exact match, try best similar match above threshold
            if not memory and original_memories:
                best_match = None
                highest_score = 0
                for mem in original_memories:
                    if mem['similarity_score'] >= 0.40 and mem['similarity_score'] > highest_score:
                        best_match = mem
                        highest_score = mem['similarity_score']
                
                if best_match:
                    memory = best_match
                    original_content = best_match['content']  # Update original_content to what we found
                    logging.info(f"Found similar match with score {highest_score}: '{original_content[:50]}...'")
                    
            if not memory:
                logging.warning(f"No suitable match found in vector_db for: '{original_content[:50]}...'")
                return f"Unable to retrieve full metadata for memory: '{original_content[:50]}...'"

            # Get metadata from the original memory BEFORE deletion
            metadata = memory.get('metadata', {})
            memory_type = metadata.get('type', 'general')
            source = metadata.get('source', '')
            confidence = metadata.get('confidence', 0.5)

            # Add note about correction
            if source:
                source = f"{source} (corrected)"
            else:
                source = "corrected"
        
            # Update metadata
            metadata["source"] = source

            # CRITICAL FIX: Use coordinated deletion instead of separate calls
            logging.info(f"Using coordinated deletion for: '{original_content[:50]}...'")
            deletion_success = self.delete_memory_with_coordination(original_content)
            
            if not deletion_success:
                logging.error(f"Coordinated deletion failed for: '{original_content[:50]}...'")
                return f"Error deleting original memory: '{original_content[:50]}...'"
            
            logging.info(f"Coordinated deletion successful, now storing corrected content")
            
            # Store the corrected memory using transaction coordination
            success, memory_id = self.store_memory_with_transaction(
                content=new_content,
                memory_type=memory_type,
                metadata=metadata,
                confidence=confidence
            )

            if success:
                logging.info(f"Successfully corrected memory with new ID: {memory_id}")
                return f"Successfully corrected memory:\nOriginal: '{original_content[:100]}...'\nCorrected: '{new_content[:100]}...'"
            else:
                logging.error(f"Failed to store corrected memory, but original was already deleted!")
                # This is a critical state - original is deleted but new content failed to store
                return f"CRITICAL ERROR: Original memory deleted but failed to store corrected version: '{new_content[:50]}...'"
        
        except Exception as e:
            logging.error(f"Error correcting memory: {e}", exc_info=True)
            return f"Error correcting memory: {str(e)}"
        
    def delete_memory_with_coordination(self, content: str) -> bool:
        """
        Delete a memory ensuring both SQL and Vector databases stay synchronized.
        Uses transaction-like behavior with rollback capability.
        """
        try:
            logging.info(f"DELETE_COORDINATION: Starting synchronized deletion for: '{content[:100]}...'")
            
            # Step 1: Find the memory using vector search WITH IDs
            vector_results = self.vector_db.search_with_ids(  # Use the new method
                query=content,
                mode="comprehensive",
                k=15
            )
            
            if not vector_results:
                logging.warning("DELETE_COORDINATION: No vector results found")
                return False
            
            # Step 2: Find best match and extract identifiers
            best_match = self._find_best_vector_match(content, vector_results)
            if not best_match:
                logging.warning("DELETE_COORDINATION: No suitable vector match found")
                return False
            
            vector_id = best_match.get('id')  # This should now have a value!
            vector_content = best_match.get('content', '')
            metadata = best_match.get('metadata', {})
            tracking_id = metadata.get('tracking_id') or metadata.get('memory_id')
            
            logging.info(f"DELETE_COORDINATION: Best match - Vector ID: {vector_id}, Tracking ID: {tracking_id}")

            
            # Step 3: Backup information for potential rollback
            backup_info = {
                'vector_id': vector_id,
                'vector_content': vector_content,
                'tracking_id': tracking_id,
                'metadata': metadata
            }
            
            # Step 4: Delete from SQL database (with backup for rollback)
            sql_memory_backup = None
            sql_success = False

            # Find and backup the SQL memory before deletion
            sql_memory_backup = self._backup_sql_memory(tracking_id, vector_content, content)

            if sql_memory_backup:
                # Found SQL record - proceed with normal deletion
                sql_success = self._delete_sql_with_identifiers(tracking_id, vector_content, content)
                logging.info(f"DELETE_COORDINATION: SQL deletion: {'success' if sql_success else 'failed'}")
            else:
                # No SQL record found - this might be an orphaned vector entry
                logging.warning("DELETE_COORDINATION: No SQL record found - treating as orphaned vector entry")
                sql_success = True  # Consider SQL "deletion" successful since there's nothing to delete
                
                # Create a dummy backup for consistency
                sql_memory_backup = {
                    'orphaned_entry': True,
                    'tracking_id': tracking_id,
                    'vector_content': vector_content
                }
            
            # Step 5: If SQL deletion failed, abort
            if not sql_success:
                logging.error("DELETE_COORDINATION: SQL deletion failed - aborting")
                return False

            # Step 6: Delete from Vector database
            vector_success = self._delete_vector_with_identifiers(vector_id, tracking_id, vector_content)
            logging.info(f"DELETE_COORDINATION: Vector deletion: {'success' if vector_success else 'failed'}")

            # Step 7: Handle rollback if vector deletion failed
            if not vector_success:
                # Only attempt SQL rollback if we actually deleted something from SQL
                if not sql_memory_backup.get('orphaned_entry', False):
                    logging.warning("DELETE_COORDINATION: Vector deletion failed - attempting SQL rollback")
                    rollback_success = self._rollback_sql_deletion(sql_memory_backup)
                    
                    if rollback_success:
                        logging.info("DELETE_COORDINATION: Successfully rolled back SQL deletion")
                    else:
                        logging.error("DELETE_COORDINATION: CRITICAL - SQL rollback failed! Databases are out of sync!")
                else:
                    logging.info("DELETE_COORDINATION: Vector deletion failed but no SQL rollback needed (orphaned entry)")
                
                return False

            # Step 8: Success - both databases updated (or orphaned vector entry cleaned up)
            if sql_memory_backup.get('orphaned_entry', False):
                logging.info("DELETE_COORDINATION: Successfully cleaned up orphaned vector entry")
            else:
                logging.info("DELETE_COORDINATION: Successfully deleted from both SQL and Vector databases")

            return True
            
        except Exception as e:
            logging.error(f"DELETE_COORDINATION: Exception during deletion: {e}", exc_info=True)
            return False

    def _find_best_vector_match(self, content: str, vector_results: list = None) -> dict:
        """Find the best matching vector result, using direct search if needed."""
        best_match = None
        best_score = 0
        
        # If no vector_results provided, do a direct search with IDs
        if not vector_results:
            try:
                vector_results = self.vector_db.search_with_ids(
                    query=content,
                    mode="comprehensive",
                    k=10
                )
            except Exception as e:
                logging.error(f"Error in direct search with IDs: {e}")
                return None
        
        for result in vector_results:
            score = result.get('similarity_score', 0)
            result_content = result.get('content', '')
            
            # Calculate word overlap
            content_words = set(content.lower().split())
            result_words = set(result_content.lower().split())
            
            if content_words and result_words:
                word_overlap = len(content_words.intersection(result_words)) / len(content_words)
                combined_score = (score * 0.7) + (word_overlap * 0.3)
                
                is_good_match = (
                    score >= 0.45 or
                    word_overlap >= 0.40 or
                    content.lower() in result_content.lower() or
                    result_content.lower() in content.lower()
                )
                
                if is_good_match and combined_score > best_score:
                    best_match = result
                    best_score = combined_score
                    logging.info(f"BEST_MATCH: Score {score:.2f}, Overlap {word_overlap:.2f}, Combined {combined_score:.2f}, ID: {result.get('id', 'None')}")
        
        return best_match

    def _backup_sql_memory(self, tracking_id: str, vector_content: str, original_content: str) -> dict:
        """Backup SQL memory before deletion, handling cases where SQL record might not exist."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Try to find by tracking_id first
                if tracking_id:
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE tracking_id = ?
                    """, (tracking_id,))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by tracking_id: {tracking_id}")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                    else:
                        logging.warning(f"BACKUP_SQL: No SQL record found for tracking_id: {tracking_id}")
                
                # Try to find by content matching (fuzzy approach)
                for content_to_try in [vector_content, original_content]:
                    if not content_to_try:
                        continue
                        
                    # Try exact match first
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE content = ?
                    """, (content_to_try,))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by exact content match")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                    
                    # Try partial match
                    cursor.execute("""
                        SELECT id, content, memory_type, source, weight, access_count, 
                            created_at, last_accessed, tags, metadata, tracking_id
                        FROM memories WHERE content LIKE ?
                    """, (f'%{content_to_try}%',))
                    result = cursor.fetchone()
                    
                    if result:
                        logging.info(f"BACKUP_SQL: Found memory by partial content match")
                        return {
                            'id': result[0], 'content': result[1], 'memory_type': result[2],
                            'source': result[3], 'weight': result[4], 'access_count': result[5],
                            'created_at': result[6], 'last_accessed': result[7], 'tags': result[8],
                            'metadata': result[9], 'tracking_id': result[10]
                        }
                
                # If no SQL record found, this might be an orphaned vector entry
                logging.warning("BACKUP_SQL: No matching SQL memory found - this may be an orphaned vector entry")
                return None
                
        except Exception as e:
            logging.error(f"BACKUP_SQL: Error backing up memory: {e}")
            return None

    def _delete_sql_with_identifiers(self, tracking_id: str, vector_content: str, original_content: str) -> bool:
        """Delete from SQL using multiple identifier approaches."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Try tracking_id first (most reliable)
                if tracking_id:
                    cursor.execute("DELETE FROM memories WHERE tracking_id = ?", (tracking_id,))
                    if cursor.rowcount > 0:
                        conn.commit()
                        logging.info(f"SQL_DELETE: Success via tracking_id: {tracking_id}")
                        return True
                
                # Try exact content matches
                for content_to_try in [vector_content, original_content]:
                    if not content_to_try:
                        continue
                        
                    cursor.execute("DELETE FROM memories WHERE content = ?", (content_to_try,))
                    if cursor.rowcount > 0:
                        conn.commit()
                        logging.info(f"SQL_DELETE: Success via content match")
                        return True
                
                logging.warning("SQL_DELETE: No matches found for deletion")
                return False
                
        except Exception as e:
            logging.error(f"SQL_DELETE: Error: {e}")
            return False

    def _delete_vector_with_identifiers(self, vector_id: str, tracking_id: str, vector_content: str) -> bool:
        """Delete from Vector database using multiple approaches."""
        try:
            from qdrant_client.http import models as rest
            
            # Approach 1: Delete by vector ID
            if vector_id:
                try:
                    self.vector_db.delete_by_id(vector_id)
                    logging.info(f"VECTOR_DELETE: Success via vector ID: {vector_id}")
                    return True
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via vector ID: {e}")
            
            # Approach 2: Delete by tracking_id filter
            if tracking_id:
                try:
                    delete_filter = rest.Filter(
                        must=[rest.FieldCondition(
                            key="tracking_id", 
                            match=rest.MatchValue(value=tracking_id)
                        )]
                    )
                    
                    self.vector_db.delete(
                        collection_name="deepseek_memory",
                        points_selector=rest.FilterSelector(filter=delete_filter)
                    )
                    logging.info(f"VECTOR_DELETE: Success via tracking_id filter: {tracking_id}")
                    return True
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via tracking_id filter: {e}")
            
            # Approach 3: Search and delete by content
            if vector_content:
                try:
                    search_results = self.vector_db.search(
                        query=vector_content,
                        mode="selective",
                        k=3
                    )
                    
                    for result in search_results:
                        result_id = result.get('id')
                        similarity = result.get('similarity_score', 0)
                        
                        if result_id and similarity > 0.9:  # High confidence match
                            try:
                                self.vector_db.delete_by_id(result_id)
                                logging.info(f"VECTOR_DELETE: Success via content search, ID: {result_id}")
                                return True
                            except Exception as del_e:
                                logging.warning(f"VECTOR_DELETE: Failed to delete ID {result_id}: {del_e}")
                    
                except Exception as e:
                    logging.warning(f"VECTOR_DELETE: Failed via content search: {e}")
            
            logging.error("VECTOR_DELETE: All approaches failed")
            return False
            
        except Exception as e:
            logging.error(f"VECTOR_DELETE: Critical error: {e}")
            return False

    def _rollback_sql_deletion(self, backup_info: dict) -> bool:
        """Rollback SQL deletion by restoring the backed-up memory."""
        try:
            if not backup_info:
                logging.error("ROLLBACK_SQL: No backup info available")
                return False
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Restore the memory
                cursor.execute("""
                    INSERT INTO memories 
                    (content, memory_type, source, weight, access_count, created_at, 
                    last_accessed, tags, metadata, tracking_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backup_info['content'], backup_info['memory_type'], backup_info['source'],
                    backup_info['weight'], backup_info['access_count'], backup_info['created_at'],
                    backup_info['last_accessed'], backup_info['tags'], backup_info['metadata'],
                    backup_info['tracking_id']
                ))
                
                conn.commit()
                logging.info("ROLLBACK_SQL: Successfully restored memory")
                return True
                
        except Exception as e:
            logging.error(f"ROLLBACK_SQL: Failed to rollback: {e}")
            return False
    
    def delete_memory_by_id_with_coordination(self, memory_id):
        """Delete a memory by ID with coordination between both databases.
        
        Args:
            memory_id: The database ID of the memory
            
        Returns:
            bool: Success status
        """
        try:
            # Convert memory_id to integer if it's a string
            if isinstance(memory_id, str) and memory_id.isdigit():
                memory_id = int(memory_id)
                
            if not isinstance(memory_id, int):
                logging.error(f"Invalid memory_id type: {type(memory_id)}. Expected int.")
                return False
                
            # Step 1: Get the memory details from SQLite
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content, tracking_id FROM memories
                    WHERE id = ?
                """, (memory_id,))
                
                result = cursor.fetchone()
                if not result:
                    logging.warning(f"No memory found with ID {memory_id}")
                    return False
                    
                content, tracking_id = result
                
                # Step 2: Delete from SQLite
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                sqlite_success = cursor.rowcount > 0
                conn.commit()
                
                if not sqlite_success:
                    logging.warning(f"Failed to delete memory {memory_id} from SQLite")
                    return False
                    
                # Step 3: Delete from vector database using tracking_id or content
                vector_success = False
                if tracking_id and hasattr(self, 'vector_db') and self.vector_db:
                    try:
                        # Try to delete by tracking_id
                        self.vector_db.delete_by_metadata({"tracking_id": tracking_id})
                        vector_success = True
                        logging.info(f"Deleted memory with tracking_id {tracking_id} from vector database")
                    except Exception as e:
                        logging.error(f"Failed to delete from vector database by tracking_id: {e}")
                        
                        # Fall back to content-based deletion
                        try:
                            if content:
                                self.vector_db.delete_text(content)
                                vector_success = True
                                logging.info(f"Deleted memory with content from vector database")
                        except Exception as content_error:
                            logging.error(f"Failed to delete from vector database by content: {content_error}")
                
                return sqlite_success or vector_success
                
        except Exception as e:
            logging.error(f"Error deleting memory by ID: {e}", exc_info=True)
            return False

    def debug_tracking_id_sync(self, tracking_id: str):
        """Debug method to check tracking_id sync between databases."""
        try:
            logging.info(f"DEBUG_SYNC: Checking tracking_id: {tracking_id}")
            
            # Check SQL
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, content FROM memories WHERE tracking_id = ?", (tracking_id,))
                sql_result = cursor.fetchone()
                
                if sql_result:
                    logging.info(f"DEBUG_SYNC: SQL found - ID: {sql_result[0]}, Content: {sql_result[1][:50]}...")
                else:
                    logging.warning(f"DEBUG_SYNC: SQL NOT found for tracking_id: {tracking_id}")
            
            # Check Vector
            from qdrant_client.http import models as rest
            
            search_filter = rest.Filter(
                must=[rest.FieldCondition(
                    key="tracking_id", 
                    match=rest.MatchValue(value=tracking_id)
                )]
            )
            
            vector_results = self.vector_db.scroll(
                collection_name="deepseek_memory",
                scroll_filter=search_filter,
                limit=5
            )
            
            if vector_results[0]:  # scroll returns (points, next_page_offset)
                logging.info(f"DEBUG_SYNC: Vector found {len(vector_results[0])} entries")
                for point in vector_results[0]:
                    logging.info(f"DEBUG_SYNC: Vector point ID: {point.id}, payload: {point.payload}")
            else:
                logging.warning(f"DEBUG_SYNC: Vector NOT found for tracking_id: {tracking_id}")
                
        except Exception as e:
            logging.error(f"DEBUG_SYNC: Error checking sync: {e}")
    
    def update_system_prompt(self, additional_prompt: str) -> str:
        """Update the system prompt file."""
        try:
            cleaned_prompt = additional_prompt.strip()
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                current_content = f.read()
            separator = "\n\n" if not current_content.endswith("\n\n") else "\n"
            with open(self.system_prompt_file, 'a', encoding='utf-8') as f:
                f.write(f"{separator}{cleaned_prompt}")
            self._initialize_system_prompt()
            self.llm = self._initialize_llm()
            logging.info(f"System prompt updated with: {cleaned_prompt}")
            return f"Successfully updated system prompt with: {cleaned_prompt}"
        except Exception as e:
            logging.error(f"Error updating system prompt: {e}")
            return f"Failed to update system prompt: {str(e)}"
    
    def _parse_params(self, params_str: str) -> dict:
        """Parse parameter string into a dictionary."""
        params = {}
        if params_str:
            param_parts = params_str.split('|')
            for part in param_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key.strip()] = value.strip()
        return params

    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence value, ensuring it's between 0.1 and 1.0."""
        try:
            confidence = float(confidence_str)
            return max(0.1, min(1.0, confidence))
        except (ValueError, TypeError):
            return 0.5

    def _format_summary_date(self, date_str: str) -> str:
        """Format a date string with robust error handling for display."""
        if not date_str or not isinstance(date_str, str):
            return "Unknown date"
        
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    date_obj = dt.strptime(date_str.split('.')[0], fmt)
                    return date_obj.strftime("%b %d, %Y")
                except ValueError:
                    continue
                
            # If we couldn't parse with standard formats, check if it's an ISO format
            if 'T' in date_str:
                try:
                    date_parts = date_str.split('T')[0]
                    date_obj = dt.strptime(date_parts, '%Y-%m-%d')
                    return date_obj.strftime("%b %d, %Y")
                except ValueError:
                    pass
                
            # Return original if we can't parse it
            return date_str
        except Exception as e:
            logging.error(f"Error formatting date: {e}")
            return "Unknown date"

    def _assess_for_memory_storage(self, user_input: str, response: str) -> None:
        """
        Forward memory storage commands to deepseek_enhancer for processing.
        This is now a simple wrapper to maintain backward compatibility and 
        ensure all commands are processed through a single path.
        """
        try:
            # Guard against None or invalid response
            if response is None or not isinstance(response, str) or not response.strip():
                logging.warning("Received None, non-string, or empty response in _assess_for_memory_storage")
                return
    
            # No-op: Simply log that we're redirecting to deepseek_enhancer
            logging.info("_assess_for_memory_storage called - all command processing now handled by deepseek_enhancer")
        
            # All actual processing has already been done in deepseek_enhancer.process_response
            # This method now exists only for backward compatibility
        
        except Exception as e:
            logging.error(f"Error in _assess_for_memory_storage wrapper: {e}", exc_info=True)