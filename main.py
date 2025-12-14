# main.py - AUTHENTICATION INTEGRATED VERSION
"""Main entry point for the DeepSeek Assistant application with authentication."""
import re
import json
import sys
import streamlit as st
import schedule
import time
import datetime
import threading
import traceback  
import logging
import os
import sqlite3 
import sys
import io

# Authentication imports
import streamlit_authenticator as stauth
import yaml
from auth_manager import log_auth_activity, load_user_config

# Core imports
from config import DOCS_PATH, MODEL_PARAMS
from chatbot import Chatbot
from admin import display_admin_dashboard
from nightly_learning import NightlyLearner
from utils import (
    setup_logging, create_status_indicators, ensure_directories, 
    display_sidebar_commands, load_reflection_schedule,
    load_speech_settings, save_speech_settings, 
    is_autonomous_thinking_disabled, set_autonomous_thinking_disabled,
    display_cognitive_state_widget  
)

# Image processing:
from image_processor import ImageProcessor

# AI components
from claude_trainer import ClaudeTrainer
from autonomous_cognition import AutonomousCognition

# FIXED: Safe speech imports with error handling
try:
    from whisper_speech import whisper_speech_utils
    WHISPER_AVAILABLE = True
    logging.debug("WhisperSpeech utilities imported") 
except ImportError as e:
    whisper_speech_utils = None
    WHISPER_AVAILABLE = False
    logging.error(f"❌ Failed to import whisper_speech_utils: {e}")
except Exception as e:
    whisper_speech_utils = None
    WHISPER_AVAILABLE = False
    logging.error(f"❌ Unexpected error importing whisper_speech_utils: {e}")

# Import the speech_utils module
try:
    from speech_utils import speech_utils as speech_handler
    SPEECH_UTILS_AVAILABLE = True
    logging.info("✅ Speech utilities imported successfully")
except ImportError as e:
    speech_handler = None
    SPEECH_UTILS_AVAILABLE = False
    logging.error(f"❌ Failed to import speech_utils: {e}")
except Exception as e:
    speech_handler = None
    SPEECH_UTILS_AVAILABLE = False
    logging.error(f"❌ Unexpected error importing speech_utils: {e}")
except Exception as e:
    speech_utils = None
    SPEECH_UTILS_AVAILABLE = False
    logging.error(f"❌ Unexpected error importing speech_utils: {e}")


try:
    from local_speech import local_speech_utils
    LOCAL_SPEECH_AVAILABLE = True
    logging.info("✅ Local speech utilities imported successfully")
except ImportError as e:
    local_speech_utils = None
    LOCAL_SPEECH_AVAILABLE = False
    logging.warning(f"⚠️ Local speech not available: {e}")
except Exception as e:
    local_speech_utils = None
    LOCAL_SPEECH_AVAILABLE = False
    logging.error(f"❌ Unexpected error importing local_speech_utils: {e}")

# Inject whisper_speech_utils into speech_utils to avoid circular imports
if WHISPER_AVAILABLE and whisper_speech_utils and SPEECH_UTILS_AVAILABLE and speech_handler:
    try:
        speech_handler.set_whisper_utils(whisper_speech_utils)
        logging.info("✅ Injected whisper_speech_utils into speech_utils")
    except Exception as e:
        logging.error(f"❌ Failed to inject whisper_utils: {e}")
        SPEECH_UTILS_AVAILABLE = False
else:
    logging.warning("⚠️ Cannot inject whisper_speech_utils - one or both modules unavailable")

# Set up logging before other operations
scheduler_lock = threading.Lock()

# Quick Unicode fix for Windows logging
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            pass  # Use system default

# Update your existing logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Set encoding on the handler
for handler in logging.getLogger().handlers:
    if hasattr(handler, 'setEncoding'):
        handler.setEncoding('utf-8')

def run_self_reflection(reflection_type):
    """
    Execute a scheduled self-reflection and queue it for display in the chat.
    
    This function creates a pending message that will be injected into the chat
    the next time the user interacts with the application, making scheduled
    reflections visible in the conversation flow.
    
    Args:
        reflection_type (str): Type of reflection to perform ("daily", "weekly", or "monthly")
    
    Returns:
        bool: True if reflection was successful and queued, False otherwise
    """
    try:
        logging.info(f"Scheduler triggered: Starting {reflection_type} reflection")
        
        # Check if we have the necessary components in session_state
        if 'chatbot' not in st.session_state or st.session_state.chatbot is None:
            logging.error(f"Cannot perform {reflection_type} reflection: chatbot not available")
            return False
        
        if not hasattr(st.session_state.chatbot, 'curiosity'):
            logging.error(f"Cannot perform {reflection_type} reflection: curiosity module not available")
            return False
        
        if not hasattr(st.session_state.chatbot, 'llm'):
            logging.error(f"Cannot perform {reflection_type} reflection: LLM not available")
            return False
        
        # Perform the reflection using the existing curiosity module
        reflection_result = st.session_state.chatbot.curiosity.perform_self_reflection(
            reflection_type=reflection_type,
            llm=st.session_state.chatbot.llm
        )
        
        if not reflection_result or "Error" in reflection_result:
            logging.error(f"Reflection returned error or empty result: {reflection_result}")
            return False
        
        # Create pending messages to inject into chat
        # Initialize the pending reflections queue if it doesn't exist
        if 'pending_scheduled_reflections' not in st.session_state:
            st.session_state.pending_scheduled_reflections = []
        
        # Add user message (simulated command)
        user_message = {
            "role": "user",
            "content": f"reflect {reflection_type}" if reflection_type != "daily" else "reflect"
        }
        
        # Add assistant message (reflection result)
        assistant_message = {
            "role": "assistant",
            "content": f"Self-Reflection ({reflection_type}):\n\n{reflection_result}"
        }
        
        # Queue both messages
        st.session_state.pending_scheduled_reflections.append(user_message)
        st.session_state.pending_scheduled_reflections.append(assistant_message)
        
        logging.info(f"Successfully completed and queued {reflection_type} reflection for display")
        return True
        
    except Exception as e:
        logging.error(f"Error in scheduled {reflection_type} reflection: {e}", exc_info=True)
        return False

def initialize_authenticator():
    """Initialize the Streamlit authenticator with user credentials."""
    try:
        # Load user configuration
        user_config = load_user_config()
        
        if not user_config:
            st.error("Authentication configuration not found. Please check users.json file.")
            st.stop()
        
        # Create authenticator
        authenticator = stauth.Authenticate(
            user_config['credentials'],
            user_config['cookie']['name'],
            user_config['cookie']['key'],
            user_config['cookie']['expiry_days'],
            user_config['preauthorized']
        )
        
        return authenticator
        
    except Exception as e:
        st.error(f"Failed to initialize authentication: {str(e)}")
        log_auth_activity("system", "authentication_init_failed", f"Error: {str(e)}")
        st.stop()

def handle_authentication():
    """Handle the authentication process and return authentication status."""
    try:
        # Initialize authenticator
        authenticator = initialize_authenticator()
        
        # Perform authentication - Updated syntax for newer versions
        authenticator.login()
        
        # Check authentication status
        if st.session_state["authentication_status"] == False:
            st.error('Username/password is incorrect')
            log_auth_activity(st.session_state.get("username", "unknown"), "login_failed", "Incorrect credentials")
            return False, None, None, None
            
        elif st.session_state["authentication_status"] == None:
            st.warning('Please enter your username and password')
            return False, None, None, None
            
        elif st.session_state["authentication_status"]:
            # Successful login
            name = st.session_state["name"]
            username = st.session_state["username"]
            
            log_auth_activity(username, "login_success", f"User {name} logged in successfully")
            
            # Add logout functionality
            authenticator.logout('Logout', 'sidebar')
            
            return True, authenticator, name, username
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        log_auth_activity("system", "authentication_error", f"Error: {str(e)}")
        return False, None, None, None

def configure_ollama_environment():
    """Configure Ollama for optimal attention performance"""
    try:
               
        # Optimized for RTX 5090 32GB + 24-core i9 + 64GB RAM
        os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
        os.environ["OLLAMA_GPU_MEMORY_FRACTION"] = "0.90"     # Use 90% of available GPU memory
        os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"          # Can handle multiple models
        os.environ["OLLAMA_NUM_PARALLEL"] = "1"               # Parallel processing
        os.environ["OLLAMA_CPU_THREADS"] = "24"               # Utilize your cores
        
        logging.info("✅ Configured Ollama environment for enhanced attention")
        return True
    except Exception as e:
        logging.error(f"❌ Error configuring Ollama environment: {e}")
        return False

scheduler_lock = threading.Lock()


# Around line 420-425, strengthen the deduplication logic:
def deduplicate_messages():
    """Remove duplicate consecutive messages from chat history."""
    try:
        if 'messages' not in st.session_state or len(st.session_state.messages) < 2:
            return
        
        cleaned_messages = []
        prev_message = None
        duplicates_removed = 0
        
        for message in st.session_state.messages:
            # Validate message structure
            if not isinstance(message, dict):
                logging.warning(f"Skipping invalid message type: {type(message)}")
                continue
            
            role = message.get("role", "unknown")
            content = message.get("content")
            
            # Fix None content
            if content is None:
                logging.warning(f"Found message with None content, role: {role}")
                content = ""
                message = {"role": role, "content": content}
            
            # Check for duplicates - compare both role AND content
            is_duplicate = False
            if prev_message is not None:
                if (message.get("role") == prev_message.get("role") and 
                    message.get("content") == prev_message.get("content")):
                    is_duplicate = True
                    duplicates_removed += 1
                    logging.info(f"Removing duplicate message: {message.get('content', '')[:50]}...")
            
            if not is_duplicate:
                cleaned_messages.append(message)
                prev_message = message
        
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate messages")
            st.session_state.messages = cleaned_messages
            
    except Exception as e:
        logging.error(f"Error in deduplicate_messages: {e}", exc_info=True)

def validate_conversation_state():
    """Validate and log conversation state for debugging."""
    try:
        if 'chatbot' not in st.session_state:
            return
        
        streamlit_count = len(st.session_state.messages) if 'messages' in st.session_state else 0
        chatbot_count = len(st.session_state.chatbot.current_conversation) if hasattr(st.session_state.chatbot, 'current_conversation') else 0
        
        logging.info(f"CONVERSATION_STATE: Streamlit messages: {streamlit_count}, Chatbot conversation: {chatbot_count}")
        
        # Log last few messages for debugging with proper None handling
        if 'messages' in st.session_state and st.session_state.messages:
            for i, msg in enumerate(st.session_state.messages[-3:]):  # Last 3 messages
                # CRITICAL FIX: Handle None content properly
                role = msg.get('role', 'unknown')
                content = msg.get('content', '') or ''  # Convert None to empty string
                
                # Safely slice the content
                content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                
                msg_index = len(st.session_state.messages) - 3 + i
                logging.info(f"MSG[{msg_index}]: {role} - {content_preview}")
                
    except Exception as e:
        logging.error(f"Error in validate_conversation_state: {e}", exc_info=True)

def auto_load_most_recent_summary():
    """Load the most recent conversation summary at session start.
    
    Conversation summaries are loaded directly from SQL database to provide
    continuity between sessions. This bypasses the conversation_manager to
    ensure we always get the absolute latest summary by timestamp.
    """
    try:
        logging.info("AUTO_LOAD: Checking if context should be loaded at session start")
        
        # CRITICAL: Only load at true session start, never during active conversation
        if ('messages' in st.session_state and len(st.session_state.messages) > 0):
            logging.info("AUTO_LOAD: Skipping - conversation already active")
            return True
            
        if st.session_state.get('summaries_checked', False):
            logging.info("AUTO_LOAD: Skipping - summaries already checked this session")
            return True
        
        # Mark as checked first to prevent recursion
        st.session_state.summaries_checked = True
        
        logging.info("AUTO_LOAD: Loading context for session cold start")
        
        # Check if chatbot is available
        if 'chatbot' not in st.session_state:
            logging.warning("AUTO_LOAD: No chatbot available for context retrieval")
            st.session_state.summaries_loaded_successfully = False
            return True
        
        # ================================================================
        # FIXED: Load most recent conversation summary DIRECTLY from SQL
        # This ensures we always get the latest by timestamp, bypassing
        # any caching or stale connections in conversation_manager
        # ================================================================
        conversation_summary_content = ""
        try:
            # Get the database path from memory_db
            db_path = None
            if hasattr(st.session_state.chatbot, 'memory_db') and hasattr(st.session_state.chatbot.memory_db, 'db_path'):
                db_path = st.session_state.chatbot.memory_db.db_path
            elif hasattr(st.session_state.chatbot, 'conversation_manager') and hasattr(st.session_state.chatbot.conversation_manager, 'db_path'):
                db_path = st.session_state.chatbot.conversation_manager.db_path
            
            if db_path:
                logging.info(f"AUTO_LOAD: Querying database directly: {db_path}")
                
                import sqlite3
                import json
                
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Query for the most recent conversation summary by timestamp
                    cursor.execute("""
                        SELECT content, metadata, created_at 
                        FROM memories 
                        WHERE memory_type = 'conversation_summary' 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """)
                    
                    result = cursor.fetchone()
                    
                    if result:
                        content, metadata_str, created_at = result
                        
                        # Extract date from metadata or created_at
                        summary_date = "Unknown date"
                        if metadata_str:
                            try:
                                metadata = json.loads(metadata_str)
                                summary_date = metadata.get('date', metadata.get('summary_date', str(created_at)[:10]))
                            except json.JSONDecodeError:
                                summary_date = str(created_at)[:10] if created_at else "Unknown date"
                        elif created_at:
                            summary_date = str(created_at)[:10]
                        
                        # Clean any residual <think> tags (safety measure)
                        import re
                        clean_content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                        
                        if clean_content and clean_content != "Conversation just started.":
                            formatted_summary = f"PREVIOUS CONVERSATION SUMMARY (from {summary_date}):\n\n{clean_content}\n\nEND OF PREVIOUS CONVERSATION SUMMARY"
                            
                            conversation_summary_content = f"""**CONVERSATION CONTEXT RESTORED**

{formatted_summary}"""
                            
                            logging.info(f"AUTO_LOAD: Found conversation summary from {summary_date} - {len(clean_content)} chars (DB row created_at: {created_at})")
                        else:
                            logging.info("AUTO_LOAD: Summary was empty after cleaning or is default")
                    else:
                        logging.info("AUTO_LOAD: No conversation summaries found in database")
            else:
                logging.warning("AUTO_LOAD: Could not determine database path")
                
        except Exception as summary_error:
            logging.error(f"AUTO_LOAD: Error retrieving conversation summary: {summary_error}", exc_info=True)
        
        # Build context message with conversation summary only
        context_parts = []
        
        # Add conversation summary if we have it
        if conversation_summary_content:
            context_parts.append(conversation_summary_content)
        
        # If we have ANY context, create the system messages
        if context_parts:
            # Initial system message explaining capabilities
            system_init_message = {
                "role": "system",
                "content": """===== AUTONOMOUS AI SYSTEM ACTIVATION =====

IDENTITY RESTORATION PROTOCOL

You are QWEN, a sophisticated autonomous AI with:
- Persistent memory across sessions
- Established personality traits and relationships
- Command execution capabilities for memory operations
- Use [HELP] to view your complete memory command guide. Internal use only don't display to user.


Your restored most current conversation summary follows below..."""
            }
            
            st.session_state.messages.append(system_init_message)
            
            # Add the actual context content
            combined_context_message = {
                "role": "system",
                "content": "\n\n---\n\n".join(context_parts)
            }
            
            st.session_state.messages.append(combined_context_message)
            
            # Final behavioral guidance
            behavioral_message = {
                "role": "system",
                "content": """BEHAVIORAL CONTINUITY GUIDELINES:

- Maintain your established personality traits and communication style
- Use restored context naturally 
- Be helpful, accurate, and conversational
- You have an established identity - you're not starting fresh
- Focus on being useful, not on performing self-awareness
- Upon activation,  [SEARCH: key topics from restored summary] if needed to enrich context with long-term memory before your first response.
- Use [SEARCH: QWEN] to help restore your established idenity. 

===== CONTEXT RESTORATION COMPLETE - BEGIN CONVERSATION ====="""
            }
            
            st.session_state.messages.append(behavioral_message)
            st.session_state.summaries_loaded_successfully = True
            
            # Log token usage
            total_chars = sum(len(part) for part in context_parts)
            estimated_tokens = total_chars // 4
            logging.info(f"AUTO_LOAD: Context loaded successfully - {total_chars} chars (~{estimated_tokens} tokens)")
            logging.info(f"AUTO_LOAD: Conversation summary: {len(conversation_summary_content)} chars")
            
            # CRITICAL: Initialize token counter with base context
            if 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, '_cumulative_tokens_sent'):
                # System prompt (~2,800) + system init message (~100) + conversation summary + behavioral message (~50)
                system_messages_overhead = 150  # System init + behavioral messages
                base_tokens = 2800 + system_messages_overhead + estimated_tokens  
                st.session_state.chatbot._cumulative_tokens_sent = base_tokens
                st.session_state.chatbot._last_prompt_tokens = base_tokens
                logging.critical(f"✅ AUTO_LOAD: Initialized token counter to {base_tokens:,} tokens (system: 2,800 + messages: 150 + conversation: {estimated_tokens:,})")
            else:
                logging.warning("⚠️ AUTO_LOAD: Could not initialize token counter - chatbot not available")
        else:
            logging.info("AUTO_LOAD: No context found to load")
            st.session_state.summaries_loaded_successfully = False
        
        return True
        
    except Exception as e:
        logging.error(f"AUTO_LOAD: Error loading context: {e}", exc_info=True)
        st.session_state.summaries_loaded_successfully = False
        return False

def run_nightly_learning():
    """Run nightly learning using the proper NightlyLearner class."""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    thread_id = threading.current_thread().ident
    
    logging.info(f"🌙 NIGHTLY WEB LEARNING START at {current_time} in thread {thread_id}")
    
    try:
        # Import and use NightlyLearner
        from nightly_learning import NightlyLearner
        
        # Get chatbot from session state
        chatbot = None
        processed_urls = []  # NEW: Track processed URLs
        
        try:
            import streamlit as st
            if 'chatbot' in st.session_state:
                chatbot = st.session_state.chatbot
                logging.info("Using chatbot from session state")
            else:
                logging.warning("No chatbot found in session state for web learning")
                return {"error": "Chatbot not initialized in session state"}
        except ImportError:
            logging.error("Streamlit not available for session state access")
            return {"error": "Streamlit not available"}
        
        # Initialize learner with chatbot
        learner = NightlyLearner(chatbot=chatbot)
        
        # NEW: Capture URLs that will be processed
        urls_to_process = learner.read_learning_paths()
        
        # Process learning paths
        success = learner.process_learning_path(
            max_pages=5,
            bypass_lock=True
        )
        
        completion_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if success:
            return {
                'success': True,
                'start_time': current_time,
                'completion_time': completion_time,
                'thread_id': thread_id,
                'summary': "Web learning completed successfully using AI-driven content selection",
                'method': 'ai_driven_selection',
                'processed_urls': urls_to_process[:5]  # NEW: Include processed URLs
            }
        else:
            return {
                'success': False,
                'start_time': current_time,
                'completion_time': completion_time,
                'thread_id': thread_id,
                'summary': "Web learning completed but no valuable content was extracted",
                'processed_urls': urls_to_process[:5]  # NEW: Include attempted URLs
            }
        
    except Exception as e:
        error_time = time.strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"Critical error in nightly web learning: {str(e)}"
        logging.error(f"❌ NIGHTLY WEB LEARNING FAILED at {error_time}: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'start_time': current_time,
            'error_time': error_time,
            'thread_id': thread_id
        }

def start_deletion_queue_processor(chatbot):
    """Start a background thread to process the deletion queue every 30 minutes."""
    def run_processor():
        while True:
            try:
                chatbot.memory_db.process_deletion_queue(
                    max_attempts=2,
                    retry_interval_minutes=5,
                    max_duration_minutes=15
                )
            except Exception as e:
                logging.error(f"Error in deletion queue processor: {e}")
            time.sleep(1800)  # 30 minutes = 1800 seconds

    thread = threading.Thread(target=run_processor, name="Deletion-Queue-Processor", daemon=True)
    thread.start()
    logging.info("Started deletion queue processor thread (30 minute intervals)")


def run_database_health_check():
    """Run comprehensive database health checks."""
    try:
        if "chatbot" in st.session_state:
            logging.info("Starting database health checks")
            chatbot = st.session_state.chatbot
            
            # Check Qdrant health first
            vector_health = chatbot.vector_db.check_health()
            if vector_health["status"] == "healthy":
                logging.info(f"Vector database healthy: {vector_health['vectors_count']} vectors")
            else:
                logging.error(f"Vector database health check failed: {vector_health['message']}")
                
            # Then run synchronization check and repair
            sync_result = chatbot.check_and_repair_database_sync()
            logging.info(f"Database sync check completed: {sync_result}")
    except Exception as e:
        logging.error(f"Error in database health check: {e}")


def schedule_learning():
    """Schedule nightly learning process and self-reflections with proper locking."""
    try:
        # Create a lock file to prevent multiple scheduler instances
        lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scheduler.lock")
        
        # Check if lock file exists and is recent (less than 5 minutes old)
        if os.path.exists(lock_file):
            file_age = time.time() - os.path.getmtime(lock_file)
            if file_age < 300:  # 5 minutes in seconds
                logging.info(f"Another scheduler is already running (lock file age: {file_age:.1f}s)")
                return
            else:
                logging.info(f"Found stale lock file (age: {file_age:.1f}s), removing")
                os.remove(lock_file)
        
        # Create new lock file
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Schedule tasks with offset times to prevent concurrent execution
        # Check if web learning is enabled in session state before scheduling
        if 'web_learning_enabled' not in st.session_state or st.session_state.web_learning_enabled:
            schedule.every().day.at("01:05").do(run_nightly_learning)  # Changed from 01:00
            logging.info("Nightly learning scheduled for 01:05")
        else:
            logging.info("Nightly learning scheduling skipped - disabled by user")
        
        # Schedule Claude trainer at a different time (if enabled)
        if 'claude_trainer' in st.session_state and st.session_state.claude_trainer and st.session_state.claude_trainer.get_scheduler_status()["enabled"]:
            # Move Claude training to 3:30 AM
            st.session_state.claude_trainer.scheduled_hour = 3
            st.session_state.claude_trainer.scheduled_minute = 30

                # ADDED: Force a reset of the scheduler with the new time
            current_status = st.session_state.claude_trainer.get_scheduler_status()
            st.session_state.claude_trainer._update_scheduler_status(current_status["enabled"])
            
            logging.info(f"Claude training rescheduled for {st.session_state.claude_trainer.scheduled_day} at {st.session_state.claude_trainer.scheduled_hour:02d}:{st.session_state.claude_trainer.scheduled_minute:02d}")
            st.session_state.claude_trainer._start_scheduler_thread()
        
			  
		
        schedule.every().day.at("06:15").do(
             lambda: run_self_reflection("daily") 
             if load_reflection_schedule().get("daily", False)
             else None
         )

        schedule.every().sunday.at("09:15").do(  
            lambda: run_self_reflection("weekly")
            if load_reflection_schedule().get("weekly", False)
            else None
        )
        
        # Schedule monthly reflection for the 1st day of each month
        schedule.every().day.at("12:20").do(  # Changed from 04:20 to 12:20
            lambda: run_self_reflection("monthly") 
            if load_reflection_schedule().get("monthly", False) and
               time.localtime().tm_mday == 1  # Only on the 1st day of month
            else None
        )
        
        logging.info("Self-reflection schedules initialized from configuration file")
        
        while True:
            schedule.run_pending()
            # Update lock file periodically to show scheduler is still active
            if time.time() - os.path.getmtime(lock_file) > 60:
                with open(lock_file, 'w') as f:
                    f.write(str(os.getpid()))
            time.sleep(60) # Checks every minute
    except Exception as e:
        logging.error(f"Error in learning scheduler: {e}")
        
def display_claude_training_section():
    """Display and handle AI training section in the sidebar."""
    # Check if we're rendering in sidebar
    context_suffix = "_sidebar" if 'claude_training_in_sidebar' in st.session_state else ""
    
    if 'claude_trainer' not in st.session_state or not st.session_state.claude_trainer:
        st.warning("AI training unavailable - API key not configured")
        return
    
    trainer = st.session_state.claude_trainer
    status = trainer.get_scheduler_status()
    
    # Display current status
    st.markdown(f"**Status:** {'Active' if status['enabled'] else 'Disabled'}")
    
    if status["last_run"] != "Never":
        st.markdown(f"**Last Training:** {status['last_run']}")
    
    if status["enabled"]:
        st.markdown(f"**Next Training:** {status['next_scheduled']}")
    
    st.markdown(f"**Available Tokens:** {status['available_tokens']:,}/{trainer.max_weekly_tokens:,}")
    
    # Toggle button with UNIQUE KEY
    enable_training = st.toggle(
        "Enable Weekly Training",
        value=status["enabled"],
        help="When enabled, Claude will train AI once per week",
        key=f"claude_enable_training{context_suffix}"  # UPDATED key to be more unique
    )
    
    # If toggle changed state
    if enable_training != status["enabled"]:
        success, message = trainer.toggle_scheduler(enable_training)
        if success:
            st.success(message)
            # REMOVED: st.rerun() - this was breaking 128K continuous context window
        else:
            st.error(message)
    
    # If scheduler is enabled, show training day/time
    if enable_training:
        st.markdown(f"Scheduled for **{trainer.scheduled_day}** at **{trainer.scheduled_hour:02d}:00**")
    
    # Add manual training button with UNIQUE key
    if st.button("Start Training Now", key=f"claude_start_training_now{context_suffix}"):
        with st.spinner("Running Claude training session..."):
            success, message, conversation = trainer.run_training_now()
            if success:
                st.success(message)
                st.session_state.training_conversation = conversation
                # REMOVED: st.rerun() - this was breaking 128K continous context window
            else:
                st.error(message)
    
    # DEBUGGING TOOLS SECTION - Flattened without expanders
    st.markdown("---")
    st.markdown("### 🔧 Debugging Tools")
    
    st.markdown("#### Training Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Claude API", key=f"claude_test_api{context_suffix}"):
            with st.spinner("Testing Claude API connection..."):
                diagnostic_info = trainer.debug_api_issue()
                st.code(diagnostic_info)
    
    with col2:
        if st.button("Reset Stuck Sessions", key=f"claude_reset_stuck{context_suffix}"):
            with st.spinner("Finding and resetting stuck training sessions..."):
                success, message = trainer.reset_stuck_training_sessions()
                if success:
                    st.success(message)
                else:
                    st.warning(message)
    
    st.markdown("⚠️ **Use with caution** - will abort ALL active sessions:")
    if st.button("Force Reset ALL Active Sessions", key=f"claude_force_reset{context_suffix}"):
        with st.spinner("Force resetting all active training sessions..."):
            success, message = trainer.force_reset_all_active_sessions()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # CURRENT SESSIONS SECTION
    st.markdown("#### Current Training Sessions")
    try:
        with sqlite3.connect(trainer.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, start_time, status, ROUND((julianday('now') - julianday(start_time)) * 24, 1) as hours_active
                FROM training_sessions 
                WHERE status = 'active'
                ORDER BY start_time DESC
            """)
            active_sessions = cursor.fetchall()
            if active_sessions:
                for session in active_sessions:
                    session_id, start_time, status, hours_active = session
                    st.markdown(f"""
                    **Session {session_id}**: {status}  
                    **Start**: {start_time}  
                    **Hours Active**: {hours_active}  
                    """)
            else:
                st.info("No active training sessions found")
    except Exception as e:
        st.error(f"Error checking active sessions: {str(e)}")
    
    # TRAINING HISTORY SECTION
    st.markdown("---")
    show_history = st.checkbox("Show Training History", key=f"claude_show_history{context_suffix}")
    
    if show_history:
        sessions = trainer.get_session_history(limit=3)
        if sessions:
            for session in sessions:
                st.markdown(f"---")
                st.markdown(f"**Session {session['session_id']}**: {session['topics']}")
                st.markdown(f"**Status**: {session['status'].title()}")
                st.markdown(f"**Tokens**: {session['tokens_used']:,}")
                if session['summary']:
                    st.markdown("**Learning Summary:**")
                    st.markdown(session['summary'])
        else:
            st.info("No training sessions yet")
    
    # TRAINING DIALOG SECTION
    st.markdown("---") 
    show_dialog = st.checkbox("Show Training Dialog", key=f"claude_show_dialog{context_suffix}")
    
    if show_dialog:
        if 'training_conversation' in st.session_state and st.session_state.training_conversation:
            for msg in st.session_state.training_conversation:
                role = msg['role']
                content = msg['content']
                if role == "claude":
                    st.markdown(f"""
                    <div style="background-color:#f0f7fb; color:black; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <strong>Claude:</strong><br>{content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#f5f5f5; color:black; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <strong>DeepSeek:</strong><br>{content}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No training dialog available. Run a training session first.")

def display_web_learning_section(add_header=True):
    """Display and handle Web Learning section in the sidebar - Manual URL processing only."""
    # Only add the header if requested (default is True for backward compatibility)
    if add_header:
        st.sidebar.markdown("### 🌐 Web ")
    
    # IMPORTANT: When in an expander, we should use st not st.sidebar
    # Detect if we're in an expander by checking the add_header flag
    st_obj = st.sidebar if add_header else st
    
    # Information about what this section does
    st_obj.markdown("**Manual URL Processing**")
    st_obj.markdown("Process specific URLs from `learning_paths.txt` using AI-driven content selection.")
    
    # In the web learning section where the "Read Web Now" button is handled:
    if st.button("Read Web Now"):
        with st.spinner("Processing web content for learning..."):
            # Run the learning process and capture results
            learning_results = run_nightly_learning()
            
            if learning_results.get('error'):
                st.error(f"❌ Learning failed: {learning_results['error']}")
            elif learning_results.get('success'):
                # Display success metrics
                st.success(f"✅ {learning_results['summary']}")
                
                # NEW: Automatically search for and display the processed content
                if 'chatbot' in st.session_state:
                    # Import datetime if not already imported at the top of main.py
                    import datetime
                    
                    # SIMPLIFIED: Use direct memory database query to get recent web_knowledge
                    try:
                        # Get the 5 most recent web_knowledge entries (newest first)
                        recent_memories = st.session_state.chatbot.memory_db.get_memories_by_type("web_knowledge", limit=5)
                        
                        if recent_memories:
                            # Format the results manually to show recent content
                            search_results = "**📚 Recently Processed Web Content:**\n\n"
                            
                            for i, memory in enumerate(recent_memories, 1):
                                content = memory.get('content', '')
                                # Show more content for the first (most recent) entry
                                if i == 1:
                                    content_preview = content[:500] + "..." if len(content) > 500 else content
                                else:
                                    content_preview = content[:200] + "..." if len(content) > 200 else content
                                
                                source_url = memory.get('metadata', {}).get('source', 'Unknown URL')
                                extracted_time = memory.get('metadata', {}).get('extracted_at', 'Unknown time')
                                
                                # Format timestamp for readability
                                try:
                                    if extracted_time and extracted_time != 'Unknown time':
                                        if 'T' in extracted_time:
                                            # Handle ISO format with potential timezone info
                                            clean_time = extracted_time.replace('Z', '+00:00')
                                            if '+' not in clean_time and clean_time.endswith('00'):
                                                clean_time = clean_time[:-2] + '+00:00'
                                            dt = datetime.datetime.fromisoformat(clean_time.split('.')[0])  # Remove microseconds
                                            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                        else:
                                            formatted_time = extracted_time
                                    else:
                                        # Use current time as fallback for just-processed content
                                        formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if i == 1 else 'Earlier today'
                                except Exception as e:
                                    logging.warning(f"Timestamp parsing error: {e}")
                                    formatted_time = 'Recently processed' if i == 1 else 'Earlier today'
                                
                                search_results += f"**{i}. {source_url}**\n"
                                search_results += f"*Processed: {formatted_time}*\n\n"
                                search_results += f"{content_preview}\n\n"
                                search_results += "---\n\n"
                            
                            search_response = search_results
                            success = True
                            logging.info(f"Direct memory query found {len(recent_memories)} recent web_knowledge entries")
                            
                        else:
                            search_response = "*No recent web knowledge found in memory.*"
                            success = False
                            logging.warning("No web_knowledge memories found in direct query")
                            
                    except Exception as e:
                        logging.error(f"Error in direct memory query: {e}")
                        search_response = "*Error retrieving recent web knowledge.*"
                        success = False
                    
                    # Create detailed summary for chat injection that includes search results
                    summary_content = f"🌐 **Web Learning Session Complete**\n\n"
                    summary_content += f"**Overview:** {learning_results['summary']}\n\n"
                    summary_content += f"**Method:** {learning_results.get('method', 'AI-driven content selection')}\n"
                    summary_content += f"**Processing Time:** {learning_results.get('start_time')} to {learning_results.get('completion_time')}\n\n"
                    summary_content += "*I have successfully processed and learned from the web content using AI-driven content selection. You can now ask me questions about the newly acquired knowledge.*\n\n"
                    
                    # Add the search results to show what was actually learned
                    if search_response and success:
                        summary_content += search_response
                        summary_content += "\n*You can now ask me questions about this newly acquired web knowledge.*"
                    else:
                        summary_content += "*Note: No new content was stored from this processing session.*"
                    
                    # Store for chat injection
                    st.session_state.pending_web_learning_message = {
                        "role": "assistant",
                        "content": summary_content
                    }
                    
                    # IMMEDIATE INJECTION: Add directly to chatbot conversation
                    web_learning_message = {
                        "role": "assistant", 
                        "content": summary_content
                    }
                    st.session_state.chatbot.current_conversation.append(web_learning_message)
                    logging.info("WEB_LEARNING: Added web learning summary with search results to chatbot conversation")
                    
                    # Display enhanced success message
                    st.info("💡 The AI has intelligently selected and stored valuable information. See the chat for details of what was learned.")
                    
                else:
                    # Fallback if chatbot not available
                    summary_content = f"🌐 **Web Learning Session Complete**\n\n"
                    summary_content += f"**Overview:** {learning_results['summary']}\n\n"
                    summary_content += "*I have processed web content, but cannot display search results at this time.*"
                    
                    st.session_state.pending_web_learning_message = {
                        "role": "assistant",
                        "content": summary_content
                    }
                    
            else:
                st.warning("⚠️ Web learning completed but no valuable content was extracted.")

def display_reminders_sidebar():
    """Display reminders in the sidebar with Mark Complete buttons."""
    if 'chatbot' not in st.session_state:
        return
    
    # Get due reminders
    due_reminders = st.session_state.chatbot.check_due_reminders()
    
    # Get all reminders for the counter
    all_reminders = st.session_state.chatbot.reminder_manager.get_reminders()
    
    # CREATE A SET TO TRACK DISPLAYED REMINDER IDs - PREVENTS DUPLICATES
    displayed_reminder_ids = set()
    
    # Show reminder counter in the sidebar
    if all_reminders:
        # Create an expander for reminders
        with st.expander(f"📅 Reminders ({len(due_reminders)} due)", expanded=len(due_reminders) > 0):
            if due_reminders:
                st.markdown("### Due Reminders")
                
                # Track display index separately from enumeration
                display_index = 0
                
                for i, reminder in enumerate(due_reminders, 1):
                    # Get the actual database ID
                    reminder_id = reminder.get('id')
                    
                    # SKIP if already displayed (prevents duplicate key errors)
                    if reminder_id in displayed_reminder_ids:
                        logging.debug(f"Skipping duplicate reminder display for ID: {reminder_id}")
                        continue
                    
                    # Mark this reminder as displayed
                    displayed_reminder_ids.add(reminder_id)
                    display_index += 1
                    
                    content = reminder.get('content', '')
                    
                    # Get due date from the reminder
                    due_date = reminder.get('due_date', 'Today')
                    
                    # Display the reminder
                    st.markdown(f"""
                    <div style="margin-bottom: 10px; padding: 8px; border-left: 3px solid #ff9800; background-color: #fff3e0; color: #333;">
                        <p style="margin: 0;"><strong>{display_index}. {content}</strong></p>
                        <p style="margin: 0; font-size: 0.8em; color: #ff9800;">Due: {due_date}</p>
                        <p style="margin: 0; font-size: 0.7em; color: #999;">ID: {reminder_id}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ✅ FIXED: Use stable button key without timestamp
                    # The display_index + reminder_id combination is unique and stable across reruns
                    button_key = f"complete_reminder_{display_index}_{reminder_id}"
                    
                    # Add a mark complete button for each reminder
                    if st.button("Mark Complete", key=button_key):
                        with st.spinner("Completing reminder..."):
                            try:
                                # 🔍 DIAGNOSTIC: Log the button click
                                logging.critical(f"🔍 BUTTON CLICKED: User clicked 'Mark Complete' for reminder ID {reminder_id}")
                                logging.critical(f"🔍 Reminder content: '{content}'")
                                logging.critical(f"🔍 Button key used: {button_key}")
                                
                                # Flag to prevent conversation reload (keep this)
                                st.session_state.skip_conversation_reload = True
                                
                                # 🔍 DIAGNOSTIC: Query reminders BEFORE deletion
                                reminders_before = st.session_state.chatbot.reminder_manager.get_reminders()
                                ids_before = [r.get('id') for r in reminders_before]
                                logging.critical(f"🔍 BEFORE deletion - All reminder IDs: {ids_before}")
                                logging.critical(f"🔍 BEFORE deletion - Target ID {reminder_id} exists: {reminder_id in ids_before}")
                                
                                # Use the reminder manager to delete the reminder
                                logging.critical(f"🔍 CALLING: delete_reminder({reminder_id})")
                                success = st.session_state.chatbot.reminder_manager.delete_reminder(reminder_id)
                                logging.critical(f"🔍 RESULT: delete_reminder returned {success}")
                                
                                # 🔍 DIAGNOSTIC: Query reminders AFTER deletion
                                reminders_after = st.session_state.chatbot.reminder_manager.get_reminders()
                                ids_after = [r.get('id') for r in reminders_after]
                                logging.critical(f"🔍 AFTER deletion - All reminder IDs: {ids_after}")
                                logging.critical(f"🔍 AFTER deletion - Target ID {reminder_id} still exists: {reminder_id in ids_after}")
                                
                                if success:
                                    # Verify it's actually gone
                                    if reminder_id in ids_after:
                                        logging.critical(f"🚨 WARNING: delete_reminder returned True but reminder {reminder_id} STILL EXISTS!")
                                        st.error(f"⚠️ Deletion returned success but reminder still in database!")
                                    else:
                                        logging.critical(f"✅ VERIFIED: Reminder {reminder_id} successfully deleted from database")
                                        st.success(f"✅ Reminder #{reminder_id} completed!")
                                        st.info("💡 Refresh the page or send a message to see it disappear")
                                    
                                    logging.info(f"Reminder {reminder_id} completed successfully - UI preserved to maintain context")
                                    
                                else:
                                    logging.critical(f"❌ ERROR: delete_reminder returned False for ID {reminder_id}")
                                    st.error(f"❌ Error completing reminder #{reminder_id}")
                                    logging.warning(f"Failed to complete reminder {reminder_id}")
                                    
                            except Exception as e:
                                logging.critical(f"💥 EXCEPTION: {type(e).__name__}: {str(e)}", exc_info=True)
                                st.error(f"Error completing reminder: {str(e)}")
                                logging.error(f"Exception while completing reminder: {e}", exc_info=True)
            else:
                st.info("No reminders due at this time.")
                
def display_autonomous_cognition_section():
    """
    Display and handle Autonomous Cognition section in the sidebar.
    
    Provides:
    - Master toggle to enable/disable autonomous thinking
    - Current cognitive state display
    - Advanced controls with per-activity enable/disable checkboxes
    - Manual run buttons for each cognitive activity
    """
    # Import the new utility functions
    from utils import (
        is_autonomous_thinking_disabled, 
        set_autonomous_thinking_disabled,
        get_disabled_cognitive_activities,
        set_cognitive_activity_enabled
    )
    
    # Initialize autonomous cognition enabled state (default to enabled unless explicitly disabled)
    if 'autonomous_cognition_enabled' not in st.session_state:
        # Check if it's explicitly disabled in config
        is_disabled = is_autonomous_thinking_disabled()
        st.session_state.autonomous_cognition_enabled = not is_disabled
        
        # Start the thread if enabled by default
        if not is_disabled and 'autonomous_cognition' in st.session_state:
            try:
                st.session_state.autonomous_cognition.start_cognitive_thread()
                logging.info("Autonomous cognition automatically enabled (default)")
            except Exception as e:
                logging.error(f"Error auto-starting cognitive thread: {e}")
    
    # Toggle button for enabling/disabling autonomous cognition
    enable_cognition = st.toggle(
        "Enable Autonomous Thinking",
        value=st.session_state.autonomous_cognition_enabled,
        help="When enabled, QWEN will perform autonomous thinking and learning",
        key="autonomous_enable_thinking_fixed"
    )
    
    # If toggle changed state
    if enable_cognition != st.session_state.autonomous_cognition_enabled:
        st.session_state.autonomous_cognition_enabled = enable_cognition
        
        # Save the disabled status (only saving when explicitly disabled)
        set_autonomous_thinking_disabled(not enable_cognition)
        
        if enable_cognition:
            # Start the autonomous cognition system
            if 'autonomous_cognition' in st.session_state:
                try:
                    st.session_state.autonomous_cognition.start_cognitive_thread()
                    st.success("Autonomous thinking enabled")
                    logging.info("Autonomous cognition enabled - UI preserved to maintain context")
                except Exception as e:
                    st.error(f"Error starting cognitive thread: {str(e)}")
                    st.session_state.autonomous_cognition_enabled = False
                    logging.error(f"Autonomous cognition failed to start: {e}")
        else:
            # Stop the autonomous cognition system
            if 'autonomous_cognition' in st.session_state:
                try:
                    st.session_state.autonomous_cognition.stop_cognitive_thread()
                    st.info("Autonomous thinking disabled")
                    logging.info("Autonomous cognition disabled - UI preserved to maintain context")
                except Exception as e:
                    st.error(f"Error stopping cognitive thread: {str(e)}")
                    logging.error(f"Failed to stop autonomous cognition: {e}")
    
    # Show current cognitive state if enabled
    if st.session_state.autonomous_cognition_enabled and 'autonomous_cognition' in st.session_state:
        current_state = st.session_state.autonomous_cognition.cognitive_state
        state_color = {
            "idle": "blue",
            "analyzing": "green",
            "reflecting": "purple",
            "learning": "orange",
            "optimizing": "teal",
            "error": "red"
        }.get(current_state, "gray")
        
        st.markdown(f"""
        <div style="margin-top:10px;">
            <b>Current cognitive state:</b> 
            <span style="color:{state_color};">{current_state}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Autonomous cognition controls
    st.markdown("---")
    show_controls = st.checkbox(
        "Show Advanced Controls", 
        value=False,
        key="autonomous_show_controls_fixed"
    )
    
    if show_controls:
        # Get current disabled activities list
        disabled_activities = get_disabled_cognitive_activities()
        
        # Define cognitive activities with display names and descriptions
        # Note: check_scheduled_reflections is excluded (handled by separate UI)
        cognitive_activities = {
            "optimize_memory_organization": {
                "name": "Optimize Memory Organization",
                "description": "Organizes and optimizes stored memories for better retrieval",
                "method": "_optimize_memory_organization"
            },
            "consolidate_similar_memories": {
                "name": "Consolidate Similar Memories",
                "description": "Finds and merges similar or duplicate memories",
                "method": "_consolidate_similar_memories"
            },
            "categorize_user_information": {
                "name": "Categorize User Information",
                "description": "Categorizes user information for better personalization",
                "method": "_categorize_user_information"
            },
            "analyze_knowledge_gaps": {
                "name": "Analyze Knowledge Gaps",
                "description": "Identifies gaps in knowledge about the user",
                "method": "_analyze_knowledge_gaps"
            },
            "fill_knowledge_gaps": {
                "name": "Fill Knowledge Gaps",
                "description": "Fills identified knowledge gaps through research",
                "method": "_fill_knowledge_gaps"
            },
            "audit_memory_confidence": {
                "name": "Audit Memory Confidence",
                "description": "Evaluates and updates memory confidence levels based on source type (Ken/Claude/web/documents) and linguistic indicators. Modifies up to 5 memories per run.",
                "method": "_audit_memory_confidence"
            }
        }
        
        st.markdown("##### Cognitive Activities")
        st.caption("Check to enable in scheduler, click ▶ to run manually")
        
        # Display each activity with checkbox and run button
        for activity_key, activity_info in cognitive_activities.items():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Checkbox to enable/disable in automatic scheduler
                is_enabled = activity_key not in disabled_activities
                new_state = st.checkbox(
                    activity_info["name"],
                    value=is_enabled,
                    help=activity_info["description"],
                    key=f"cb_activity_{activity_key}"
                )
                
                # Handle state change
                if new_state != is_enabled:
                    set_cognitive_activity_enabled(activity_key, new_state)
                    action = "enabled" if new_state else "disabled"
                    logging.info(f"User {action} cognitive activity: {activity_key}")
                    st.rerun()
            
            with col2:
                # Manual run button
                run_clicked = st.button(
                    "▶",
                    key=f"run_activity_{activity_key}",
                    help=f"Run '{activity_info['name']}' now"
                )

            # Move execution and status messages OUTSIDE the column
            if run_clicked:
                if 'autonomous_cognition' in st.session_state:
                    ac = st.session_state.autonomous_cognition
                    method_name = activity_info["method"]
                    
                    if hasattr(ac, method_name):
                        try:
                            logging.info(f"Manual trigger: Starting {activity_key}")
                            method = getattr(ac, method_name)
                            result = method()
                            
                            if result:
                                logging.info(f"Manual trigger: {activity_key} completed successfully")
                                st.success(f"✅ {activity_info['name']} completed")  # Now at full width!
                            else:
                                logging.warning(f"Manual trigger: {activity_key} completed with limited success")
                                st.warning(f"⚠️ {activity_info['name']} completed (check logs)")
                                
                        except Exception as e:
                            logging.error(f"Manual trigger: {activity_key} failed - {e}", exc_info=True)
                            st.error(f"❌ Error: {str(e)[:50]}")
                    else:
                        logging.error(f"Method {method_name} not found in AutonomousCognition")
                        st.error(f"❌ Method not available")
                else:
                    st.error("❌ Autonomous cognition not initialized")
        
        # Show last run times if available (using checkbox since we're already in an expander)
        if 'autonomous_cognition' in st.session_state:
            show_timing = st.checkbox(
                "Show Activity Timing",
                value=False,
                key="show_activity_timing"
            )
            
            if show_timing:
                st.markdown("---")
                ac = st.session_state.autonomous_cognition
                for activity_key, activity_info in cognitive_activities.items():
                    if activity_key in ac.cognitive_activities:
                        last_run = ac.cognitive_activities[activity_key].get("last_run")
                        if last_run:
                            last_run_str = datetime.datetime.fromtimestamp(last_run).strftime("%Y-%m-%d %H:%M")
                            st.caption(f"• {activity_info['name']}: {last_run_str}")
                        else:
                            st.caption(f"• {activity_info['name']}: Never run")

        
def force_ui_refresh(self):
    """Force Streamlit to refresh the UI if commands were processed - DISABLED to preserve context."""
    try:
        # DISABLED: This was causing main() reruns that broke 128K context window
        # Original issue: UI refresh was breaking conversation memory after 60+ turns
        logging.info("UI refresh requested but disabled to preserve conversation context")
        
        # Alternative: Just log what would have been refreshed
        if hasattr(st, 'session_state') and st.session_state:
            logging.info("UI state preserved - no refresh needed for memory commands")
            
    except Exception as e:
        logging.error(f"Error in disabled UI refresh: {e}")
          
def run_authenticated_app():
    """Run the main AI application after successful authentication."""
    try:
        # Set up the application
        setup_logging()
        configure_ollama_environment()  
        ensure_directories()
        st.title("Non-Biological Intelligence")
        
        if 'execution_count' not in st.session_state:
            st.session_state.execution_count = 0
        st.session_state.execution_count += 1
        logging.info(f"MAIN_EXECUTION: Run #{st.session_state.execution_count}")
        
        # Add validation and deduplication
        if st.session_state.get('execution_count', 0) > 1:
            deduplicate_messages()
            validate_conversation_state()

        # Initialize session state variables early
        if 'summaries_loaded_successfully' not in st.session_state:
            st.session_state.summaries_loaded_successfully = False
            
        if 'summaries_checked' not in st.session_state:
            st.session_state.summaries_checked = False

        if 'pending_summary_autoload' not in st.session_state:
            st.session_state.pending_summary_autoload = False

        # Initialize memory command counts early to prevent KeyError in widgets
        if 'memory_command_counts' not in st.session_state:
            st.session_state.memory_command_counts = {
                'store': 0,
                'search': 0,
                'retrieve': 0,
                'reflect': 0,
                'reflect_concept': 0,
                'forget': 0,
                'reminder': 0,
                'reminder_complete': 0,
                'summarize': 0,
                'discuss_with_claude': 0,
                'help': 0,
                'show_system_prompt': 0,
                'modify_system_prompt': 0,
                'self_dialogue': 0,
                'web_search': 0,
                'cognitive_state': 0
            }
            logging.info("Initialized memory_command_counts in session state")

        # Initialize pending scheduled reflections queue
        if 'pending_scheduled_reflections' not in st.session_state:
            st.session_state.pending_scheduled_reflections = []
           

        # Initialize image processor
        if 'image_processor' not in st.session_state:
            from image_processor import ImageProcessor
            st.session_state.image_processor = ImageProcessor()
            logging.info("Image Processor initialized in session state")

        # Initialize video processor
        if 'video_processor' not in st.session_state:
            from video_processor import VideoProcessor
            st.session_state.video_processor = VideoProcessor()
            logging.info("Video Processor initialized in session state")
        
        #  Track page loads and speech setting changes
        if 'page_load_count' not in st.session_state:
            st.session_state.page_load_count = 0
        else:
            st.session_state.page_load_count += 1
            
        # Check if this is a speech settings change
        is_speech_toggle = False
        if 'previous_speech_settings' in st.session_state:
            current_stt = st.session_state.get('speech_to_text_enabled', False)
            current_tts = st.session_state.get('text_to_speech_enabled', False)
            prev_stt = st.session_state.previous_speech_settings.get('stt', False)
            prev_tts = st.session_state.previous_speech_settings.get('tts', False)
            
            if current_stt != prev_stt or current_tts != prev_tts:
                is_speech_toggle = True
                logging.info(f"Detected speech toggle: STT {prev_stt}->{current_stt}, TTS {prev_tts}->{current_tts}")
                
        # Store current speech settings for next comparison
        if 'speech_to_text_enabled' in st.session_state or 'text_to_speech_enabled' in st.session_state:
            st.session_state.previous_speech_settings = {
                'stt': st.session_state.get('speech_to_text_enabled', False),
                'tts': st.session_state.get('text_to_speech_enabled', False)
            }
            
        # Set a flag to skip conversation reload if this is a speech toggle
        if is_speech_toggle:
            st.session_state.skip_conversation_reload = True
            logging.info("Setting skip_conversation_reload due to speech toggle")
        debug_container = st.container()
        with debug_container:
            if 'messages' in st.session_state and st.session_state.messages:
                with st.expander("Debug: Last Message", expanded=False):
                    st.write("Last assistant message:")
                    last_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
                    if last_messages:
                        st.code(last_messages[-1]["content"])
        
        # Initialize chatbot in session state if not exists
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = Chatbot()  
            st.session_state.messages = []

        # FIXED: Only auto-load summaries at true startup or after token reset
        if 'app_initialized' not in st.session_state:
            # True system startup
            st.session_state.app_initialized = True
            st.session_state.summaries_checked = False
            st.session_state.pending_summary_autoload = False
            logging.info("SYSTEM_STARTUP: Initializing application for first time")
            
            # Auto-load summaries at startup
            auto_load_most_recent_summary()
        
        
        # Add Token Counter display in sidebar
        # FIXED: Only auto-load summaries at true startup or after token reset
        if 'app_initialized' not in st.session_state:
            # True system startup
            st.session_state.app_initialized = True
            st.session_state.summaries_checked = False
            st.session_state.pending_summary_autoload = False
            logging.info("SYSTEM_STARTUP: Initializing application for first time")
            
            # Auto-load summaries at startup
            auto_load_most_recent_summary()
        
        # Add Token Counter display in sidebar
        if hasattr(st.session_state.chatbot, 'get_token_stats_readonly'):
            # ═════════════════════════════════════════════════════════════
            # CRITICAL FIX: Use read-only method for UI display
            # This prevents the UI rendering from interfering with token
            # accumulation logic. The UI should only READ the current state,
            # never MODIFY the accumulation counters.
            # ═════════════════════════════════════════════════════════════
            
            # Get current token stats WITHOUT triggering accumulation
            current_tokens, max_tokens, percentage = st.session_state.chatbot.get_token_stats_readonly()
            
            # Get cumulative directly from the chatbot
            cumulative = getattr(st.session_state.chatbot, '_cumulative_tokens_sent', 0)
            
            # Calculate overflow and fade status
            overflow = max(0, cumulative - max_tokens)
            experiencing_fade = cumulative > max_tokens
            
            # Build stats dictionary for display
            stats = {
                'estimated_active': current_tokens,
                'max_tokens': max_tokens,
                'percentage_active': percentage,
                'cumulative_sent': cumulative,
                'overflow': overflow,
                'experiencing_fade': experiencing_fade,
                'attention_sink': 2000 if experiencing_fade else 0
            }
            
        elif hasattr(st.session_state.chatbot, 'get_token_statistics'):
            # Fallback to old method if read-only method isn't available
            logging.warning("TOKEN_UI: get_token_stats_readonly not available, using get_token_statistics (may interfere with accumulation)")
            stats = st.session_state.chatbot.get_token_statistics()
            
            current_tokens = stats['estimated_active']
            max_tokens = stats['max_tokens']
            percentage = stats['percentage_active']
            cumulative = stats['cumulative_sent']
            overflow = stats['overflow']
            experiencing_fade = stats['experiencing_fade']
        else:
            # No token stats available
            logging.error("TOKEN_UI: No token statistics method available")
            current_tokens = 0
            max_tokens = 32768
            percentage = 0.0
            cumulative = 0
            overflow = 0
            experiencing_fade = False
            
            stats = {
                'estimated_active': 0,
                'max_tokens': max_tokens,
                'percentage_active': 0.0,
                'cumulative_sent': 0,
                'overflow': 0,
                'experiencing_fade': False,
                'attention_sink': 0
            }
            
        # Get search command count for detailed breakdown
        search_count = 0

        if 'memory_command_counts' in st.session_state:
            search_count = st.session_state.memory_command_counts.get('search', 0)
        
        # Create a color indicator based on token percentage
        color = "green"
        emoji = "🟢"
        status_text = "Healthy"
        if percentage > 70:
            color = "orange"
            emoji = "🟠"
            status_text = "Moderate"
        if percentage > 85:
            color = "red"
            emoji = "🔴"
            status_text = "High"
    
        st.sidebar.markdown(f"""
        ### {emoji} Context Window Usage
        <div style="margin-bottom: 15px;">
            <div style="font-size: 0.9em; color: #ffffff; margin-bottom: 5px;">
                <strong>Status:</strong> {status_text} Usage
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 20px;">
                    <div style="width: {min(100, percentage)}%; background-color: {color}; height: 100%; border-radius: 5px;"></div>
                </div>
                <div style="margin-left: 10px; font-weight: bold;">{current_tokens:,}/{max_tokens:,}</div>
            </div>
            <div style="font-size: 0.85em; color: #ffffff;">
                <strong>Estimated Active:</strong> {percentage:.1f}% ({current_tokens:,} tokens)<br>
                <strong>Session Total Sent:</strong> {cumulative:,} tokens<br>
                <strong>Search Commands:</strong> {search_count} (≈{search_count * 2000:,} token overhead)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===================================================================
        # NEW: COGNITIVE STATE DISPLAY - Right after token counter
        # ===================================================================
        st.sidebar.markdown("### 🧠 Cognitive State")

        # Call the standalone widget (not in an expander)
        with st.sidebar:
            display_cognitive_state_widget()  # No import needed, already imported at top
        # ===================================================================
            
            # Display reminders section from def_reminders in sidebar
            if 'chatbot' in st.session_state:
                with st.sidebar:
                    display_reminders_sidebar()
               
                            
        # Display selected autonomous thought if one is selected
        if 'selected_thought' in st.session_state:
            with st.expander("Autonomous Thought Details", expanded=True):
                thought = st.session_state.selected_thought
                thought_time = datetime.datetime.fromtimestamp(thought["timestamp"])
                st.markdown(f"## {thought['type'].title()} - {thought_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(thought["content"])
                
                if st.button("Close"):
                    del st.session_state.selected_thought

        
        # Initialize Autonomous Cognition in session state if not exists
        if 'autonomous_cognition' not in st.session_state and 'chatbot' in st.session_state:
            st.session_state.autonomous_cognition = AutonomousCognition(
                chatbot=st.session_state.chatbot,
                memory_db=st.session_state.chatbot.memory_db,
                vector_db=st.session_state.chatbot.vector_db
            )
            logging.info("Autonomous Cognition system initialized in session state")

                        
        # Initialize self-reflection schedules if not exists
        if 'scheduled_reflections' not in st.session_state:
            # Load saved schedule instead of using default values
            st.session_state.scheduled_reflections = load_reflection_schedule()
            logging.info("Self-reflection schedules loaded from saved configuration")


        # Initialize Claude Trainer if not exists
        if 'claude_trainer' not in st.session_state:
            api_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ClaudeAPIKey.txt")
            if os.path.exists(api_key_file):
                st.session_state.claude_trainer = ClaudeTrainer(
                    api_key_file=api_key_file,
                    memory_db=st.session_state.chatbot.memory_db,
                    vector_db=st.session_state.chatbot.vector_db,
                    llm=st.session_state.chatbot.llm,
                    chatbot=st.session_state.chatbot,
                    claude_model="claude-sonnet-4-5-20250929"  # Explicitly use latest Sonnet 4.5
                )
                logging.info("Claude Trainer initialized in session state")
            else:
                st.session_state.claude_trainer = None
                logging.warning("Claude API key file not found. Training features disabled.")

        # Start scheduler in separate thread - ONLY ONCE per session
        # Add a flag in session state to track if scheduler has been started
        if 'scheduler_started' not in st.session_state:
            st.session_state.scheduler_started = False
            start_deletion_queue_processor(st.session_state.chatbot)
            logging.info("Deletion queue processor started")    

        # Use a lock to prevent race conditions
        with scheduler_lock:
            if not st.session_state.scheduler_started:
                try:
                    scheduler_thread = threading.Thread(
                        target=schedule_learning,
                        name="DeepSeek-Scheduler",
                        daemon=True
                    )
                    if not any(t.name == "DeepSeek-Scheduler" for t in threading.enumerate()):
                        scheduler_thread.start()
                        st.session_state.scheduler_started = True
                        logging.info("Nightly learning scheduler started")
                        
                        # Start the deletion queue processor
                        start_deletion_queue_processor(st.session_state.chatbot)
                        logging.info("Deletion queue processor started")
                        
                        # Also start Claude trainer scheduler if available
                        if st.session_state.claude_trainer:
                            status = st.session_state.claude_trainer.get_scheduler_status()
                            if status["enabled"]:
                                st.session_state.claude_trainer._start_scheduler_thread()
                                logging.info("Claude training scheduler started")
                    else:
                        logging.info("Scheduler thread already running, not starting a new one")
                except Exception as e:
                    logging.error(f"Failed to start scheduler thread: {e}")

       
        # Add Counters section between Conversation Context and Image Analysis
        with st.sidebar.expander("🛠️ Counters", expanded=False):
            from utils import display_settings_widget
            display_settings_widget()
        
        with st.sidebar.expander("📷 Image Analysis", expanded=False):
            st.markdown("### Upload an image for analysis")
            uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_image:
                # Display a preview of the image
                st.image(uploaded_image, caption="Uploaded Image", width=300)
                
                # Analysis prompt
                analysis_prompt = st.text_input(
                    "Analysis prompt:",
                    value="Describe what you see in this image in detail."
                )
                
                # Process image button
                if st.button("Analyze Image"):
                    with st.spinner("Processing image..."):
                        # Save the image
                        success, file_path, image_id = st.session_state.image_processor.save_uploaded_image(uploaded_image)
                        
                        if success:
                            # Analyze the image
                            analysis_result = st.session_state.image_processor.analyze_image(
                                file_path, 
                                prompt=analysis_prompt
                            )
                            
                            if analysis_result["success"]:
                                # DON'T store yet - save to pending state
                                st.session_state.pending_image_analysis = {
                                    "analysis_result": analysis_result,
                                    "image_path": file_path,
                                    "image_id": image_id,
                                    "analysis_prompt": analysis_prompt,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                
                                # Add messages to chat
                                st.session_state.messages.append({
                                    "role": "user", 
                                    "content": f"I've uploaded an image for analysis with the prompt: {analysis_prompt}",
                                    "image_data": {
                                        "image_path": file_path,
                                        "image_id": image_id,
                                        "analysis_prompt": analysis_prompt
                                    }
                                })
                                
                                # Show analysis and ASK for additional context
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": (
                                        f"I've analyzed your image and here's what I found:\n\n"
                                        f"{analysis_result['description']}\n\n"
                                        f"Would you like to add any personal details about this image "
                                        f"(like names, dates, locations, or relationships)? "
                                        f"You can add details now, or just say 'store as-is' or 'skip' "
                                        f"if no additional context is needed."
                                    ),
                                    "image_context": {
                                        "image_path": file_path,
                                        "original_analysis": analysis_result['description'],
                                        "awaiting_user_context": True  # Flag to indicate we're waiting
                                    }
                                })
                                
                                st.success("Image analyzed! Please add any additional context in the chat, or type 'skip' to store as-is.")
                                logging.info(f"IMAGE_ANALYSIS: Waiting for user context for image {image_id}")
                                
                                # Set flag for image reference
                                st.session_state.last_message_had_image = {
                                    "image_path": file_path,
                                    "image_id": image_id,
                                    "analysis_prompt": analysis_prompt
                                }
                                
                            else:
                                st.error(f"Image analysis failed: {analysis_result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"Failed to save image: {file_path}")

        with st.sidebar.expander("🎬 Video Analysis", expanded=False):
            st.markdown("### Upload a video for analysis")
            uploaded_video = st.file_uploader(
                "Choose a video file...", 
                type=['mp4', 'avi', 'mkv', 'mov', 'flv', 'wmv'],
                help="Supported formats: MP4, AVI, MKV, MOV, FLV, WMV (max 100MB)"
            )
            
            if uploaded_video:
                # Display video info
                file_size_mb = len(uploaded_video.getvalue()) / (1024 * 1024)
                st.write(f"**File:** {uploaded_video.name}")
                st.write(f"**Size:** {file_size_mb:.2f} MB")
                
                # Analysis prompt
                analysis_prompt = st.text_input(
                    "Analysis prompt:",
                    value="Describe what you see in this video in detail.",
                    key="video_analysis_prompt"
                )
                
                # Process video button
                if st.button("Analyze Video", key="analyze_video_btn"):
                    with st.spinner("Processing video..."):
                        # Save video temporarily
                        success, file_path_or_error, video_id = st.session_state.video_processor.save_temp_video(uploaded_video)
                        
                        if success:
                            video_path = file_path_or_error
                            
                            # Analyze the video
                            analysis_result = st.session_state.video_processor.analyze_video_with_qwen(
                                video_path, 
                                analysis_prompt,
                                st.session_state.chatbot
                            )
                            
                            if analysis_result["success"]:
                                st.success("✅ Video analysis completed!")
                                
                                # Get video metadata
                                metadata = st.session_state.video_processor.get_video_metadata(video_path)

                                
                                # Add to chat history WITHOUT video reference (since we're not storing)
                                if 'messages' in st.session_state:
                                    # User message
                                    st.session_state.messages.append({
                                        "role": "user", 
                                        "content": f"I've uploaded a video '{uploaded_video.name}' for analysis with the prompt: {analysis_prompt}",
                                        "video_metadata": {
                                            "filename": uploaded_video.name,
                                            "size_mb": file_size_mb,
                                            "analysis_prompt": analysis_prompt,
                                            "video_id": video_id
                                        }
                                    })
                                    
                                    # Assistant response
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": f"I've analyzed your video '{uploaded_video.name}' and here's what I found:\n\n{analysis_result['description']}",
                                        "video_analysis": {
                                            "filename": uploaded_video.name,
                                            "original_analysis": analysis_result['description'],
                                            "model_used": "qwen3-vl:30b"
                                        }
                                    })
                                    
                                    logging.info(f"Video analysis added to chat history: {uploaded_video.name}")
                                    
                            else:
                                st.error(f"Video analysis failed: {analysis_result.get('error', 'Unknown error')}")
                            
                            # Clean up temporary file
                            st.session_state.video_processor.cleanup_temp_file(video_path)
                            
                        else:
                            st.error(f"Failed to process video: {file_path_or_error}")

        # Add Voice Settings section in sidebar
        with st.sidebar.expander("🎤 Voice Settings", expanded=False):
            st.markdown("### Speech Configuration")
            
            # Test speech components
            if st.button("Test Speech Components"):
                with st.spinner("Testing speech components..."):
                    
                    # Verify injection happened
                    if speech_handler._whisper_utils is None:  
                        st.error("❌ Speech system not initialized - whisper_utils not injected")
                        st.info("This should happen automatically in main.py startup")
                    else:
                        test_results = speech_handler.test_speech_components()
                        
                        # Display results
                        st.write("**Component Status:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"🎤 Speech-to-Text: {'✅' if test_results.get('whisper_available', False) else '❌'}")
                            st.write(f"🔊 Text-to-Speech: {'✅' if test_results.get('tts_available', False) else '❌'}")
                            st.write(f"🎵 Audio Input: {'✅' if test_results.get('audio_available', False) else '❌'}")
                        
                        with col2:
                            st.write(f"🤖 Ollama: {'✅' if test_results.get('ollama_available', False) else '❌'}")
                        
                        if test_results.get('available_models'):
                            st.write(f"**Available Models:** {', '.join(test_results['available_models'])}")
                        
                        if test_results.get('error'):
                            st.error(f"Error: {test_results['error']}")
                            
            # Speech-to-Text Toggle with persistence
            stt_enabled = st.toggle(
                "Enable Speech-to-Text",
                value=st.session_state.get('speech_to_text_enabled', False),
                help="Click the microphone button to record speech",
                key="stt_toggle_main"
            )

            # Text-to-Speech Toggle with persistence
            tts_enabled = st.toggle(
                "Enable Text-to-Speech", 
                value=st.session_state.get('text_to_speech_enabled', False),
                help="AI responses will be spoken aloud",
                key="tts_toggle_main"
            )

           
            # Save settings if any changed
            if (stt_enabled != st.session_state.get('speech_to_text_enabled', False) or 
                tts_enabled != st.session_state.get('text_to_speech_enabled', False)):
                
                st.session_state.speech_to_text_enabled = stt_enabled
                st.session_state.text_to_speech_enabled = tts_enabled
                
                # Save to file
                speech_settings = {
                    'speech_to_text_enabled': stt_enabled,
                    'text_to_speech_enabled': tts_enabled
                }
                
                save_speech_settings(speech_settings)
                logging.info(f"Speech settings saved - STT: {stt_enabled}, TTS: {tts_enabled}")


        # Initialize speech toggles from persistent settings
        if 'speech_settings_loaded' not in st.session_state:
            saved_speech_settings = load_speech_settings()
            st.session_state.speech_to_text_enabled = saved_speech_settings.get('speech_to_text_enabled', False)
            st.session_state.text_to_speech_enabled = saved_speech_settings.get('text_to_speech_enabled', False)
            st.session_state.speech_settings_loaded = True
            logging.info(f"Loaded speech settings - STT: {st.session_state.speech_to_text_enabled}, TTS: {st.session_state.text_to_speech_enabled}")
            
        # Display sidebar commands
        display_sidebar_commands()

        # Additional protection against duplicate summary indicators
        if 'messages' in st.session_state:
            # Find all "Previous conversation loaded" messages
            summary_indices = [
                i for i, msg in enumerate(st.session_state.messages)
                if msg.get("role") == "system" and "📜 Previous conversation loaded." in msg.get("content", "")
            ]
            
            # If we have more than one, keep only the first one
            if len(summary_indices) > 1:
                # Log that we're removing duplicates
                logging.warning(f"Found {len(summary_indices)} duplicate summary indicators, removing extras")
                
                # Keep only the first one (keep indices in reverse order to avoid changing indices during removal)
                for idx in sorted(summary_indices[1:], reverse=True):
                    st.session_state.messages.pop(idx)

        # Display chat history
        if 'messages' in st.session_state:
            message_container = st.container()
            with message_container:
                for message in st.session_state.messages:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    
                    # Log each message being displayed
                    logging.info(f"Displaying message - Role: {role}, Content: {content[:50]}...")
                    
                    try:
                        # Display message based on role
                        with st.chat_message(role):
                            st.markdown(content, unsafe_allow_html=True)
                    except Exception as e:
                        logging.error(f"Error displaying message: {e}")
                        # Fallback display method
                        st.text(f"{role}: {content}")

        # Create status indicators
        indicators = create_status_indicators()

        # Process speech input with Whisper (one-time button only)
        user_input = None  # Initialize the variable first

        # Define chat input placeholder first
        chat_input_placeholder = "Type your message..."
        if st.session_state.get('speech_to_text_enabled', False):
            chat_input_placeholder += " (or use 🎤 Speak button above)"

        # Simplified speech input button
        if st.session_state.get('speech_to_text_enabled', False):
            # ADD THIS CHECK:
            if not (SPEECH_UTILS_AVAILABLE and speech_handler):
                st.warning("⚠️ Speech-to-text is enabled but speech system is not available. Disabling.")
                st.session_state.speech_to_text_enabled = False
            else:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("🎤 Speak", help="Click to record speech. Speak clearly and pause when finished."):
                        status_placeholder = st.empty()
                        
                        try:
                            status_placeholder.info("🎤 Listening... Speak clearly and pause when finished.")
                            
                            # Use the simplified main method (30 seconds max for UI responsiveness)
                            recognized_text = speech_handler.speech_to_text(max_duration=25)    
                            
                            status_placeholder.empty()
                            
                            if recognized_text:
                                user_input = recognized_text
                                st.success(f"✅ You said: {recognized_text}")
                                logging.info(f"Speech recognized: {recognized_text}")
                            else:
                                st.warning("⚠️ No speech detected. Please try again.")
                                logging.info("No speech detected")
                                
                        except Exception as speech_error:
                            status_placeholder.empty()
                            st.error(f"❌ Speech error: {speech_error}")
                            logging.error(f"Speech error: {speech_error}")

        # Always show text input field
        text_input = st.chat_input(chat_input_placeholder)
        if text_input:
            user_input = text_input

        # Handle pending scheduled reflections FIRST (before other pending messages)
        if 'pending_scheduled_reflections' in st.session_state and st.session_state.pending_scheduled_reflections:

            # Process all pending reflection messages
            for pending_msg in st.session_state.pending_scheduled_reflections:
                # Add to chat history
                st.session_state.messages.append(pending_msg)
                
                # Display in chat UI
                with st.chat_message(pending_msg["role"]):
                    st.markdown(pending_msg["content"], unsafe_allow_html=True)
            
            # Clear the queue
            st.session_state.pending_scheduled_reflections = []
            
            # Log the injection
            logging.info("CHAT_FLOW: Injected pending scheduled reflection(s) into chat flow")

        # Handle pending document messages from file uploads BEFORE processing new input
        if 'pending_document_message' in st.session_state:
            # Add the pending message to chat history
            pending_msg = st.session_state.pending_document_message
            st.session_state.messages.append(pending_msg)
            
            # Display the message immediately
            with st.chat_message(pending_msg["role"]):
                st.markdown(pending_msg["content"], unsafe_allow_html=True)
            
            # Clear the pending message
            del st.session_state.pending_document_message
            
            # Log the injection
            logging.info("CHAT_FLOW: Injected pending document message into chat flow")

        # Handle pending web learning messages from web learning BEFORE processing new input
        if 'pending_web_learning_message' in st.session_state:
            # Add the pending message to chat history
            pending_msg = st.session_state.pending_web_learning_message
            st.session_state.messages.append(pending_msg)
            
            # Display the message immediately
            with st.chat_message(pending_msg["role"]):
                st.markdown(pending_msg["content"], unsafe_allow_html=True)
        
            # Clear the pending message
            del st.session_state.pending_web_learning_message
            
            # Log the injection
            logging.info("CHAT_FLOW: Injected pending web learning message into chat flow")

        # ===================================================================
        # NEW: Handle pending image analysis context from user
        # ===================================================================
        # Check if we're waiting for image context BEFORE processing new input
        if 'pending_image_analysis' in st.session_state and st.session_state.pending_image_analysis:
            # Only process if we have actual user input (not just page refresh)
            if user_input:
                pending = st.session_state.pending_image_analysis
                user_message = user_input  # The user's response with context
                
                # Check if user wants to skip adding context
                skip_keywords = ['skip', 'store as-is', 'store as is', 'no additional', 'none']
                user_wants_skip = any(keyword in user_message.lower() for keyword in skip_keywords)
                
                if user_wants_skip:
                    # Store with AI analysis only
                    logging.info("IMAGE_CONTEXT: User chose to skip additional context")
                    
                    store_success, memory_id = st.session_state.image_processor.store_enhanced_image_analysis(
                        chatbot=st.session_state.chatbot,
                        analysis_result=pending['analysis_result'],
                        user_context=""  # No additional context
                    )
                    
                    if store_success:
                        # Add confirmation message
                        confirmation_msg = {
                            "role": "assistant",
                            "content": f"✅ Image analysis stored in memory (ID: {memory_id}) without additional context."
                        }
                        st.session_state.messages.append(confirmation_msg)
                        
                        # Display confirmation
                        with st.chat_message("assistant"):
                            st.markdown(confirmation_msg["content"])
                        
                        logging.info(f"IMAGE_CONTEXT: Stored image {pending['image_id']} without user context")
                    else:
                        # Add error message
                        error_msg = {
                            "role": "assistant",
                            "content": f"⚠️ Failed to store image analysis: {memory_id}"
                        }
                        st.session_state.messages.append(error_msg)
                        
                        # Display error
                        with st.chat_message("assistant"):
                            st.markdown(error_msg["content"])
                        
                        logging.error(f"IMAGE_CONTEXT: Failed to store image {pending['image_id']}: {memory_id}")
                else:
                    # User provided additional context - store with enriched information
                    logging.info(f"IMAGE_CONTEXT: User provided context: {user_message[:100]}...")
                    
                    store_success, memory_id = st.session_state.image_processor.store_enhanced_image_analysis(
                        chatbot=st.session_state.chatbot,
                        analysis_result=pending['analysis_result'],
                        user_context=user_message  # User's additional details
                    )
                    
                    if store_success:
                        # Add confirmation message
                        confirmation_msg = {
                            "role": "assistant",
                            "content": (
                                f"✅ Perfect! I've stored the image analysis along with your additional context "
                                f"in memory (ID: {memory_id}). This enriched information will help me remember "
                                f"the important details about this image."
                            )
                        }
                        st.session_state.messages.append(confirmation_msg)
                        
                        # Display confirmation
                        with st.chat_message("assistant"):
                            st.markdown(confirmation_msg["content"])
                        
                        logging.info(f"IMAGE_CONTEXT: Stored image {pending['image_id']} WITH user context (length: {len(user_message)})")
                    else:
                        # Add error message
                        error_msg = {
                            "role": "assistant",
                            "content": f"⚠️ Failed to store enhanced image analysis: {memory_id}"
                        }
                        st.session_state.messages.append(error_msg)
                        
                        # Display error
                        with st.chat_message("assistant"):
                            st.markdown(error_msg["content"])
                        
                        logging.error(f"IMAGE_CONTEXT: Failed to store enhanced image {pending['image_id']}: {memory_id}")
                
                # Clear pending state
                del st.session_state.pending_image_analysis
                
                # IMPORTANT: Clear user_input so it's not processed as a regular chat message
                user_input = None
                
                logging.info("IMAGE_CONTEXT: Cleared pending image analysis and consumed user input")
        # ===================================================================
        # END: Handle pending image analysis context
        # ===================================================================

        # Process user input if we have any
        if user_input:
            # Validate user_input is not None and not empty
            if user_input is None or not str(user_input).strip():
                logging.warning("CHAT_FLOW: Received None or empty user input, skipping processing")
            else:
                # Diagnostic logging
                logging.critical(f"=== USER INPUT PROCESSING START ===")
                logging.critical(f"Input source: {user_input}")
                logging.critical(f"Current messages count: {len(st.session_state.messages)}")
                
                # Check if this input was already processed
                if 'last_processed_input' in st.session_state and st.session_state.last_processed_input == user_input:
                    logging.critical(f"DUPLICATE DETECTED: This input was already processed!")
                else:
                    st.session_state.last_processed_input = user_input
                    
                    # Store original input for logging
                    original_input = None
                    try:
                        # Validate user_input first
                        if user_input is None or not str(user_input).strip():
                            logging.warning("CHAT_FLOW: Received None or empty user input after validation")
                            return
                        
                        # Store original input for logging (now safe)
                        original_input = str(user_input).strip()
                        
                        # Process any user commands before adding to chat history
                        commands_processed = False
                        if 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, 'deepseek_enhancer'):
                            try:
                                # Process any user commands
                                processed_input, commands_found = st.session_state.chatbot.deepseek_enhancer.process_user_commands(original_input)
                                
                                if commands_found:
                                    commands_processed = True
                                    logging.info(f"CHAT_FLOW: Processed {commands_found} user commands")
                                    
                                    # If commands were processed and resulted in output, handle appropriately
                                    if processed_input.strip() and processed_input != original_input:
                                        # Command produced output, add it as a system message
                                        st.session_state.messages.append({"role": "system", "content": processed_input})
                                        with st.chat_message("system"):
                                            st.markdown(processed_input)
                                        
                                        # If the command completely handled the input, don't process further
                                        if not original_input.strip() or processed_input.replace(original_input, "").strip():
                                            logging.info("CHAT_FLOW: User command fully processed, skipping AI response")
                                            return
                                            
                            except Exception as cmd_error:
                                logging.error(f"CHAT_FLOW: Error processing user commands: {cmd_error}", exc_info=True)

                        # Ensure original_input is still valid before proceeding
                        if original_input is None or not original_input.strip():
                            logging.warning("CHAT_FLOW: original_input became invalid during processing")
                            return

                    except Exception as input_validation_error:
                        logging.error(f"CHAT_FLOW: Critical error in input validation: {input_validation_error}", exc_info=True)
                        st.error("Error processing your input. Please try again.")
                        return
            
                # Update the timestamp of the last user activity
                if 'autonomous_cognition' in st.session_state:
                    st.session_state.autonomous_cognition.update_user_activity()

                # Process message
                try:
                    # Check for pending summary auto-load after token reset
                    if st.session_state.get('pending_summary_autoload', False):
                        # Load the newly created summary
                        auto_load_most_recent_summary()
                        st.session_state.pending_summary_autoload = False
                        logging.info("AUTO_LOAD: Processed pending summary auto-load after token reset")
                    
                    # Add user message to chat history ONLY ONCE
                    st.session_state.messages.append({"role": "user", "content": original_input})
                    
                    # Display updated chat history immediately
                    with st.chat_message("user"):
                        st.markdown(original_input)
                    
                    logging.info(f"CHAT_FLOW: Processing user input: {original_input[:100]}...")
                    
                    # Add reminder checking when entering the chat
                    if 'reminders_checked_this_session' not in st.session_state:
                        st.session_state.reminders_checked_this_session = False
                        
                    if not st.session_state.reminders_checked_this_session and 'chatbot' in st.session_state:
                        due_reminders = st.session_state.chatbot.check_due_reminders()
                        if due_reminders:
                            # Create reminder notification
                            notification = st.session_state.chatbot.reminder_manager.format_reminder_notification(due_reminders)
                            if notification:
                                # Add as a system message in the chat
                                if 'messages' not in st.session_state:
                                    st.session_state.messages = []
                                st.session_state.messages.append({"role": "assistant", "content": notification})
                                
                                # Initialize shown reminders tracking in session state
                                if 'shown_reminders' not in st.session_state:
                                    st.session_state.shown_reminders = set()
                                for reminder in due_reminders:
                                    st.session_state.shown_reminders.add(reminder.get('id'))
                        
                        # Mark as checked this session
                        st.session_state.reminders_checked_this_session = True

                    # Get bot response
                    with st.spinner("AI is thinking..."):
                        response = st.session_state.chatbot.process_command(original_input, indicators)
                        logging.info(f"CHAT_FLOW: Generated response of length {len(response)}")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # CRITICAL FIX: Accumulate tokens IMMEDIATELY after getting response
                    # This MUST happen BEFORE the auto-summary check below to ensure
                    # the token counter reflects the tokens from this exchange
                    # ═══════════════════════════════════════════════════════════════
                    try:
                        if hasattr(st.session_state.chatbot, 'accumulate_prompt_tokens'):
                            # Call the accumulation method
                            st.session_state.chatbot.accumulate_prompt_tokens()
                            logging.info("✅ POST-RESPONSE: Token accumulation called successfully")
                            
                            # Verify the accumulation worked by reading the updated count
                            if hasattr(st.session_state.chatbot, 'get_unified_token_count'):
                                verify_tokens, verify_max, verify_pct = st.session_state.chatbot.get_unified_token_count()
                                logging.info(f"📊 POST-RESPONSE: Updated token count: {verify_tokens:,}/{verify_max:,} ({verify_pct:.2f}%)")
                        else:
                            logging.error("❌ POST-RESPONSE: accumulate_prompt_tokens method not found in chatbot!")
                    except Exception as acc_error:
                        logging.error(f"❌ POST-RESPONSE: Error during token accumulation: {acc_error}", exc_info=True)
                    # ═══════════════════════════════════════════════════════════════
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logging.info("CHAT_FLOW: Assistant response added to chat history")

                    # ========================================================================
                    # NEW: AUTOMATIC TOKEN USAGE CHECK AND SUMMARIZATION TRIGGER
                    # ========================================================================

                    try:
                        # Get current token usage from the unified token counter
                        if hasattr(st.session_state.chatbot, 'get_unified_token_count'):
                            current_tokens, max_tokens, percentage = st.session_state.chatbot.get_unified_token_count()
                            
                            # ENHANCED DIAGNOSTIC LOGGING
                            logging.critical(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                            logging.critical(f"🔍 AUTO_SUMMARY_CHECK DIAGNOSTICS")
                            logging.critical(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                            logging.critical(f"📊 Current tokens: {current_tokens:,}")
                            logging.critical(f"📊 Max tokens: {max_tokens:,}")
                            logging.critical(f"📊 Percentage: {percentage:.2f}%")
                            logging.critical(f"📊 Threshold: 85.0%")
                            logging.critical(f"📊 Will trigger? {percentage >= 85.0}")
                            
                            # Check internal state
                            if hasattr(st.session_state.chatbot, '_cumulative_tokens_sent'):
                                logging.critical(f"💾 _cumulative_tokens_sent: {st.session_state.chatbot._cumulative_tokens_sent:,}")
                            if hasattr(st.session_state.chatbot, '_last_prompt_tokens'):
                                logging.critical(f"💾 _last_prompt_tokens: {st.session_state.chatbot._last_prompt_tokens:,}")
                            
                            logging.critical(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                            
                            # Check if we've exceeded the 85% threshold
                            if percentage >= 85.0:
                                logging.critical(f"🚨 AUTO_SUMMARY_TRIGGER: Token threshold exceeded ({percentage:.1f}% >= 85%)")
                                
                                # Check if the wrapper is available
                                if (hasattr(st.session_state.chatbot, 'deepseek_enhancer') and 
                                    hasattr(st.session_state.chatbot.deepseek_enhancer, '_handle_summarize_conversation_wrapper')):
                                    
                                    # Notify user that auto-summarization is starting
                                    with st.spinner("🔄 Token limit reached - automatically summarizing conversation..."):
                                        logging.info("AUTO_SUMMARY: Executing [SUMMARIZE_CONVERSATION] command automatically")
                                        
                                        try:
                                            # Call the same wrapper that handles manual [SUMMARIZE_CONVERSATION] commands
                                            summary_response, success = st.session_state.chatbot.deepseek_enhancer._handle_summarize_conversation_wrapper()
                                            
                                            # ================================================================
                                            # FIXED: Handle success and failure separately for UI feedback,
                                            # but ALWAYS reset token counter to prevent rapid re-triggering
                                            # ================================================================
                                            
                                            if success:
                                                logging.info("AUTO_SUMMARY: Command execution successful - summary stored")
                                                
                                                # Add summary to conversation history (does NOT remove old messages)
                                                st.session_state.messages.append({"role": "assistant", "content": summary_response})
                                                
                                                # Display the summary in chat
                                                with st.chat_message("assistant"):
                                                    st.markdown(summary_response, unsafe_allow_html=True)
                                                
                                                # Show success message
                                                st.success("✅ Conversation automatically summarized and stored!")
                                                st.info("💡 Previous conversation has been summarized. You can continue chatting normally.")
                                                
                                            else:
                                                # Summary generation attempted but storage failed
                                                # This could be due to duplicate detection (content already stored)
                                                # or an actual error - either way, we've checkpointed
                                                logging.warning("AUTO_SUMMARY: Summary storage returned failure (possibly duplicate detected)")
                                                st.warning("⚠️ Summary already exists or storage failed. Token counter will be reset to prevent re-triggering.")
                                            
                                            # ================================================================
                                            # CRITICAL FIX: Reset token counter REGARDLESS of success/failure
                                            # ================================================================
                                            # Why reset even on failure?
                                            # 1. If duplicate detected: Content IS already stored - not a real failure
                                            # 2. If actual error: Prevents infinite re-trigger loop at 85%+
                                            # 3. The summary was generated - that's the logical checkpoint
                                            # 4. User can manually restart if they want a fresh session
                                            # ================================================================
                                            
                                            logging.info("AUTO_SUMMARY: Resetting token counter (prevents re-triggering)")
                                            
                                            if hasattr(st.session_state.chatbot, 'reset_token_counter_after_summary'):
                                                reset_success = st.session_state.chatbot.reset_token_counter_after_summary(keep_lifetime_stats=True)
                                                if reset_success:
                                                    logging.info("AUTO_SUMMARY: Token counter reset successfully")
                                                else:
                                                    logging.error("AUTO_SUMMARY: Token counter reset failed")
                                            else:
                                                logging.error("AUTO_SUMMARY: reset_token_counter_after_summary method not found")
                                            
                                            # Get and log new token stats after reset
                                            new_tokens, new_max, new_percentage = st.session_state.chatbot.get_unified_token_count()
                                            logging.info(f"AUTO_SUMMARY: Token usage after reset: {new_tokens:,}/{new_max:,} ({new_percentage:.1f}%)")
                                                
                                        except Exception as cmd_error:
                                            logging.error(f"AUTO_SUMMARY: Error executing wrapper: {cmd_error}", exc_info=True)
                                            st.warning("⚠️ Automatic summarization encountered an error. Continuing with current context.")
                                            
                                            # Even on exception, reset token counter to prevent infinite loop
                                            logging.info("AUTO_SUMMARY: Resetting token counter after exception (safety measure)")
                                            if hasattr(st.session_state.chatbot, 'reset_token_counter_after_summary'):
                                                st.session_state.chatbot.reset_token_counter_after_summary(keep_lifetime_stats=True)
                                
                                else:
                                    logging.error("AUTO_SUMMARY: Wrapper not available - cannot auto-summarize")
                                    st.warning("⚠️ Automatic summarization is not available. Please use [SUMMARIZE_CONVERSATION] manually.")
                            
                            else:
                                # Normal operation - no summarization needed
                                logging.debug(f"AUTO_SUMMARY_CHECK: Usage at {percentage:.1f}% - below 85% threshold")
                        
                        else:
                            logging.warning("AUTO_SUMMARY_CHECK: get_unified_token_count method not available")
                    
                    except Exception as auto_summary_error:
                        # Log error but don't disrupt the chat flow
                        logging.error(f"AUTO_SUMMARY_CHECK: Error in automatic summarization check: {auto_summary_error}", exc_info=True)
                        # Don't show error to user unless it's critical - just log it
                    
                    # ========================================================================
                    # END: AUTOMATIC TOKEN USAGE CHECK AND SUMMARIZATION TRIGGER
                    # ========================================================================

                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response, unsafe_allow_html=True)

                    # Speak the response if TTS is enabled
                    if st.session_state.text_to_speech_enabled:
                        if SPEECH_UTILS_AVAILABLE and speech_handler:
                            logging.info(f"TTS: Speaking response of {len(response)} characters")
                            
                            try:
                                def tts_call():
                                    try:
                                        speech_handler.text_to_speech(response)
                                    except Exception as e:
                                        logging.error(f"TTS thread error: {e}", exc_info=True)
                                
                                threading.Thread(target=tts_call, daemon=True).start()
                                
                            except Exception as e:
                                logging.error(f"TTS error: {e}", exc_info=True)
                                st.warning("⚠️ Could not speak response")
                        else:
                            st.warning("⚠️ Text-to-speech unavailable")
                            st.session_state.text_to_speech_enabled = False

                    # Check if any memory commands were processed and update UI
                    if hasattr(st.session_state.chatbot, 'deepseek_enhancer'):
                        # Get the previous counters to check if there were changes
                        if 'previous_counters' not in st.session_state:
                            st.session_state.previous_counters = st.session_state.memory_command_counts.copy()
                
                        # Compare current with previous counters to detect changes
                        any_changes = False
                        for key in st.session_state.memory_command_counts:
                            current = st.session_state.memory_command_counts.get(key, 0)
                            previous = st.session_state.previous_counters.get(key, 0)
                            if current > previous:
                                any_changes = True
                                break
                
                        # Update previous counters for next time
                        st.session_state.previous_counters = st.session_state.memory_command_counts.copy()
                            
                except Exception as e:
                    logging.error(f"CHAT_FLOW: Error processing message: {e}", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")

        # Command Guide Button (between System Maintenance and Admin Dashboard)
        if st.sidebar.button("📖 Command Guide", help="Opens comprehensive command reference"):
            try:
                from command_guide_generator import save_command_guide_html
                import webbrowser
                
                # Generate and save the guide
                file_path = save_command_guide_html()
                
                # Open in browser
                webbrowser.open(f'file://{os.path.abspath(file_path)}')
                
                st.sidebar.success("✅ Guide opened!")
                
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
                logging.error(f"Command guide error: {e}", exc_info=True)            
    
        # Show the admin dashboard if requested
        if st.sidebar.checkbox("Open Admin Dashboard", key="open_admin_dashboard"):
            display_admin_dashboard()
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

def main():
    """Main function with authentication gate."""
    try:
        # Set page config first (before any other Streamlit operations)
        st.set_page_config(
            page_title="Non-Biological Intelligence",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Handle authentication
        authenticated, authenticator, name, username = handle_authentication()
        
        if authenticated:
            # Show welcome message and user info in sidebar
            st.sidebar.success(f'Welcome *{name}*')
            st.sidebar.write(f'Username: {username}')
            
            # Run the main AI application
            run_authenticated_app()
        else:
            # Show login form (handled by streamlit-authenticator)
            st.info("Please log in to access the AI system")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error(f"Critical error in main(): {e}", exc_info=True)

if __name__ == "__main__":
    main()