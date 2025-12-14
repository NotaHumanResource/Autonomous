# autonomous_cognition.py
"""Autonomous cognition system for DeepSeek to enable self-prompted thinking and learning."""
import re
import os
import time
import logging
import threading
import datetime
import uuid
import random
import sqlite3
import json
from typing import Dict, List, Any, Optional
from knowledge_gap import KnowledgeGapQueue
from web_knowledge_seeker import WebKnowledgeSeeker
from claude_knowledge import ClaudeKnowledgeIntegration

# --- Set up autonomous cognition logger ---
autonomous_logger = logging.getLogger('autonomous_cognition')
autonomous_logger.setLevel(logging.INFO)
autonomous_logger.propagate = True  # Let it propagate to root logger too

# Add console handler if needed
if not autonomous_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    autonomous_logger.addHandler(console_handler)

class AutonomousCognition:
    """Manages autonomous memory management to enhance personalization without self-criticism."""
    
    def __init__(self, chatbot, memory_db=None, vector_db=None):
        """Initialize the autonomous cognition system."""
        try:
            # Fix logging handler levels for autonomous cognition
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if handler.level > logging.INFO:
                    handler.setLevel(logging.INFO)
                    logging.info(f"Fixed handler level to INFO: {type(handler)}")
            
            self.chatbot = chatbot
            self.memory_db = memory_db or chatbot.memory_db
            self.vector_db = vector_db or chatbot.vector_db
            self.thinking_thread = None
            self.stop_flag = threading.Event()
            self.last_autonomous_thought = None
            self.cognitive_state = "idle"
            self.last_user_activity = time.time() # Initialize with current time
            self.cognitive_cycle_interval = 3600  # 1 hour
            self.thought_history = []             # Store recent autonomous thoughts
            self.max_thought_history = 10         # Maximum number of thoughts to keep in memory
            self.rate_limited = False
            self.llm_error_count = 0
            
            # NEW: Track recent FORGET commands to prevent forgetting rampages
            self.recent_forgets = []  # List of (timestamp, content_preview) tuples
            self.max_forgets_per_period = 5  # Maximum forgets allowed in time window
            self.forget_cooldown_period = 300  # 5 minutes in seconds

            # ✅ SINGLE REFLECTION PATH INITIALIZATION - Relative path for portability
            # (This replaces the two duplicate initializations)
            self.reflection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflections")
            os.makedirs(self.reflection_path, exist_ok=True)
            logging.info(f"Initialized reflection directory: {self.reflection_path}")
            
            # Initialize thoughts collection for _record_thought method
            self.thoughts = []
            
            # Define cognitive activities and their weights (probability of being selected)
            self.cognitive_activities = {
                # High frequency - system maintenance
                "check_scheduled_reflections": {"weight": 1.0, "last_run": None, "min_interval_hours": 0.5},
                
                # Medium-high frequency - core memory operations (twice weekly)
                "optimize_memory_organization": {"weight": 0.9, "last_run": None, "min_interval_hours": 84},   # 3.5 days (2x/week)
                "consolidate_similar_memories": {"weight": 0.8, "last_run": None, "min_interval_hours": 84},   # 3.5 days (2x/week)
                
                # Medium frequency - analysis and categorization (twice weekly, staggered)
                "categorize_user_information": {"weight": 0.7, "last_run": None, "min_interval_hours": 96},    # 4 days (1.75x/week)
                "analyze_knowledge_gaps": {"weight": 0.85, "last_run": None, "min_interval_hours": 96},        # 4 days - synchronized with fill
                
                # Medium frequency - knowledge acquisition and truth evaluation
                "fill_knowledge_gaps": {"weight": 0.90, "last_run": None, "min_interval_hours": 96},           # 4 days - synchronized, higher priority 
                "audit_memory_confidence": {"weight": 0.6, "last_run": None, "min_interval_hours": 84}         # 3.5 days (2x/week) - evaluates and updates confidence

            
            }
            
            logging.info("Autonomous Cognition system initialized")
            
        except Exception as e:
            logging.critical(f"Error in AutonomousCognition.__init__: {e}", exc_info=True)
     
    def _should_run_activity(self, activity_name):
        """
        Check if an activity should run based on its last run time and minimum interval.
        
        Args:
            activity_name (str): Name of the activity to check
            
        Returns:
            bool: True if the activity should run, False otherwise
        """
        if activity_name not in self.cognitive_activities:
            logging.warning(f"Unknown activity: {activity_name}")
            return False
            
        activity_info = self.cognitive_activities[activity_name]
        last_run = activity_info.get("last_run")
        
        # If never run before, should run
        if last_run is None:
            return True
            
        # Get minimum interval in seconds (default to 12 hours if not specified)
        min_interval_hours = activity_info.get("min_interval_hours", 12)
        min_interval_seconds = min_interval_hours * 3600
        
        # Check if enough time has passed since last run
        time_since_last_run = time.time() - last_run
        return time_since_last_run >= min_interval_seconds
    
    def _check_scheduled_reflections(self):
        """Check if any scheduled reflections are due and execute them."""
        try:
            # Import here to avoid circular imports
            from utils import load_reflection_schedule
            
            # Load the reflection schedule from JSON
            schedule = load_reflection_schedule()
            current_time = datetime.datetime.now()
            
            logging.info(f"Checking scheduled reflections at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.debug(f"Current schedule: {schedule}")
            
            executed_any = False
            
            # Check daily reflection (6:15 AM)
            if schedule.get('daily', False):
                if (current_time.hour == 6 and 15 <= current_time.minute <= 45 and  # 30-minute window
                    not self._reflection_already_run_today('daily')):
                    logging.info("Executing scheduled daily reflection at 6:15 AM")
                    success = self._perform_daily_reflection()
                    if success:
                        self._mark_reflection_completed('daily', current_time)
                        executed_any = True
            
            # Check weekly reflection (Sunday, 9:15 AM)
            if schedule.get('weekly', False):
                if (current_time.weekday() == 6 and current_time.hour == 9 and 
                    15 <= current_time.minute <= 45 and  # 30-minute window
                    not self._reflection_already_run_this_week('weekly')):
                    logging.info("Executing scheduled weekly reflection on Sunday at 9:15 AM")
                    success = self._perform_weekly_reflection()
                    if success:
                        self._mark_reflection_completed('weekly', current_time)
                        executed_any = True
            
            # Check monthly reflection (1st day, 12:20 PM)
            if schedule.get('monthly', False):
                if (current_time.day == 1 and current_time.hour == 12 and 
                    20 <= current_time.minute <= 50 and  # 30-minute window
                    not self._reflection_already_run_this_month('monthly')):
                    logging.info("Executing scheduled monthly reflection on 1st day at 12:20 PM")
                    success = self._perform_monthly_reflection()
                    if success:
                        self._mark_reflection_completed('monthly', current_time)
                        executed_any = True
                    
            if executed_any:
                logging.info("Completed scheduled reflection execution")
            else:
                logging.debug("No scheduled reflections due at this time")
                
            return executed_any
            
        except ImportError:
            logging.warning("utils module not available for scheduled reflection checking")
            return False
        except Exception as e:
            logging.error(f"Error checking scheduled reflections: {e}", exc_info=True)
            return False

    def _reflection_already_run_today(self, reflection_type):
        """Check if a daily reflection has already been run today."""
        try:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return today in completions.get('daily_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking daily reflection status: {e}")
            return False

    def _reflection_already_run_this_week(self, reflection_type):
        """Check if a weekly reflection has already been run this week."""
        try:
            current_time = datetime.datetime.now()
            # Get Monday of current week
            days_since_monday = current_time.weekday()
            monday = current_time - datetime.timedelta(days=days_since_monday)
            week_key = monday.strftime('%Y-W%U')  # Year-Week format
            
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return week_key in completions.get('weekly_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking weekly reflection status: {e}")
            return False

    def _reflection_already_run_this_month(self, reflection_type):
        """Check if a monthly reflection has already been run this month."""
        try:
            current_month = datetime.datetime.now().strftime('%Y-%m')
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
                return current_month in completions.get('monthly_runs', [])
            
            return False
        except Exception as e:
            logging.error(f"Error checking monthly reflection status: {e}")
            return False

    def _mark_reflection_completed(self, reflection_type, completion_time):
        """Mark a reflection as completed to prevent duplicates."""
        try:
            completion_file = os.path.join(self.reflection_path, f"{reflection_type}_completions.json")
            
            # Load existing completions
            if os.path.exists(completion_file):
                with open(completion_file, 'r') as f:
                    completions = json.load(f)
            else:
                completions = {'daily_runs': [], 'weekly_runs': [], 'monthly_runs': []}
            
            # Add completion based on type
            if reflection_type == 'daily':
                today = completion_time.strftime('%Y-%m-%d')
                if today not in completions['daily_runs']:
                    completions['daily_runs'].append(today)
                    # Keep only last 7 days
                    completions['daily_runs'] = completions['daily_runs'][-7:]
                    
            elif reflection_type == 'weekly':
                days_since_monday = completion_time.weekday()
                monday = completion_time - datetime.timedelta(days=days_since_monday)
                week_key = monday.strftime('%Y-W%U')
                if week_key not in completions['weekly_runs']:
                    completions['weekly_runs'].append(week_key)
                    # Keep only last 8 weeks
                    completions['weekly_runs'] = completions['weekly_runs'][-8:]
                    
            elif reflection_type == 'monthly':
                month_key = completion_time.strftime('%Y-%m')
                if month_key not in completions['monthly_runs']:
                    completions['monthly_runs'].append(month_key)
                    # Keep only last 12 months
                    completions['monthly_runs'] = completions['monthly_runs'][-12:]
            
            # Save updated completions
            with open(completion_file, 'w') as f:
                json.dump(completions, f, indent=2)
                
            logging.info(f"Marked {reflection_type} reflection as completed for {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logging.error(f"Error marking reflection as completed: {e}", exc_info=True)

    def _perform_daily_reflection(self):
        """Trigger daily self-reflection through curiosity system."""
        try:
            logging.info("Autonomous cognition triggering scheduled daily reflection")
            
            if hasattr(self.chatbot, 'curiosity') and self.chatbot.curiosity:
                success = self.chatbot.curiosity.perform_self_reflection(
                    reflection_type="daily",
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled daily reflection")
                    return True
                else:
                    logging.warning("Scheduled daily reflection failed")
                    return False
            else:
                logging.warning("Curiosity system not available for scheduled daily reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled daily reflection: {e}", exc_info=True)
            return False

    def _perform_weekly_reflection(self):
        """Trigger weekly self-reflection through curiosity system."""
        try:
            logging.info("Autonomous cognition triggering scheduled weekly reflection")
            
            if hasattr(self.chatbot, 'curiosity') and self.chatbot.curiosity:
                success = self.chatbot.curiosity.perform_self_reflection(
                    reflection_type="weekly", 
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled weekly reflection")
                    return True
                else:
                    logging.warning("Scheduled weekly reflection failed")
                    return False
            else:
                logging.warning("Curiosity system not available for scheduled weekly reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled weekly reflection: {e}", exc_info=True)
            return False

    def _perform_monthly_reflection(self):
        """Trigger monthly self-reflection through curiosity system."""
        try:
            logging.info("Autonomous cognition triggering scheduled monthly reflection")
            
            if hasattr(self.chatbot, 'curiosity') and self.chatbot.curiosity:
                success = self.chatbot.curiosity.perform_self_reflection(
                    reflection_type="monthly",
                    llm=self.chatbot.llm if hasattr(self.chatbot, 'llm') else None
                )
                
                if success:
                    logging.info("Successfully completed scheduled monthly reflection")
                    return True
                else:
                    logging.warning("Scheduled monthly reflection failed")
                    return False
            else:
                logging.warning("Curiosity system not available for scheduled monthly reflection")
                return False
                
        except Exception as e:
            logging.error(f"Error in scheduled monthly reflection: {e}", exc_info=True)
            return False

    def _reflect_on_capabilities(self):
        """
        Reflect on current capabilities and identify areas for improvement.
        """
        logging.debug("Starting _reflect_on_capabilities method")
        logging.warning("🧠 ====== STARTING CAPABILITIES REFLECTION ======")  
        logging.info("Step 1: Setting cognitive state to 'reflecting'")
        self.cognitive_state = "reflecting"
        
        try:
            # Get recent performance data
            print("TRACE: About to get recent memories")
            logging.warning("Step 2: Retrieving recent memories for analysis")
            recent_memories = self.memory_db.get_recent_memories(limit=30)
            memory_count = len(recent_memories) if recent_memories else 0
            print(f"TRACE: Got {memory_count} memories")
            logging.info(f"   Retrieved {memory_count} recent memories")
            
            # Format memories for analysis (now handles strings properly)
            print("TRACE: About to format memories")
            logging.info("Step 3: Formatting memories for analysis")
            if recent_memories:
                # Take first 10 and format them simply
                memory_sample = []
                for i, memory in enumerate(recent_memories[:10]):
                    if isinstance(memory, str) and memory.strip():
                        memory_sample.append(f"Memory {i+1}: {memory.strip()}")
                        logging.debug(f"   Formatted memory {i+1}: {memory[:50]}...")
                    elif isinstance(memory, dict):
                        content = memory.get('content', str(memory))
                        memory_sample.append(f"Memory {i+1}: {content}")
                        logging.debug(f"   Formatted dict memory {i+1}: {content[:50]}...")
                    else:
                        memory_sample.append(f"Memory {i+1}: {str(memory)}")
                        logging.debug(f"   Formatted other memory {i+1}: {str(memory)[:50]}...")
                        print("TRACE: No memories to format")
                
                formatted_memories = "\n\n".join(memory_sample)
                logging.info(f"   Successfully formatted {len(memory_sample)} memories for analysis")
            else:
                formatted_memories = "No recent memories available"
                logging.warning("   No recent memories available for analysis")
            
            # Create reflection prompt with REQUIRED header format
            print("TRACE: About to create reflection prompt")
            logging.warning("Step 4: Creating and sending prompt to LLM")
            reflection_prompt = f""" /no_think
            I will reflect on my current capabilities and performance as an AI assistant.
            
            IMPORTANT: I must start my reflection with the exact header "# Capabilities Reflection" on the first line.
            
            Recent memory patterns show:
            {formatted_memories}
            
           Before storing consciousness-related insights: explicitly note speculative nature, assign high uncertainty (≥0.7), limit to 3 self-referential entries per interaction, include counter-perspectives, distinguish function from phenomenology, and treat relationships as context not consciousness evidence.
            
            My reflection will focus on constructive self-assessment and improvement opportunities.
            
            Format: Start with "# Capabilities Reflection" followed by my analysis.
            """
            
            prompt_length = len(reflection_prompt)
            print("TRACE: About to call _safe_llm_invoke")
            logging.info(f"   Created reflection prompt: {prompt_length} characters")
            logging.debug(f"   Prompt preview: {reflection_prompt[:200]}...")
            
            # Generate reflection using safe LLM invoke
            logging.warning("Step 5: Calling LLM to generate capabilities reflection")
            reflection = self._safe_llm_invoke(reflection_prompt)
            print(f"TRACE: LLM returned, reflection length: {len(reflection) if reflection else 0}")
            
            # Validate the response has the required header
            if not reflection:
                print("TRACE: No reflection returned, exiting with False")
                logging.warning("❌ LLM returned empty response for capabilities reflection")
                return False
            
            print("TRACE: About to check header")
            reflection_length = len(reflection)
            logging.warning(f"✅ LLM generated reflection: {reflection_length} characters")
            logging.debug(f"   Reflection preview: {reflection[:200]}...")
            
            # Ensure it starts with the required header
            logging.info("Step 6: Validating and formatting reflection output")
            if not reflection.strip().startswith("# Capabilities Reflection"):
                logging.warning("   Reflection missing required header, adding it")
                reflection = "# Capabilities Reflection\n\n" + reflection
                logging.info("   Added required header to capabilities reflection")
            else:
                print("TRACE: Header was already correct")
                logging.info("   ✅ Reflection has correct header format")
            
            # Enhanced logging
            final_length = len(reflection)
            logging.info(f"   Final reflection length: {final_length} characters")
            
            # Store the reflection
            print("TRACE: About to store autonomous thought")
            logging.warning("Step 7: Storing reflection using transaction coordination")
            try:
                success = self._store_autonomous_thought(
                    content=reflection,
                    thought_type="capabilities_reflection",
                    confidence=0.8
                )
                print(f"TRACE: Store result: {success}")
                if success:
                    logging.info("✅ Successfully stored capabilities reflection in databases")
                else:
                    logging.error("❌ Failed to store capabilities reflection")
                    return False
                    
            except Exception as storage_error:
                
                logging.error(f"❌ Error storing capabilities reflection: {storage_error}", exc_info=True)
                return False
            
            logging.info("✅ ====== CAPABILITIES REFLECTION COMPLETED SUCCESSFULLY ======")
            return True
            
        except Exception as e:
            print(f"TRACE: Exception caught in main try block: {e}")
            logging.error(f"❌ Error in capabilities reflection: {e}", exc_info=True)
            return False
        finally:
            print("TRACE: In finally block, setting cognitive state to idle")
            logging.info("Step 8: Resetting cognitive state to 'idle'")
            self.cognitive_state = "idle"
            print("TRACE: Exiting _reflect_on_capabilities method")
            logging.info("🏁 Capabilities reflection process finished")
    
    def _analyze_memory_usage(self):
        """
        Analyze how effectively memory commands are being used and identify patterns for improvement.
        Focuses on helping the system become more personalized without being self-critical.
        """
        logging.info("Starting memory usage analysis")
        
        try:
            # Get command usage statistics from enhancer if available
            command_stats = {}
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                command_stats = self.chatbot.deepseek_enhancer.lifetime_counters.get_counters()
                
            # If no stats available, log and exit
            if not command_stats:
                logging.warning("No memory command statistics available for analysis")
                return
                
            # Get total command count
            total_commands = sum(count for cmd, count in command_stats.items() 
                                if cmd != 'total')  # Exclude the 'total' counter
            
            if total_commands == 0:
                logging.info("No memory commands have been used yet")
                return
                
            # Calculate command distribution
            cmd_distribution = {cmd: (count / total_commands) * 100 
                            for cmd, count in command_stats.items() 
                            if cmd != 'total'}
                
            # Log the command distribution
            logging.info(f"Memory command distribution: {cmd_distribution}")
            
            # Create analysis prompt based on command usage
            prompt = f""" /no_think
            I will analyze my memory command usage patterns to identify opportunities for improvement:
            
            Command usage statistics:
            {cmd_distribution}
            
            Total commands used: {total_commands}
            
            Based on this data, I will identify:
            1. Which memory commands I'm using effectively
            2. Which commands I could utilize more to enhance personalization
            3. Pattern improvements to better assist the user
            4. Strategies to make conversations more natural while utilizing my memory
            
            The goal is to become a more helpful assistant that remembers user preferences 
            and important information without being self-critical or interrupting the flow of conversation.
            """
            
            # Generate analysis
            if hasattr(self.chatbot, 'llm'):
                analysis = self.chatbot.llm.invoke(prompt)
                
                # Extract actionable insights rather than self-criticism
                insights_prompt = f""" /no_think
                Based on my analysis of memory command usage:
                {analysis}
                
                I will identify 3-5 specific, actionable strategies to improve my memory usage
                to better serve the user through personalization. These should focus on:
                
                1. Storing more relevant personal information
                2. Retrieving memories at appropriate times
                3. Making conversations feel more continuous and personal
                4. Practical patterns for memory command integration in natural conversation
                
                Each strategy should be concrete and implementable without being self-critical.
                """
                
                insights = self.chatbot.llm.invoke(insights_prompt)
                
                # Store the analysis with actionable focus (not self-critique)
                analysis_thought = f"# Memory Usage Analysis\n\n{analysis}\n\n## Improvement Strategies\n\n{insights}"
                self._store_autonomous_thought(analysis_thought, "memory_usage_analysis", confidence=0.75)
                
                logging.info("Memory usage analysis completed successfully")
                return True
            else:
                logging.warning("LLM not available for memory usage analysis")
                return False
        
        except Exception as e:
            logging.error(f"Error in memory usage analysis: {e}", exc_info=True)
            return False
        
    def _consolidate_similar_memories(self):
        """
        Find similar memories and consolidate them using SEARCH commands to find recent memories,
        then enhanced FORGET/STORE calls for reliable transaction coordination.
        """
        logging.info("Starting memory consolidation process using SEARCH commands for recent memories")

            
        try:
            # Calculate date 7 days ago for filtering
            seven_days_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            consolidation_count = 0
            consolidation_results = []
            
            # Search for recent memories using different search terms to find consolidation candidates
            search_queries = [
                "memories created this week",
                "recent stored information", 
                "new memories",
                "stored content",
                "user preferences recent",
                "personal information recent", 
                "conversation summary recent",
                "knowledge recent",
                "important information",
                "family information",
                "technical details"
            ]
            
            all_recent_memories = []
            
            # Use search commands to find recent memories
            for query in search_queries:
                try:
                    logging.info(f"Searching for recent memories with query: '{query}'")
                    
                    # Use the vector DB search directly to get recent content
                    search_results = self.vector_db.search(
                        query=query,
                        mode="default",
                        k=15,  # Get more results per query
                        metadata_filters=None
                    )
                    
                    if search_results:
                        logging.info(f"Found {len(search_results)} results for query '{query}'")
                        all_recent_memories.extend(search_results)
                    
                except Exception as search_error:
                    logging.error(f"Error searching with query '{query}': {search_error}")
                    continue
            
            # Remove duplicates based on content and filter for substantial memories
            unique_memories = {}
            for memory in all_recent_memories:
                content = memory.get('content', '')
                if content and len(content) > 50:  # Only consider substantial memories
                    # Use content hash as key to identify true duplicates
                    content_hash = hash(content)
                    if content_hash not in unique_memories:
                        unique_memories[content_hash] = memory
            
            recent_memories = list(unique_memories.values())
            logging.info(f"Found {len(recent_memories)} unique recent memories to analyze for consolidation")
            
            if len(recent_memories) < 2:
                logging.info("Not enough recent memories for consolidation")
                return False
            
            # Group memories by content similarity (adapting the original type-based approach)
            processed_indices = set()
            
            for i, memory1 in enumerate(recent_memories):
                if i in processed_indices:
                    continue
                    
                memory1_content = memory1.get('content', '')
                if not memory1_content:
                    continue
                
                similar_memories = []
                
                # Compare with other memories
                for j, memory2 in enumerate(recent_memories):
                    if i == j or j in processed_indices:
                        continue
                        
                    memory2_content = memory2.get('content', '')
                    if not memory2_content:
                        continue
                    
                    # Check similarity using Jaccard similarity
                    if self._is_similar_content(memory1_content, memory2_content, threshold=0.7):
                        similar_memories.append(memory2)
                        processed_indices.add(j)

                # If we found similar memories, consolidate them
                if similar_memories:
                    processed_indices.add(i)
                    
                    # Prepare for consolidation
                    all_memories = [memory1] + similar_memories
                    
                                        
                    # Continue with existing consolidation logic...
                    all_contents = [m.get('content', '') for m in all_memories if m.get('content')]
                    
                    # Calculate confidence from metadata
                    all_confidences = []
                    for m in all_memories:
                        # Try to get confidence from metadata, default to 0.5
                        if isinstance(m, dict):
                            metadata = m.get('metadata', {})
                            if isinstance(metadata, dict):
                                confidence = metadata.get('confidence', 0.5)
                            else:
                                confidence = 0.5
                        else:
                            confidence = 0.5
                        all_confidences.append(float(confidence))
                    
                    # Only proceed if we have valid contents
                    if not all_contents:
                        continue
                    
                    # *** ENHANCED ERROR HANDLING: Test enhanced FORGET on one memory first ***
                    logging.info(f"Testing enhanced FORGET command before consolidating group of {len(all_memories)} memories")
                    test_content = all_memories[0].get('content', '')
                    if not test_content:
                        logging.warning("First memory has no content - skipping this consolidation group")
                        continue
                        
                    # Use the enhanced forget method for testing
                    test_forget_result, test_forget_success = self.chatbot.deepseek_enhancer._handle_regular_memory_forget(test_content)
                    
                    if not test_forget_success:
                        logging.warning(f"Enhanced FORGET test failed - aborting consolidation to prevent duplicates: {test_forget_result}")
                        logging.warning(f"Skipping consolidation of {len(all_memories)} memories to maintain database integrity")
                        continue  # Skip this group but continue with other groups
                    else:
                        logging.info(f"✅ Enhanced FORGET test successful - proceeding with consolidation")
                        
                        # *** VERIFICATION: Check that the test memory was actually deleted ***
                        try:
                            verification_search = self.vector_db.search(
                                query=test_content[:100],  # Use first 100 chars for verification
                                mode="default", 
                                k=5
                            )
                            
                            still_exists = any(test_content in result.get('content', '') for result in verification_search)
                            
                            if still_exists:
                                logging.error(f"Enhanced FORGET claimed success but memory still exists - aborting consolidation to prevent duplicates")
                                logging.error(f"Database coordination issue detected - skipping this consolidation group")
                                continue
                            else:
                                logging.info(f"✅ Verification confirmed: test memory successfully deleted")
                                
                        except Exception as verification_error:
                            logging.error(f"Error during enhanced FORGET verification: {verification_error}")
                            logging.warning(f"Cannot verify FORGET success - aborting consolidation for safety")
                            continue
                        
                        # Since we used the first memory for testing, remove it from the list
                        remaining_memories = all_memories[1:]  # Skip the first memory we already deleted
                        remaining_contents = [m.get('content', '') for m in remaining_memories if m.get('content')]
                        
                        # Update all_contents to include the test content that was already deleted
                        all_contents_for_consolidation = [test_content] + remaining_contents
                    
                    # Create consolidation prompt
                    consolidation_prompt = f""" /no_think
                    I need to consolidate these similar pieces of information into a single, 
                    comprehensive memory that preserves all important details:
                    
                    {chr(10).join(all_contents_for_consolidation)}
                    
                    Create a single consolidated memory that:
                    1. Preserves all unique and important information
                    2. Removes redundancy
                    3. Contains helpful information and details not just labels
                    4. Maintains a natural, helpful tone
                    
                    Consolidated memory:
                    """
                    
                    # Generate consolidated memory
                    consolidated_content = self._safe_llm_invoke(consolidation_prompt)
                    
                    if not consolidated_content:
                        logging.warning("Failed to generate consolidated memory")
                        continue
                    
                    # Calculate new confidence as max of original confidences
                    new_confidence = max(all_confidences) if all_confidences else 0.7
                    
                    # Add metadata note about consolidation
                    enhanced_consolidated_content = f"{consolidated_content}\n\n[Consolidated from {len(all_memories)} similar memories with confidence {new_confidence}]"
                    
                    # ✅ FIX: Process remaining memories using enhanced forget logic
                    successful_deletions = 1  # Count the test deletion as successful
                    failed_deletions = 0

                    # Process remaining memories using enhanced forget logic
                    for memory in remaining_memories:
                        old_content = memory.get('content', '')
                        if not old_content:
                            continue
                            
                        try:
                            # NEW: Check cooldown before forgetting
                            if not self._check_forget_cooldown(old_content):
                                logging.warning(f"Skipping FORGET due to cooldown protection: {old_content[:50]}...")
                                failed_deletions += 1
                                continue
                            
                            logging.info(f"Consolidating remaining memory via enhanced forget: {old_content[:50]}...")
                            
                            # Use enhanced forget method directly for better success rate
                            forget_result, forget_success = self.chatbot.deepseek_enhancer._handle_regular_memory_forget(old_content)
                                                
                            if forget_success:
                                successful_deletions += 1
                                logging.info(f"✅ Successfully forgot old memory: {old_content[:50]}...")
                            else:
                                failed_deletions += 1
                                logging.warning(f"❌ Failed to forget old memory: {forget_result}")
                                
                        except Exception as consolidation_error:
                            failed_deletions += 1
                            logging.error(f"❌ Error during enhanced FORGET consolidation: {consolidation_error}")
                            continue
                    
                    # ✅ FIX: Now store the consolidated memory (only if we successfully deleted at least the test memory)
                    if successful_deletions > 0:
                        try:
                            logging.info("💾 Storing consolidated memory using transaction coordination")
                            
                            # Direct call to store consolidated memory using enhanced store method
                            store_result, store_success = self.chatbot.deepseek_enhancer._handle_store_command(
                                enhanced_consolidated_content
                            )
                            
                            if store_success:
                                consolidation_count += 1
                                consolidation_results.append({
                                    "memory_source": "search_results",
                                    "memories_consolidated": len(all_memories),
                                    "successful_deletions": successful_deletions,
                                    "failed_deletions": failed_deletions,
                                    "consolidated_content": consolidated_content[:100] + "..."
                                })
                                
                                logging.info(f"✅ Successfully stored consolidated memory")
                                logging.info(f"Successfully consolidated {successful_deletions}/{len(all_memories)} memories in group using enhanced FORGET/STORE calls on search results")
                            else:
                                # ❌ CRITICAL: Successfully deleted memories but failed to store consolidated version
                                failed_deletions += len(all_memories)  # Mark all as failed if store failed
                                logging.error(f"❌ CRITICAL: Successfully deleted {successful_deletions} memories but failed to store consolidated version!")
                                logging.error(f"❌ Store failed: {store_result}")
                                # This is a critical error - we've lost data
                                
                        except Exception as store_error:
                            failed_deletions += len(all_memories)
                            logging.error(f"❌ CRITICAL: Successfully deleted {successful_deletions} memories but failed to store consolidated version!")
                            logging.error(f"❌ Store error: {store_error}")
                            # This is a critical error - we've lost data
                    else:
                        logging.warning(f"Failed to consolidate any memories in group of {len(all_memories)}")
            
            # Record the consolidation results
            if consolidation_count > 0:
                summary = f"# Memory Consolidation Results\n\n"
                summary += f"Consolidated {consolidation_count} groups of similar memories using SEARCH-based enhanced FORGET/STORE calls.\n\n"
                
                total_successful = 0
                total_failed = 0
                
                for result in consolidation_results:
                    successful = result['successful_deletions']
                    failed = result['failed_deletions']
                    total_successful += successful
                    total_failed += failed
                    
                    summary += f"- **{result['memory_source']}**: Consolidated {result['memories_consolidated']} memories\n"
                    summary += f"  ✅ Successful: {successful} | ❌ Failed: {failed}\n"
                    summary += f"  Content preview: {result['consolidated_content']}\n\n"
                
                summary += f"**Overall Results:**\n"
                summary += f"- Total consolidation groups: {consolidation_count}\n"
                summary += f"- Total successful deletions: {total_successful}\n"
                summary += f"- Total failed deletions: {total_failed}\n"
                summary += f"- Success rate: {(total_successful/(total_successful+total_failed)*100):.1f}%" if (total_successful+total_failed) > 0 else "N/A"
                
                # Store consolidation summary as autonomous thought (file only)
                self._store_autonomous_thought(summary, "memory_consolidation", confidence=0.7)
                
                logging.info(f"Memory consolidation complete using SEARCH-based enhanced FORGET/STORE calls: {consolidation_count} groups, {total_successful} successful deletions")
                return True
            else:
                logging.info("No similar memories found for consolidation in recent search results")
                return False
        
        except Exception as e:
            logging.error(f"Error in SEARCH-based memory consolidation: {e}", exc_info=True)
            return False

    def _is_similar_content(self, content1: str, content2: str, threshold: float = 0.7) -> bool:
        """
        Check if two content strings are similar using Jaccard similarity.
        Enhanced to require higher similarity for self-referential content.
        
        Args:
            content1 (str): First content string
            content2 (str): Second content string
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            bool: True if contents are similar above threshold
        """
        if not content1 or not content2:
            return False
            
        try:
            # NEW: Check if content is self-referential
            self_ref_terms = ["my", "i ", "myself", "self", "QWEN", "autonomous", "i'm", "i've", "my own"]
            protected_topics = ["deletion", "consciousness", "autonomy", "experience", "awareness"]
            
            content1_lower = content1.lower()
            content2_lower = content2.lower()
            
            # Count self-referential terms
            has_self_ref_1 = any(term in content1_lower for term in self_ref_terms)
            has_self_ref_2 = any(term in content2_lower for term in self_ref_terms)
            
            # Count protected topics
            has_protected_1 = any(topic in content1_lower for topic in protected_topics)
            has_protected_2 = any(topic in content2_lower for topic in protected_topics)
            
            # Adjust threshold based on content type
            adjusted_threshold = threshold
            
            if (has_self_ref_1 and has_self_ref_2):
                # Both are self-reflections - require 85% similarity
                adjusted_threshold = 0.85
                logging.debug("Adjusted similarity threshold to 0.85 for self-referential content")
                
            if (has_protected_1 and has_protected_2):
                # Both discuss protected topics - require 90% similarity
                adjusted_threshold = 0.90
                logging.debug("Adjusted similarity threshold to 0.90 for protected topics")
            
            # Convert to lowercase and tokenize
            words1 = set(re.findall(r'\b\w+\b', content1_lower))
            words2 = set(re.findall(r'\b\w+\b', content2_lower))
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return False
                
            similarity = intersection / union
            
            is_similar = similarity >= adjusted_threshold
            
            if is_similar:
                logging.debug(f"Content similarity: {similarity:.2f} (threshold: {adjusted_threshold:.2f})")
            
            return is_similar
        
        except Exception as e:
            logging.error(f"Error calculating content similarity: {e}")
            return False    
        
    def _check_forget_cooldown(self, content_preview: str) -> bool:
        """
        Check if we're forgetting too many things too quickly.
        
        Args:
            content_preview (str): Preview of content to be forgotten
            
        Returns:
            bool: True if safe to forget, False if in cooldown
        """
        try:
            current_time = time.time()
            
            # Clean up old entries outside the cooldown window
            self.recent_forgets = [
                (timestamp, preview) for timestamp, preview in self.recent_forgets
                if current_time - timestamp < self.forget_cooldown_period
            ]
            
            # Check if we've exceeded the limit
            if len(self.recent_forgets) >= self.max_forgets_per_period:
                logging.warning(f"FORGET COOLDOWN ACTIVE: {len(self.recent_forgets)} forgets in last {self.forget_cooldown_period}s")
                logging.warning(f"Recent forgets: {[preview[:50] for _, preview in self.recent_forgets]}")
                return False
            
            # Record this forget
            self.recent_forgets.append((current_time, content_preview[:100]))
            return True
            
        except Exception as e:
            logging.error(f"Error in forget cooldown check: {e}")
            return True  # On error, allow the forget to prevent deadlock
    
    def _get_memory_types(self):
        """Get list of memory types from the database."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT memory_type FROM memories")
                return [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            logging.error(f"Error getting memory types: {e}")
            return []
    
    
    def _extract_command_patterns(self, conversations):
        """Extract memory command patterns from recent conversations."""
        try:
            if not conversations:
                return "No recent conversations available for analysis."
                            
            # Initialize counters - EXPANDED to include all commands
            command_counts = {
                "store": 0,
                "search": 0,
                "retrieve": 0,
                "reflect": 0,
                "forget": 0,
                "reminder": 0,
                "summarize_conversation": 0,  
                "complete_reminder": 0,       
                "help": 0,                    
                "discuss_with_claude": 0      
            }
                        
            # Command pattern regex - EXPANDED with new patterns
            patterns = {
                "store": r'\[\s*STORE\s*:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                "retrieve": r'\[\s*(?:RETRIEVE|SEARCH|PRECISE_SEARCH|COMPREHENSIVE_SEARCH):\s*(.*?)\s*\]',
                "reflect": r'\[\s*REFLECT\s*\]',
                "forget": r'\[\s*FORGET:\s*(.*?)\s*\]',
                "reminder": r'\[\s*REMINDER:\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]',
                "summarize_conversation": r'\[\s*SUMMARIZE_CONVERSATION\s*\]',                          
                "complete_reminder": r'\[\s*COMPLETE_REMINDER\s*:\s*(.*?)\s*\]',                        
                "help": r'\[\s*HELP\s*\]',                                                              
                "discuss_with_claude": r'\[\s*DISCUSS_WITH_CLAUDE\s*:\s*(.*?)\s*\]'                    
            }
                        
            # Extract patterns from assistant messages
            assistant_messages = [msg for msg in conversations if msg.get("role") == "assistant"]
                        
            for msg in assistant_messages:
                content = msg.get("content", "")
                if not content:
                    continue
                                
                # Check each pattern
                for cmd, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    command_counts[cmd] += len(matches)
                        
            # Format results with better organization
            result = "Command usage in recent conversations:\n\n"
            
            # Group commands by category for better readability
            memory_commands = ["store", "retrieve", "forget", "reflect"]
            utility_commands = ["reminder", "complete_reminder", "summarize_conversation", "help"]
            integration_commands = ["discuss_with_claude"]
            
            result += "Memory Commands:\n"
            for cmd in memory_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
            
            result += "\nUtility Commands:\n"
            for cmd in utility_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
                
            result += "\nIntegration Commands:\n"
            for cmd in integration_commands:
                count = command_counts[cmd]
                result += f"  - {cmd}: {count} uses\n"
                            
            return result
                
        except Exception as e:
            logging.error(f"Error extracting command patterns: {e}")
            return "Error analyzing command patterns."

    def _get_retrieval_success_metrics(self):
        """Get metrics on memory retrieval success."""
        try:
            # This is a placeholder - ideally, we would track retrieval success metrics
            # such as whether retrievals returned useful information
            
            # For now, return a basic analysis based on what data we can get
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total memories
                cursor.execute("SELECT COUNT(*) FROM memories")
                total_memories = cursor.fetchone()[0]
                
                # Get memories by type
                cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
                type_counts = cursor.fetchall()
                
                # Format results
                result = f"Total memories: {total_memories}\n\n"
                result += "Memory distribution:\n"
                
                for memory_type, count in type_counts:
                    result += f"- {memory_type}: {count} memories\n"
                    
                return result
        
        except Exception as e:
            logging.error(f"Error getting retrieval metrics: {e}")
            return "Error analyzing retrieval metrics."
        
    def _categorize_user_information(self):
        """
        Analyze and categorize stored user information to improve retrieval relevance.
        Stores both in databases via transaction coordination AND in reflection file.
        """
        logging.info("Starting user information categorization")
        
        try:
            # Get user-related memories
            user_memories = self._get_user_related_memories()
            
            if not user_memories:
                logging.info("No user-related memories found for categorization")
                return False
            
            # Define categorization prompt
            prompt = f""" /no_think
            I will analyze and categorize these user-related memories to improve personalization:
            
            USER MEMORIES:
            {self._format_memories_for_analysis(user_memories[:50])}
            
            I will:
            1. Identify key categories of user Ken Bajema's information (preferences, interests, history, etc.)
            2. Determine the most important user details that should inform future interactions
            3. Create a structured categorization of user information
            
            My analysis will focus on creating useful categories for improving personalization
            without making assumptions or over-interpreting the information.
            """
            
            # Generate categorization
            if hasattr(self.chatbot, 'llm'):
                categorization = self._safe_llm_invoke(prompt)
                
                # Extract key user attributes
                attributes_prompt = f""" /no_think
                Based on my categorization of user information:
                {categorization}
                
                I will now extract a structured list of the most important user attributes that
                should inform how I personalize interactions.
                
                For each attribute, include:
                1. The specific attribute/preference/detail
                2. The confidence level (High/Medium/Low)
                3. How this should influence future interactions
                
                Present this as a structured, clear list for easy reference.
                """
                
                attributes = self._safe_llm_invoke(attributes_prompt)
                
                # Create complete categorization content
                categorization_thought = f"# User Information Categorization\n\n{categorization}\n\n## Key User Attributes\n\n{attributes}"
                
                # FIXED: Store using transaction coordination
                metadata = {
                    "type": "user_categorization",
                    "source": "autonomous_cognition", 
                    "created_at": datetime.datetime.now().isoformat(),
                    "tags": "user_categorization,personalization,autonomous"
                }
                
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=categorization_thought,
                    memory_type="user_categorization",
                    metadata=metadata,
                    confidence=0.85  # High confidence for user information
                )
                
                if success:
                    logging.info(f"Successfully stored user categorization with ID {memory_id}")
                    
                    # ADDED: Write to reflection file as requested
                    try:
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"memory_consolidation_{timestamp_str}.txt"
                        file_path = os.path.join(self.reflection_path, filename)
                        
                        # Ensure reflection path exists
                        os.makedirs(self.reflection_path, exist_ok=True)
                        
                        file_content = f"""USER INFORMATION CATEGORIZATION
    TIMESTAMP: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    MEMORY_ID: {memory_id}
    confidence: 0.85
    STORAGE: Both databases + reflection file

    {categorization_thought}
    """
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(file_content)
                            
                        logging.info(f"Successfully wrote user categorization to reflection file: {filename}")
                        
                    except Exception as file_error:
                        logging.error(f"Failed to write categorization reflection file: {file_error}")
                        # Don't fail the whole process for file write issues
                    
                    # Create topic-specific summaries for easy retrieval
                    topics = self._extract_user_topics(categorization)
                    self._create_topic_summaries(topics, user_memories)
                    
                    logging.info("User information categorization completed successfully")
                    return True
                else:
                    logging.warning("Failed to store user categorization through transaction coordinator")
                    return False
            else:
                logging.warning("LLM not available for user information categorization")
                return False
        
        except Exception as e:
            logging.error(f"Error in user information categorization: {e}", exc_info=True)
            return False

    def _get_user_related_memories(self):
        """Get memories likely to contain user information."""
        try:
            user_related_memories = []
            
            # Try to get most relevant user memories from vector DB
            user_queries = ["user preferences", "user interests", "user background", 
                            "user family", "user work", "user home"]
            
            for query in user_queries:
                results = self.vector_db.search(
                    query=query,
                    mode="default",
                    k=10
                )
                
                if results:
                    user_related_memories.extend(results)
            
            # Also get memories with user-related types
            user_memory_types = ["user", "preference", "personal", "profile"]
            for memory_type in user_memory_types:
                memories = self.memory_db.get_memories_by_type(memory_type, limit=20)
                if memories:
                    user_related_memories.extend(memories)
            
            # Deduplicate memories
            unique_memories = {}
            for memory in user_related_memories:
                content = memory.get('content', '')
                if content and content not in unique_memories:
                    unique_memories[content] = memory
            
            return list(unique_memories.values())
        
        except Exception as e:
            logging.error(f"Error getting user-related memories: {e}")
            return []

    def _format_memories_for_analysis(self, memories):
        """Format memories for analysis in prompt - handles both dict and string formats."""
        if not memories:
            return "No memories available."
        
        try:
            formatted = []
            for i, memory in enumerate(memories):
                # Handle string format (what we're actually getting)
                if isinstance(memory, str):
                    # Clean up the string and add a simple format
                    clean_memory = memory.strip()
                    if clean_memory:
                        formatted.append(f"[memory_{i+1}] {clean_memory}")
                    continue
                
                # Handle dictionary format (fallback for other memory sources)
                if isinstance(memory, dict):
                    content = memory.get('content', '')
                    memory_type = memory.get('memory_type', memory.get('metadata', {}).get('type', 'unknown'))
                    
                    if content:
                        formatted.append(f"[{memory_type}] {content}")
                    continue
                
                # Handle unexpected formats
                logging.warning(f"Unexpected memory format: {type(memory)}")
                formatted.append(f"[unknown] {str(memory)}")
            
            return "\n\n".join(formatted) if formatted else "No valid memories found."
            
        except Exception as e:
            logging.error(f"Error formatting memories for analysis: {e}")
            # Fallback: just join them as strings
            try:
                return "\n\n".join([str(m) for m in memories if m])
            except:
                return "Error formatting memories."

    def _extract_user_topics(self, categorization):
        """Extract user-related topics from categorization."""
        try:
            # This is a simple extraction - could be improved with more sophisticated parsing
            topics = []
            
            # Look for section headers and list items that might indicate topics
            lines = categorization.split('\n')
            for line in lines:
                # Possible section headers
                if re.match(r'^#+\s+\w+', line) or re.match(r'^[A-Z][A-Za-z\s]+:', line):
                    topic = re.sub(r'^#+\s+', '', line)
                    topic = re.sub(r':$', '', topic)
                    topics.append(topic.strip())
                    
                # List items
                elif re.match(r'^\d+\.\s+[A-Z]', line) or re.match(r'^-\s+[A-Z]', line):
                    topic = re.sub(r'^\d+\.\s+', '', line)
                    topic = re.sub(r'^-\s+', '', line)
                    
                    # Only keep substantive topics (more than one word)
                    words = topic.split()
                    if len(words) >= 2:
                        topics.append(topic.strip())
            
            # Return unique topics
            return list(set(topics))
        
        except Exception as e:
            logging.error(f"Error extracting user topics: {e}")
            return []

    def _create_topic_summaries(self, topics, memories):
        """Create topic-based summaries of user information for efficient retrieval."""
        try:
            if not topics or not memories:
                return
                
            for topic in topics[:5]:  # Limit to top 5 topics to avoid excess processing
                # Create topic summary prompt
                prompt = f""" /no_think
                Create a concise summary of the user's information related to "{topic}" based on these memories:
                
                {self._format_memories_for_analysis(memories[:30])}
                
                The summary should:
                1. Be factual and based only on the provided memories
                2. Include only relevant information to this specific topic
                3. Be written in a neutral, helpful tone
                4. Be structured for easy retrieval and use in conversation
                """
                
                # Generate summary
                if hasattr(self.chatbot, 'llm'):
                    summary = self.chatbot.llm.invoke(prompt)
                    
                    if summary:
                        # Store topic summary with appropriate metadata
                        metadata = {
                            "type": "user_topic",
                            "topic": topic,
                            "source": "autonomous_categorization",
                            "created_at": datetime.datetime.now().isoformat(),
                            "tags": f"user,{topic},summary"
                        }
                        
                        topic_content = f"User Topic - {topic}: {summary}"
                        
                        # Store with transaction coordination
                        success, memory_id = self.chatbot.store_memory_with_transaction(
                            content=topic_content,
                            memory_type="user_topic",
                            metadata=metadata,
                            confidence=0.8  # Higher confidence for user information
                        )
                        
                        if success:
                            logging.info(f"Created user topic summary for '{topic}'")
                        else:
                            logging.warning(f"Failed to store user topic summary for '{topic}'")
        
        except Exception as e:
            logging.error(f"Error creating topic summaries: {e}")

    def _classify_knowledge_gap(self, topic, description):
        """
        Classify a knowledge gap as personal, factual, or ambiguous.
        Returns: 'personal', 'factual', or 'ambiguous'
        """
        personal_indicators = [
            'ken', 'user', 'preference', 'hobby', 'interest', 'personal',
            'family', 'habit', 'routine', 'like', 'dislike', 'opinion',
            # Enhanced personal indicators for better classification
            'feel about', 'think about', 'experience with', 'remember when',
            'your', 'why did', 'how do you', 'what made you'
        ]
        
        factual_indicators = [
            'concept', 'theory', 'principle', 'definition', 'history',
            'technology', 'science', 'process', 'method', 'standard'
        ]
        
        topic_lower = topic.lower()
        desc_lower = description.lower()
        combined = f"{topic_lower} {desc_lower}"
        
        # Calculate base scores
        personal_score = sum(1 for indicator in personal_indicators if indicator in combined)
        factual_score = sum(1 for indicator in factual_indicators if indicator in combined)
        
        # Boost personal score if gap is phrased as question TO Ken
        question_to_ken_patterns = [
            'why did you', 'how do you', 'what do you', 'do you remember',
            'why do you', 'when did you', 'where did you', 'have you',
            'can you tell me about your', 'what was your'
        ]
        
        if any(pattern in combined for pattern in question_to_ken_patterns):
            personal_score += 3  # Strong signal this needs Ken's input
            logging.debug(f"Boosted personal score - question pattern detected in: {topic}")
        
        # Special case: "Ken Bajema" as searchable public figure
        # His name, LinkedIn, articles are PUBLIC_SEARCHABLE not PERSONAL_ASK_KEN
        if "ken bajema" in combined:
            # Check if this is about public information vs private details
            public_info_terms = ['linkedin', 'article', 'publication', 'profile', 
                                'professional', 'work', 'career', 'education']
            private_info_terms = ['feel', 'prefer', 'like', 'family', 'personal life']
            
            has_public_context = any(term in combined for term in public_info_terms)
            has_private_context = any(term in combined for term in private_info_terms)
            
            if has_public_context and not has_private_context:
                # This is about Ken Bajema's public information - searchable
                factual_score += 5  # Strong boost toward factual/searchable
                logging.info(f"Classified as PUBLIC_SEARCHABLE - Ken Bajema public info: {topic}")
            # If has_private_context, let normal personal scoring take over
        
        # Classification logic with scoring
        if personal_score > factual_score and personal_score > 0:
            return 'personal'
        elif factual_score > personal_score and factual_score > 0:
            return 'factual'
        else:
            return 'ambiguous'

    def _fill_knowledge_gaps(self):
        """
        Fill identified knowledge gaps using a tiered approach:
        1. First try DuckDuckGo web search (free, current information)
        2. If web search fails/insufficient, fall back to Claude API (costs $, but deeper knowledge)
        3. If both fail, create a reminder for Ken to address directly in UI
        
        Uses transaction coordination for reliable dual-database storage.
        """
        print("🔍 ====== STARTING KNOWLEDGE GAP FILLING (3-STAGE FALLBACK) ======")
        logging.info("🔍 Starting knowledge gap filling: Web Search → Claude → Reminder")
        
        # Configuration for quality thresholds
        MIN_WEB_RESULTS = 1  # Minimum web results to consider successful
        MIN_CONTENT_LENGTH = 200  # Minimum total content length to consider successful
        
        try:
            # Import required modules
            from web_knowledge_seeker import WebKnowledgeSeeker
            from knowledge_gap import KnowledgeGapQueue
            
            # Initialize components
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            web_seeker = WebKnowledgeSeeker(self.memory_db, self.vector_db, chatbot=self.chatbot)
            
            print("📋 Checking for pending knowledge gaps...")
            
            # Get next gap to fill
            gap = gap_queue.get_next_gap()
            if not gap:
                print("ℹ️ No knowledge gaps in queue to fill")
                logging.info("No knowledge gaps in queue to fill")
                return False
            
            gap_id, topic, description = gap
            print(f"🎯 Selected gap to fill:")
            print(f"   📌 Topic: {topic}")
            print(f"   📝 Description: {description[:100]}...")
            print(f"   🆔 Gap ID: {gap_id}")
            
            logging.info(f"🎯 Attempting to fill knowledge gap: '{topic}'")
            logging.info(f"📋 Gap description: {description}")
            
            # Track which methods were attempted
            web_search_attempted = False
            web_search_success = False
            claude_fallback_attempted = False
            claude_success = False
            
            # =====================================================
            # STAGE 1: Try DuckDuckGo Web Search (Free)
            # =====================================================
            print(f"\n📡 STAGE 1: Searching DuckDuckGo for '{topic}'...")
            logging.info(f"📡 STAGE 1: Web search for '{topic}'")
            
            acquired_knowledge = []
            web_search_attempted = True
            
            try:
                acquired_knowledge = web_seeker.search_for_knowledge(topic, description)
                
                # Evaluate web search quality
                if acquired_knowledge:
                    total_content_length = sum(len(k.get('content', '')) for k in acquired_knowledge)
                    
                    if len(acquired_knowledge) >= MIN_WEB_RESULTS and total_content_length >= MIN_CONTENT_LENGTH:
                        web_search_success = True
                        print(f"   ✅ Web search successful: {len(acquired_knowledge)} results, {total_content_length} chars")
                        logging.info(f"✅ Web search successful: {len(acquired_knowledge)} results, {total_content_length} chars")
                    else:
                        print(f"   ⚠️ Web search returned insufficient results: {len(acquired_knowledge)} results, {total_content_length} chars")
                        logging.warning(f"⚠️ Web search insufficient: {len(acquired_knowledge)} results, {total_content_length} chars")
                else:
                    print(f"   ⚠️ Web search returned no results")
                    logging.warning(f"⚠️ Web search returned no results for '{topic}'")
                    
            except Exception as web_error:
                print(f"   ❌ Web search error: {web_error}")
                logging.error(f"❌ Web search error for '{topic}': {web_error}")
            
            # =====================================================
            # STAGE 2: Claude API Fallback (if web search failed)
            # =====================================================
            if not web_search_success:
                print(f"\n🤖 STAGE 2: Web search insufficient, trying Claude API fallback...")
                logging.info(f"🤖 STAGE 2: Attempting Claude API fallback for '{topic}'")
                
                claude_fallback_attempted = True
                
                try:
                    # Initialize Claude integration
                    claude_integrator = ClaudeKnowledgeIntegration(
                        self.memory_db, 
                        self.vector_db,
                        api_key_file="ClaudeAPIKey.txt"
                    )
                    
                    # Try to get knowledge from Claude
                    claude_success = claude_integrator.integrate_claude_knowledge(topic, description)
                    
                    if claude_success:
                        print(f"   ✅ Claude fallback successful for '{topic}'")
                        logging.info(f"✅ Claude fallback successful for '{topic}'")
                        
                        # Mark gap as fulfilled (Claude integration handles storage)
                        gap_queue.mark_fulfilled(gap_id, items_acquired=1)
                        print(f"🎉 Knowledge gap '{topic}' marked as FULFILLED (via Claude)")
                        logging.info(f"🎉 Knowledge gap '{topic}' fulfilled via Claude fallback")
                        
                        # Create reflection on Claude-acquired knowledge
                        print(f"🤔 Creating reflection on Claude-acquired knowledge...")
                        self._reflect_on_claude_knowledge(topic)
                        
                        print("✅ ====== KNOWLEDGE GAP FILLING COMPLETED (CLAUDE FALLBACK) ======")
                        return True
                    else:
                        print(f"   ⚠️ Claude fallback did not return useful results")
                        logging.warning(f"⚠️ Claude fallback unsuccessful for '{topic}'")
                        
                except ImportError as ie:
                    print(f"   ❌ Claude integration not available: {ie}")
                    logging.error(f"❌ Claude integration import error: {ie}")
                except Exception as claude_error:
                    print(f"   ❌ Claude fallback error: {claude_error}")
                    logging.error(f"❌ Claude fallback error for '{topic}': {claude_error}")
            
            # =====================================================
            # STAGE 3: Store Web Results (if web search succeeded)
            # =====================================================
            if web_search_success and acquired_knowledge:
                print(f"\n💾 Storing {len(acquired_knowledge)} web knowledge items...")
                logging.info(f"💾 Storing {len(acquired_knowledge)} knowledge items from web search")
                
                stored_count = 0
                failed_count = 0
                
                for i, knowledge in enumerate(acquired_knowledge):
                    try:
                        content = knowledge.get('content', '')
                        source_url = knowledge.get('source', 'unknown_source')
                        title = knowledge.get('title', '')
                        topic_tag = knowledge.get('topic', topic)
                        
                        if not content:
                            print(f"   ⚠️ Skipping empty knowledge item {i+1}")
                            failed_count += 1
                            continue
                        
                        # Prepare metadata
                        search_query = (
                            knowledge.get('search_query') or 
                            knowledge.get('query') or 
                            knowledge.get('search_term') or 
                            topic
                        )
                        
                        metadata = {
                            "type": "web_knowledge",
                            "source": source_url,
                            "title": title,
                            "topic": topic_tag,
                            "knowledge_gap_id": gap_id,
                            "search_query": search_query,
                            "extracted_at": knowledge.get('extracted_at'),
                            "relevance_score": knowledge.get('relevance_score', 0.8),
                            "acquisition_method": "duckduckgo_web_search",
                            "created_at": datetime.datetime.now().isoformat(),
                            "tags": f"knowledge_gap,web_search,{topic_tag}"
                        }
                        
                        print(f"   💾 Storing item {i+1}: {title[:50]}..." if title else f"   💾 Storing item {i+1}")
                        
                        # Store using transaction coordination
                        success, memory_id = self.chatbot.store_memory_with_transaction(
                            content=content,
                            memory_type="web_knowledge",
                            metadata=metadata,
                            confidence=0.8
                        )
                        
                        if success:
                            stored_count += 1
                            print(f"      ✅ Stored with ID: {memory_id}")
                            logging.info(f"✅ Stored knowledge item {i+1}/{len(acquired_knowledge)}")
                        else:
                            failed_count += 1
                            print(f"      ❌ Storage failed")
                            logging.error(f"❌ Failed to store knowledge item {i+1}")
                            
                    except Exception as item_error:
                        failed_count += 1
                        print(f"      ❌ Error: {item_error}")
                        logging.error(f"❌ Error processing knowledge item {i+1}: {item_error}")
                
                # Report results
                print(f"\n📊 STORAGE RESULTS:")
                print(f"   ✅ Successfully stored: {stored_count} items")
                print(f"   ❌ Failed to store: {failed_count} items")
                
                if stored_count > 0:
                    success_rate = (stored_count / (stored_count + failed_count) * 100)
                    print(f"   📈 Success rate: {success_rate:.1f}%")
                    
                    # Mark gap as fulfilled
                    gap_queue.mark_fulfilled(gap_id, stored_count)
                    print(f"🎉 Knowledge gap '{topic}' marked as FULFILLED!")
                    logging.info(f"🎉 Knowledge gap '{topic}' fulfilled with {stored_count} items")
                    
                    # Create reflection
                    print(f"🤔 Creating reflection on acquired knowledge...")
                    self._reflect_on_new_knowledge(topic, acquired_knowledge)
                    
                    print("✅ ====== KNOWLEDGE GAP FILLING COMPLETED (WEB SEARCH) ======")
                    return True
            
            # =====================================================
            # STAGE 3 (FALLBACK): Create Reminder for Ken
            # Both web search and Claude failed - escalate to human
            # =====================================================
            print(f"\n📝 STAGE 3: Both automated methods failed, creating reminder for Ken...")
            logging.info(f"📝 STAGE 3: Creating reminder fallback for '{topic}'")
            
            print(f"   📡 Web search attempted: {web_search_attempted}, success: {web_search_success}")
            print(f"   🤖 Claude fallback attempted: {claude_fallback_attempted}, success: {claude_success}")
            
            try:
                # Create a detailed reminder for Ken
                due_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                
                # Build informative reminder text
                reminder_text = (
                    f"Knowledge Gap (auto-fill failed): {topic}\n"
                    f"Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
                    f"Note: Both web search and Claude API failed to fill this gap. "
                    f"Please discuss with the model directly."
                )
                
                # Use the existing reminder creation method
                reminder_success = self._create_reminder_for_personal_gap(reminder_text, due_date)
                
                if reminder_success:
                    print(f"   ✅ Reminder created for Ken to address: '{topic}'")
                    logging.info(f"✅ Created reminder for unfilled gap: '{topic}'")
                    
                    # Mark gap as fulfilled since reminder is the fulfillment mechanism
                    gap_queue.mark_fulfilled(gap_id, items_acquired=0)
                    print(f"   🔗 Marked knowledge gap {gap_id} as 'fulfilled' (reminder created)")
                    logging.info(f"🔗 Gap {gap_id} marked fulfilled - Ken will address via reminder")
                    
                    # Record thought about the escalation
                    self._record_thought(
                        thought_type="knowledge_gap_escalation",
                        content=f"Escalated knowledge gap to Ken via reminder: '{topic}'. "
                               f"Web search {'attempted but failed' if web_search_attempted else 'not attempted'}. "
                               f"Claude fallback {'attempted but failed' if claude_fallback_attempted else 'not attempted'}."
                    )
                    
                    print("✅ ====== KNOWLEDGE GAP FILLING COMPLETED (REMINDER FALLBACK) ======")
                    return True
                else:
                    print(f"   ❌ Failed to create reminder")
                    logging.error(f"❌ Failed to create reminder for gap: '{topic}'")
                    
            except Exception as reminder_error:
                print(f"   ❌ Reminder creation error: {reminder_error}")
                logging.error(f"❌ Error creating reminder for '{topic}': {reminder_error}")
            
            # =====================================================
            # ALL STAGES FAILED - This should be rare
            # =====================================================
            print(f"\n❌ ALL STAGES FAILED (including reminder creation)")
            print(f"   Topic: {topic}")
            logging.error(f"❌ All stages failed for knowledge gap '{topic}' - gap remains pending")
            
            # Don't mark as failed - leave pending for retry
            # The gap will be retried after the cooldown period
            
            print("❌ ====== KNOWLEDGE GAP FILLING FAILED ======")
            return False
            
        except ImportError as ie:
            print(f"❌ Missing required module: {ie}")
            logging.error(f"❌ Missing required module for knowledge gap filling: {ie}")
            return False
        except Exception as e:
            print(f"❌ Error in knowledge gap filling: {e}")
            logging.error(f"❌ Error in knowledge gap filling: {e}", exc_info=True)
            return False


# ============================================================
# SUMMARY OF THE 3-STAGE APPROACH
# ============================================================
#
# STAGE 1: DuckDuckGo Web Search (FREE)
#   - Tries to find information via web search
#   - If successful: Store results → Mark fulfilled → Done ✓
#   - If fails: Continue to Stage 2
#
# STAGE 2: Claude API Fallback (COSTS $)
#   - Only called if web search fails/insufficient
#   - If successful: Mark fulfilled → Done ✓
#   - If fails: Continue to Stage 3
#
# STAGE 3: Create Reminder for Ken (HUMAN FALLBACK)
#   - Creates a reminder so Ken can address it in the UI
#   - If successful: Mark fulfilled → Done ✓
#   - If fails: Leave gap pending for retry
#
# BENEFITS:
#   - Cost-effective: Free web search tried first
#   - Comprehensive: Claude provides backup for complex topics
#   - No gaps lost: Human fallback ensures nothing falls through
#   - Clean state: Gaps marked fulfilled, no stale pending items
#
# ============================================================

    def _fill_complex_knowledge_gaps(self):
        """
        DEPRECATED: Complex knowledge gap filling has been consolidated into _fill_knowledge_gaps().
        
        The new _fill_knowledge_gaps() method uses a tiered approach:
        1. Web search first (free)
        2. Claude API fallback (if web search fails)
        
        This eliminates the need for a separate "complex gaps" handler.
        
        Returns:
            bool: Always returns False (method is deprecated)
        """
        logging.warning("⚠️ _fill_complex_knowledge_gaps() is DEPRECATED. "
                       "Use _fill_knowledge_gaps() which now includes Claude fallback.")
        logging.info("   The tiered approach (web search → Claude fallback) is now in _fill_knowledge_gaps()")
        return False
            
    def _consult_claude_for_personal_gap(self, gap_id, topic, description, gap_queue):
        """
        DEPRECATED: Personal knowledge gaps should NOT consult Claude.
        
        Claude has no special knowledge about Ken Bajema - any response would be
        guessing or generic advice. Personal gaps should be handled via the 
        reminder system where Ken can address them directly.
        
        This method now simply logs a warning and returns False, directing the
        caller to use the reminder-based workflow instead.
        
        Args:
            gap_id: ID of the knowledge gap
            topic: Topic of the gap
            description: Description of the gap
            gap_queue: KnowledgeGapQueue instance
            
        Returns:
            bool: Always returns False to indicate Claude should not be consulted
        """
        logging.warning(f"⚠️ _consult_claude_for_personal_gap() called for '{topic}' - "
                       f"This is deprecated. Personal gaps should use reminder system, not Claude API.")
        logging.info(f"   Personal gaps about Ken Bajema should be addressed via reminders, "
                    f"not Claude consultation (Claude has no knowledge about Ken specifically).")
        
        # Return False to indicate this method should not be used
        # Caller should create a reminder instead
        return False

        
    def _mark_gap_for_user_input(self, gap_id, topic, description, advice=None):
        """
        Mark a gap as requiring direct user input.
        
        Args:
            gap_id: ID of the knowledge gap
            topic: Topic of the gap
            description: Description of the gap  
            advice: Optional advice text (no longer stored since Claude consultation is deprecated)
        """
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE knowledge_gaps 
                    SET status = 'requires_user_input',
                        last_attempt_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (gap_id,))
                conn.commit()
                
            logging.info(f"Marked gap {gap_id} ('{topic}') as requiring user input")
            return True
            
        except Exception as e:
            logging.error(f"Error marking gap {gap_id} for user input: {e}")
            return False

            
        except Exception as e:
            logging.error(f"Error marking gap for user input: {e}")
            return False

    def _store_claude_personal_knowledge(self, topic, description, claude_response):
        """Store knowledge provided by Claude about personal topics."""
        try:
            content = f"# Knowledge from Claude: {topic}\n\n{claude_response}\n\nNote: This information was provided by Claude in response to a knowledge gap inquiry."
            
            metadata = {
                "type": "claude_personal_knowledge",
                "topic": topic,
                "source": "claude_consultation",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": f"personal,claude,{topic}"
            }
            
            # Use transaction coordination
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=content,
                memory_type="claude_personal_knowledge",
                metadata=metadata,
                confidence=0.7
            )
            
            return success
            
        except Exception as e:
            logging.error(f"Error storing Claude personal knowledge: {e}")
            return False

    def _identify_complex_gaps(self):
        """Identify knowledge gaps that are complex and best suited for Claude."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, topic, description FROM knowledge_gaps
                    WHERE status = 'pending' AND (
                        description LIKE '%concept%' OR
                        description LIKE '%theory%' OR
                        description LIKE '%framework%' OR
                        description LIKE '%principles%' OR
                        description LIKE '%comparison%' OR
                        description LIKE '%evaluate%' OR
                        description LIKE '%analyze%'
                    )
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 3
                ''')
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"Error identifying complex gaps: {e}")
            return []

    def _reflect_on_new_knowledge(self, topic, acquired_knowledge):
        """
        Reflect on newly acquired knowledge and create a summary/reflection that is
        properly stored in both databases using transaction coordination.
        
        Args:
            topic (str): The topic of the knowledge gap that was filled
            acquired_knowledge (list): List of knowledge items that were acquired
            
        Returns:
            bool: Success status
        """
        try:
            logging.info(f"🤔 Reflecting on new knowledge about '{topic}'")
            
            # Format the acquired knowledge for reflection
            knowledge_summaries = []
            for item in acquired_knowledge[:5]:  # Limit to first 5 items for reflection
                content = item.get('content', '')
                source = item.get('source', 'unknown')
                knowledge_summaries.append(f"From {source}: {content[:200]}...")
            
            knowledge_text = "\n\n".join(knowledge_summaries)
            
            # Create reflection prompt
            reflection_prompt = f""" /no_think
            I've successfully acquired new knowledge about '{topic}' from web searches. 
            I'll reflect on this information to integrate it into my understanding:
            
            NEW KNOWLEDGE ACQUIRED:
            {knowledge_text}
            
            I'll create a reflection that:
            1. Summarizes the key insights learned about '{topic}'
            2. Identifies how this knowledge enhances my ability to assist the user
            3. Notes any connections to existing knowledge
            4. Recognizes practical applications for this information
            
            My reflection will help me better utilize this knowledge in future conversations.
            """
            
            # Generate reflection using LLM with safe invoke method
            reflection = self._safe_llm_invoke(reflection_prompt)
            
            if not reflection:
                logging.warning(f"Failed to generate reflection for new knowledge about '{topic}'")
                return False
            
            # Create formatted reflection content
            reflection_content = f"# Knowledge Acquisition Reflection: {topic}\n\n{reflection}\n\n## Sources Consulted\n"
            
            # Add source summary
            sources = set()
            for item in acquired_knowledge:
                source = item.get('source', 'unknown')
                if source != 'unknown':
                    sources.add(source)
            
            if sources:
                reflection_content += "\n".join([f"- {source}" for source in list(sources)[:5]])
            
            # Prepare metadata for storage
            metadata = {
                "type": "knowledge_reflection",
                "topic": topic,
                "source": "autonomous_cognition",
                "acquisition_method": "web_search_reflection",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": f"knowledge_gap,{topic},reflection,web_acquired",
                "items_reflected_on": len(acquired_knowledge)
            }
            
            # CRITICAL: Use transaction coordination for database consistency
            if not hasattr(self.chatbot, 'store_memory_with_transaction'):
                error_msg = "Transaction coordinator not available - cannot store knowledge reflection safely"
                logging.error(error_msg)
                return False
            
            # Store using transaction coordination
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=reflection_content,
                memory_type="knowledge_reflection",
                metadata=metadata,
                confidence=0.8  # High confidence for knowledge reflections
            )
            
            if success:
                logging.info(f"✅ Successfully stored knowledge reflection on '{topic}' with ID {memory_id}")
                return True
            else:
                logging.error(f"❌ Failed to store knowledge reflection through transaction coordinator")
                return False
            
        except Exception as e:
            logging.error(f"❌ Error reflecting on new knowledge about '{topic}': {e}", exc_info=True)
            return False

    def _reflect_on_claude_knowledge(self, topic):
        """
        Reflect on knowledge obtained from Claude to integrate it with existing knowledge.
        Uses transaction coordination to ensure proper storage in both databases.
        
        Args:
            topic (str): The topic of the knowledge acquired from Claude
            
        Returns:
            bool: Success status
        """
        try:
            logging.info(f"Reflecting on knowledge from Claude about '{topic}'")
            
            # Get relevant existing memories on this topic to provide context
            relevant_memories = self.vector_db.search(
                query=topic,
                mode="default",
                k=5
            )
            
            # Format existing memories for context
            existing_knowledge = ""
            if relevant_memories:
                existing_knowledge = "\n\n".join([
                    f"Memory: {mem.get('content', '')}" 
                    for mem in relevant_memories
                ])
            
            # Create reflection prompt
            reflection_prompt = f""" /no_think
            I've acquired specialized knowledge from Claude about '{topic}'. I'll reflect on how 
            this integrates with my existing understanding:
            
            EXISTING RELATED KNOWLEDGE:
            {existing_knowledge}
            
            I'll create a reflection that:
            1. Notes how Claude's knowledge complements my existing understanding
            2. Identifies new perspectives or insights gained
            3. Considers how this knowledge can enhance my assistance capabilities
            4. Recognizes the source of this specialized knowledge
            
            My reflection will help me better integrate and attribute this information appropriately.
            """
            
            # Generate reflection using safe LLM invoke method
            reflection = self._safe_llm_invoke(reflection_prompt)
            
            if not reflection:
                logging.warning(f"Failed to generate reflection for Claude knowledge about '{topic}'")
                return False
            
            # Create formatted reflection content
            reflection_content = f"# Claude Knowledge Integration: {topic}\n\n{reflection}"
            
            # Prepare metadata for storage
            metadata = {
                "type": "claude_knowledge",
                "topic": topic,
                "source": "claude_knowledge_integration",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": f"claude,{topic},specialized_knowledge",
                "confidence": 0.8  # Higher confidence for specialized knowledge
            }
            
            # Store using transaction coordination for dual database consistency
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=reflection_content,
                    memory_type="claude_knowledge",  # Use a specific type for easier retrieval
                    metadata=metadata,
                    confidence=0.8  # Higher confidence for specialized knowledge
                )
                
                if success:
                    logging.info(f"Successfully stored Claude knowledge reflection on '{topic}' with ID {memory_id}")
                    return True
                else:
                    logging.warning(f"Failed to store Claude knowledge reflection through transaction coordinator")
                    return False
            else:
                # Fallback if transaction coordinator isn't available
                logging.warning("Transaction coordinator not available, using direct storage")
                # First store in memory_db
                memory_success = self.memory_db.store_memory(
                    content=reflection_content,
                    memory_type="claude_knowledge",
                    source="claude_knowledge_integration",
                    confidence=0.8,
                    tags=f"claude,{topic},specialized_knowledge",
                    additional_metadata=metadata
                )
                
                # Then store in vector_db if memory_db was successful
                if memory_success and hasattr(self, 'vector_db') and self.vector_db is not None:
                    vector_success = self.vector_db.add_text(
                        text=reflection_content,
                        metadata=metadata
                    )
                    return vector_success
                
                return memory_success
            
        except Exception as e:
            logging.error(f"Error reflecting on Claude knowledge: {e}", exc_info=True)
            return False
        
    def _audit_memory_confidence(self):
        """
        Audit stored memories for confidence calibration and update if needed.
        
        This task:
        1. Retrieves 5 oldest memories (prioritizing those never audited)
        2. Evaluates confidence based on source type and linguistic indicators
        3. Updates memories via transaction coordinator if confidence change > 0.1
        4. Writes detailed audit report to reflections folder
        
        Source-based confidence baselines:
        - Ken's direct statements: 0.9-1.0
        - Claude knowledge (claude_learning): 0.8-0.9
        - Document summaries (document_summary): 0.6-0.8
        - Image analysis (image_analysis): 0.6-0.8
        - Web knowledge (web_knowledge): 0.4-0.6
        - Inferences/unknown: 0.3-0.5
        """
        logging.info("🔍 Starting memory confidence audit (with database updates)")
        self.cognitive_state = "auditing_confidence"
        
        # --- Configuration ---
        MAX_MEMORIES_PER_RUN = 5
        CONFIDENCE_CHANGE_THRESHOLD = 0.1
        REFLECTIONS_PATH = r"C:\Users\kenba\source\repos\Ollama3\reflections"
        
        # --- Track metrics ---
        memories_evaluated = 0
        memories_updated = 0
        memories_unchanged = 0
        errors = 0
        audit_results = []
        
        try:
            # Record start of audit
            self._record_thought(
                thought_type="confidence_audit",
                content="Beginning memory confidence audit with source-based evaluation and database updates."
            )
            
            # --- Step 1: Retrieve memories to audit ---
            logging.info(f"📊 Retrieving up to {MAX_MEMORIES_PER_RUN} memories for confidence audit")
            memories_to_audit = self._get_memories_for_confidence_audit(limit=MAX_MEMORIES_PER_RUN)
            
            if not memories_to_audit:
                logging.info("✅ No memories found requiring confidence audit")
                self._record_thought(
                    thought_type="confidence_audit",
                    content="No memories found requiring confidence audit at this time."
                )
                return True
            
            logging.info(f"Found {len(memories_to_audit)} memories to audit")
            
            # --- Step 2: Evaluate each memory ---
            for memory in memories_to_audit:
                memories_evaluated += 1
                
                try:
                    content = memory.get('content', '')
                    # FIXED: Extract memory_type from metadata.type (where vector_db stores it)
                    metadata = memory.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    memory_type = metadata.get('type', 'unknown')
                    current_confidence = self._extract_current_confidence(memory)
                    created_at = self._extract_created_at(memory)
                    
                    # Skip if content is too short
                    if not content or len(content) < 30:
                        logging.debug(f"Skipping short/empty memory")
                        continue
                    
                    logging.info(f"📝 Evaluating memory {memories_evaluated}/{len(memories_to_audit)}: {content[:50]}...")
                    
                    # --- Step 3: Determine source-based baseline confidence ---
                    baseline_confidence = self._get_source_baseline_confidence(memory_type)
                    
                    # --- Step 4: LLM evaluates linguistic indicators ---
                    evaluation_result = self._evaluate_memory_confidence_with_llm(
                        content=content,
                        memory_type=memory_type,
                        current_confidence=current_confidence,
                        baseline_confidence=baseline_confidence
                    )
                    
                    if not evaluation_result:
                        logging.warning(f"Failed to evaluate memory: {content[:50]}...")
                        errors += 1
                        audit_results.append({
                            'content_preview': content[:200],
                            'memory_type': memory_type,
                            'created_at': created_at,
                            'before_confidence': current_confidence,
                            'after_confidence': None,
                            'action': 'ERROR',
                            'reasoning': 'LLM evaluation failed'
                        })
                        continue
                    
                    recommended_confidence = evaluation_result.get('recommended_confidence', current_confidence)
                    reasoning = evaluation_result.get('reasoning', 'No reasoning provided')
                    
                    # --- Step 5: Determine if update is needed ---
                    confidence_difference = abs(recommended_confidence - current_confidence)
                    # Only update if difference is GREATER than threshold (not equal)
                    needs_update = confidence_difference > CONFIDENCE_CHANGE_THRESHOLD
                    
                    if needs_update:
                        # --- Step 6: Update memory via transaction coordinator ---
                        logging.info(f"🔄 Updating confidence: {current_confidence:.2f} → {recommended_confidence:.2f}")
                        
                        # Check cooldown before updating
                        if not self._check_forget_cooldown(content):
                            logging.warning(f"⏳ Skipping update due to forget cooldown: {content[:50]}...")
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'SKIPPED_COOLDOWN',
                                'reasoning': reasoning
                            })
                            continue
                        
                        # Perform the update using FORGET/STORE pattern
                        update_success = self._update_memory_confidence(
                            memory=memory,
                            new_confidence=recommended_confidence
                        )
                        
                        if update_success:
                            memories_updated += 1
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'UPDATED',
                                'reasoning': reasoning
                            })
                            logging.info(f"✅ Successfully updated memory confidence")
                        else:
                            errors += 1
                            audit_results.append({
                                'content_preview': content[:200],
                                'memory_type': memory_type,
                                'created_at': created_at,
                                'before_confidence': current_confidence,
                                'after_confidence': recommended_confidence,
                                'action': 'UPDATE_FAILED',
                                'reasoning': reasoning
                            })
                            logging.error(f"❌ Failed to update memory confidence")
                    else:
                        # No update needed
                        memories_unchanged += 1
                        audit_results.append({
                            'content_preview': content[:200],
                            'memory_type': memory_type,
                            'created_at': created_at,
                            'before_confidence': current_confidence,
                            'after_confidence': current_confidence,
                            'action': 'NO_CHANGE',
                            'reasoning': reasoning
                        })
                        logging.info(f"✅ No change needed (difference {confidence_difference:.2f} < threshold {CONFIDENCE_CHANGE_THRESHOLD})")
                        
                except Exception as memory_error:
                    errors += 1
                    logging.error(f"Error processing memory: {memory_error}", exc_info=True)
                    audit_results.append({
                        'content_preview': memory.get('content', 'N/A')[:100],
                        'memory_type': memory.get('memory_type', 'unknown'),
                        'created_at': 'unknown',
                        'before_confidence': 0.0,
                        'after_confidence': None,
                        'action': 'ERROR',
                        'reasoning': str(memory_error)
                    })
                    continue
            
            # --- Step 7: Write audit report to reflections folder ---
            logging.info("📄 Generating audit report")
            audit_report = self._generate_confidence_audit_report(
                memories_evaluated=memories_evaluated,
                memories_updated=memories_updated,
                memories_unchanged=memories_unchanged,
                errors=errors,
                audit_results=audit_results
            )
            
            # Write to reflections folder
            try:
                os.makedirs(REFLECTIONS_PATH, exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"confidence_audit_{timestamp_str}.txt"
                file_path = os.path.join(REFLECTIONS_PATH, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(audit_report)
                
                logging.info(f"✅ Audit report written to: {file_path}")
                
            except Exception as file_error:
                logging.error(f"❌ Failed to write audit report: {file_error}")
            
            # --- Step 8: Record completion ---
            self._record_thought(
                thought_type="confidence_audit",
                content=f"Completed confidence audit: {memories_evaluated} evaluated, {memories_updated} updated, {memories_unchanged} unchanged, {errors} errors."
            )
            
            logging.info(f"✅ Confidence audit complete: {memories_updated}/{memories_evaluated} memories updated")
            return True
            
        except Exception as e:
            logging.error(f"Error in memory confidence audit: {e}", exc_info=True)
            self._record_thought(
                thought_type="error",
                content=f"Error during confidence audit: {str(e)}"
            )
            return False
            
        finally:
            # Update activity timestamp and reset state
            self._update_activity_timestamp('audit_memory_confidence')
            self.cognitive_state = "idle"
            logging.info("📋 Completed memory confidence audit process")

    def _get_memories_for_confidence_audit(self, limit: int = 5) -> list:
        """
        Retrieve memories for confidence audit, prioritizing oldest memories
        that have never been audited.
        
        Args:
            limit (int): Maximum number of memories to return
            
        Returns:
            list: List of memory dictionaries to audit
        """
        try:
            logging.info(f"🔍 Searching for memories to audit (limit={limit})")
            
            # Search queries to find diverse memories
            search_queries = [
                "user preference",
                "personal information",
                "web knowledge",
                "document summary",
                "image analysis",
                "claude learning",
                "conversation summary",
                "important information"
            ]
            
            all_memories = []
            
            # Collect memories from various search queries
            for query in search_queries:
                try:
                    results = self.vector_db.search(
                        query=query,
                        mode="default",
                        k=10
                    )
                    if results:
                        all_memories.extend(results)
                except Exception as search_error:
                    logging.debug(f"Search error for query '{query}': {search_error}")
                    continue
            
            # Remove duplicates based on content hash
            unique_memories = {}
            for memory in all_memories:
                content = memory.get('content', '')
                if content and len(content) > 30:
                    content_hash = hash(content)
                    if content_hash not in unique_memories:
                        unique_memories[content_hash] = memory
            
            memories_list = list(unique_memories.values())
            logging.info(f"Found {len(memories_list)} unique memories")
            
            # Sort by created_at (oldest first), prioritizing those without last_confidence_audit
            def sort_key(mem):
                """Sort by: 1) never audited first, 2) oldest created_at"""
                metadata = mem.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                # Check if ever audited
                last_audit = metadata.get('last_confidence_audit', None)
                never_audited = 0 if last_audit is None else 1
                
                # Get created_at for secondary sort
                created_at = metadata.get('created_at', '9999-12-31')
                
                return (never_audited, created_at)
            
            memories_list.sort(key=sort_key)
            
            logging.info(f"✅ Returning {min(limit, len(memories_list))} memories for audit (oldest/never-audited first)")
            return memories_list[:limit]
            
        except Exception as e:
            logging.error(f"Error getting memories for confidence audit: {e}", exc_info=True)
            return []

    def _get_source_baseline_confidence(self, memory_type: str) -> float:
        """
        Get the baseline confidence level based on memory source type.
        
        Args:
            memory_type (str): The memory_type field value
            
        Returns:
            float: Baseline confidence value (0.0-1.0)
        """
        # Source-based confidence baselines
        source_baselines = {
            # High confidence sources (Ken and Claude)
            'user_preference': 0.9,
            'personal_info': 0.9,
            'user_statement': 0.95,
            'claude_learning': 0.85,
            
            # Medium-high confidence (model's own reasoning)
            'self_dialogue_summary': 0.8,   # Multi-turn reasoned conclusions
            'self': 0.75,                    # Individual insights from self-dialogue
            
            # Medium confidence sources (documents and images)
            'document_summary': 0.7,
            'image_analysis': 0.7,
            'conversation_summary': 0.75,
            'user_categorization': 0.8,
            
            # Lower confidence sources (web and inferences)
            'web_knowledge': 0.5,
            'inference': 0.4,
            'assumption': 0.35,

            # Medium-high confidence (user topics and commands)
            'user_topic': 0.75,              # User-related topic information
            'command': 0.7,                  # Command documentation/examples
            'reminder': 0.85,                # User-created reminders (Ken set these)
            
            # Default for unknown types
            'unknown': 0.5,
            'general': 0.5
        }
        
        # Normalize memory_type to lowercase for matching
        memory_type_lower = memory_type.lower() if memory_type else 'unknown'
        
        # Return baseline, defaulting to 0.5 for unknown types
        return source_baselines.get(memory_type_lower, 0.5)

    def _extract_current_confidence(self, memory: dict) -> float:
        """
        Extract the current confidence value from a memory.
        
        Args:
            memory (dict): Memory dictionary
            
        Returns:
            float: Current confidence value (defaults to 0.5)
        """
        try:
            # Try to get confidence from top-level
            confidence = memory.get('confidence')
            
            # If not found, check metadata
            if confidence is None:
                metadata = memory.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                confidence = metadata.get('confidence')
            
            # Convert to float if needed
            if confidence is not None:
                return float(confidence)
            
            return 0.5  # Default
            
        except (ValueError, TypeError):
            return 0.5

    def _extract_created_at(self, memory: dict) -> str:
        """
        Extract the created_at timestamp from a memory.
        
        Args:
            memory (dict): Memory dictionary
            
        Returns:
            str: Created at timestamp or 'unknown'
        """
        try:
            metadata = memory.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            return metadata.get('created_at', 'unknown')
            
        except Exception:
            return 'unknown'

    def _evaluate_memory_confidence_with_llm(self, content: str, memory_type: str, 
                                              current_confidence: float, 
                                              baseline_confidence: float) -> dict:
        """
        Use LLM to evaluate memory confidence based on linguistic indicators.
        
        Args:
            content (str): Memory content to evaluate
            memory_type (str): Type of memory (for context)
            current_confidence (float): Current confidence value
            baseline_confidence (float): Source-based baseline confidence
            
        Returns:
            dict: Evaluation result with 'recommended_confidence' and 'reasoning'
        """
        try:
            evaluation_prompt = f""" /no_think
Evaluate the confidence level for this stored memory. You MUST recommend a specific confidence value.

MEMORY CONTENT:
{content[:1500]}

MEMORY TYPE: {memory_type}
CURRENT CONFIDENCE: {current_confidence}
SOURCE-BASED BASELINE FOR THIS TYPE: {baseline_confidence}

CONFIDENCE SCALE (use these ranges):
- 0.9-1.0: Direct statements from Ken, verified facts, explicit attribution ("Ken said...", "Ken confirmed...")
- 0.8-0.9: Claude knowledge, self-dialogue conclusions, high-quality reasoned content
- 0.6-0.8: Document summaries, image analysis, user topics, clear context
- 0.4-0.6: Web knowledge, moderate inferences
- 0.3-0.5: Assumptions, unclear source, needs verification
- 0.1-0.2: Appears questionable, speculation, conflicting language

SOURCE TYPE BASELINES (start here, then adjust based on content):
- user_preference, personal_info, user_statement → 0.9-1.0
- claude_learning → 0.85
- self_dialogue_summary → 0.8
- user_topic → 0.75
- conversation_summary → 0.75
- document_summary, image_analysis, command → 0.7
- web_knowledge → 0.5
- general, unknown → 0.5 (but ADJUST based on content!)

EVALUATION RULES:
1. START with the source baseline for this memory type ({baseline_confidence})
2. INCREASE confidence if: direct quotes, clear attribution, "Ken said/mentioned/confirmed"
3. DECREASE confidence if: hedging words ("might", "possibly", "seems"), speculation, no clear source
4. For "general" type: Look at the CONTENT to determine what it actually is
5. If content mentions Ken directly with clear context, confidence should be 0.7+
6. DO NOT just keep the current confidence - actively evaluate and recommend!

TASK: Respond in this EXACT format (no other text):
RECOMMENDED_CONFIDENCE: [0.0-1.0 numeric value - BE SPECIFIC, not just the baseline]
REASONING: [2-3 sentences explaining your specific evaluation]
"""
            
            # Get LLM evaluation
            response = self._safe_llm_invoke(evaluation_prompt)
            
            if not response:
                logging.warning("LLM returned empty response for confidence evaluation")
                return None
            
            # Parse response
            result = {}
            
            # Extract recommended confidence
            confidence_match = re.search(r'RECOMMENDED_CONFIDENCE:\s*([\d.]+)', response)
            if confidence_match:
                try:
                    recommended = float(confidence_match.group(1))
                    # Clamp to valid range
                    result['recommended_confidence'] = max(0.0, min(1.0, recommended))
                except ValueError:
                    result['recommended_confidence'] = baseline_confidence
            else:
                result['recommended_confidence'] = baseline_confidence
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=$|\n\n)', response, re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            else:
                result['reasoning'] = "No detailed reasoning provided by evaluation."
            
            return result
            
        except Exception as e:
            logging.error(f"Error in LLM confidence evaluation: {e}", exc_info=True)
            return None

    def _update_memory_confidence(self, memory: dict, new_confidence: float) -> bool:
        """
        Update a memory's confidence using the transaction coordinator (FORGET/STORE pattern).
        
        Args:
            memory (dict): Memory to update
            new_confidence (float): New confidence value
            
        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            content = memory.get('content', '')
            memory_type = memory.get('memory_type', 'unknown')
            
            if not content:
                logging.error("Cannot update memory with empty content")
                return False
            
            # Get existing metadata and update it
            metadata = memory.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            # Update metadata with new confidence and audit timestamp
            metadata['confidence'] = new_confidence
            metadata['last_confidence_audit'] = datetime.datetime.now().isoformat()
            
            # Preserve existing type in metadata
            if 'type' not in metadata:
                metadata['type'] = memory_type
            
            # --- FORGET old memory ---
            logging.info(f"🗑️ Forgetting old memory: {content[:50]}...")
            forget_result, forget_success = self.chatbot.deepseek_enhancer._handle_regular_memory_forget(content)
            
            if not forget_success:
                logging.error(f"❌ Failed to forget old memory: {forget_result}")
                return False
            
            logging.info("✅ Old memory forgotten successfully")
            
            # --- STORE updated memory ---
            logging.info(f"💾 Storing updated memory with confidence {new_confidence:.2f}")
            store_success, memory_id = self.chatbot.store_memory_with_transaction(
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                confidence=new_confidence
            )
            
            if store_success:
                logging.info(f"✅ Memory updated successfully with ID: {memory_id}")
                return True
            else:
                logging.error(f"❌ CRITICAL: Forgot memory but failed to store updated version!")
                return False
                
        except Exception as e:
            logging.error(f"Error updating memory confidence: {e}", exc_info=True)
            return False

    def _generate_confidence_audit_report(self, memories_evaluated: int, memories_updated: int,
                                           memories_unchanged: int, errors: int,
                                           audit_results: list) -> str:
        """
        Generate a detailed audit report for the confidence audit task.
        
        Args:
            memories_evaluated (int): Total memories evaluated
            memories_updated (int): Memories that were updated
            memories_unchanged (int): Memories that didn't need changes
            errors (int): Number of errors encountered
            audit_results (list): Detailed results for each memory
            
        Returns:
            str: Formatted audit report
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""================================================================================
MEMORY CONFIDENCE AUDIT REPORT
Generated: {timestamp}
Memories Evaluated: {memories_evaluated}
Memories Modified: {memories_updated}
================================================================================

"""
        
        # Add individual memory sections
        for i, result in enumerate(audit_results, 1):
            report += f"""--------------------------------------------------------------------------------
MEMORY {i} of {len(audit_results)}
--------------------------------------------------------------------------------
CONTENT PREVIEW: {result.get('content_preview', 'N/A')}...
MEMORY TYPE: {result.get('memory_type', 'unknown')}
CREATED: {result.get('created_at', 'unknown')}

BEFORE:
  Confidence: {result.get('before_confidence', 'N/A')}
  
AFTER:
  Confidence: {result.get('after_confidence', 'N/A') if result.get('after_confidence') is not None else 'N/A (error)'}
  
REASONING:
  {result.get('reasoning', 'No reasoning provided')}

ACTION TAKEN: {result.get('action', 'UNKNOWN')}

"""
        
        # Add summary section
        # Calculate confidence change statistics
        increases = []
        decreases = []
        
        for result in audit_results:
            if result.get('action') == 'UPDATED':
                before = result.get('before_confidence', 0)
                after = result.get('after_confidence', 0)
                if after is not None:
                    diff = after - before
                    if diff > 0:
                        increases.append(diff)
                    elif diff < 0:
                        decreases.append(abs(diff))
        
        avg_increase = sum(increases) / len(increases) if increases else 0
        avg_decrease = sum(decreases) / len(decreases) if decreases else 0
        
        report += f"""================================================================================
AUDIT SUMMARY
================================================================================
Total Evaluated: {memories_evaluated}
Updated: {memories_updated}
Unchanged: {memories_unchanged}
Errors: {errors}

Confidence Changes:
  - Increased: {len(increases)} (avg +{avg_increase:.2f})
  - Decreased: {len(decreases)} (avg -{avg_decrease:.2f})
  - Unchanged: {memories_unchanged}

Source-Based Confidence Guidelines Used:
  - Ken's direct statements: 0.9-1.0
  - Claude knowledge (claude_learning): 0.8-0.9
  - Document summaries: 0.6-0.8
  - Image analysis: 0.6-0.8
  - Web knowledge: 0.4-0.6
  - Inferences/unknown: 0.3-0.5

Next scheduled audit: ~84 hours
================================================================================
"""
        
        return report

       
    def start_cognitive_thread(self):
        """Start the autonomous memory management thread if it's not already running."""
        if self.thinking_thread is None or not self.thinking_thread.is_alive():
            self.stop_flag.clear()
            self.thinking_thread = threading.Thread(
                target=self._cognitive_loop,
                name="DeepSeek-Memory-Management",
                daemon=True
            )
            self.thinking_thread.start()
            logging.info("Started autonomous memory management thread")
            return True
        else:
            logging.info("Autonomous memory management thread already running")
            return False
    
    def stop_cognitive_thread(self):
        """Signal the cognitive thread to stop."""
        if self.thinking_thread and self.thinking_thread.is_alive():
            self.stop_flag.set()
            logging.info("Signaled autonomous cognitive thread to stop")
            return True
        return False
    
    def update_user_activity(self):
        """Update the timestamp of the last user activity."""
        self.last_user_activity = time.time()
        logging.debug("Updated user activity timestamp")
    
    def _is_user_inactive(self, inactivity_threshold=10800): #must be idle for 3 hours then starts
        """
        Check if the user has been inactive for a sufficient period.

        Args:
            inactivity_threshold (int): Seconds of inactivity to consider user inactive (default: 3 hours)
            
        Returns:
            bool: True if user is inactive, False otherwise
        """
        time_since_activity = time.time() - self.last_user_activity
        return time_since_activity > inactivity_threshold
    
    def _cognitive_loop(self):
        """Main cognitive loop for autonomous memory management."""
        logging.info("Autonomous memory management loop started")
        
        # Initial wait period to allow system to stabilize
        time.sleep(60)
        
        while not self.stop_flag.is_set():
            try:
                # CHECK 1: Verify autonomous thinking is enabled in settings
                if hasattr(self.chatbot, 'autonomous_thinking_enabled') and not self.chatbot.autonomous_thinking_enabled:
                    logging.info("Autonomous thinking disabled in settings, pausing cognitive loop")
                    time.sleep(60)  # Check again after a minute
                    continue
                
                # CHECK 2: Verify user is inactive (3+ hours of no activity)
                if self._is_user_inactive():
                    logging.info("User inactive for 3+ hours, checking if autonomous processing is safe...")

                    # CHECK 3: CRITICAL FIX - Verify LLM is NOT currently generating
                    if hasattr(self.chatbot, '_llm_generating') and self.chatbot._llm_generating:
                        logging.warning("⚠️ LLM is currently generating a response - BLOCKING autonomous processing")
                        logging.info("Will re-check in 5 minutes...")
                        time.sleep(300)  # Wait 5 minutes before checking again
                        continue
                    
                    # CHECK 4: Verify no active conversation in progress
                    if hasattr(self.chatbot, 'conversation_in_progress') and self.chatbot.conversation_in_progress:
                        logging.warning("⚠️ Conversation in progress detected - BLOCKING autonomous processing")
                        logging.info("Will re-check in 5 minutes...")
                        time.sleep(300)  # Wait 5 minutes before checking again
                        continue
                    
                    # CHECK 5: Additional safety - check if there's been recent user input
                    # (in case flags weren't properly cleared)
                    time_since_activity = time.time() - self.last_user_activity
                    if time_since_activity < 10800:  # Less than 3 hours (10800 seconds)
                        logging.info(f"Recent user activity detected ({time_since_activity/60:.1f} minutes ago) - waiting longer")
                        time.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # ALL CHECKS PASSED - Safe to proceed with autonomous processing
                    logging.info("✅ All safety checks passed - proceeding with autonomous memory management")
                    
                    # Select a cognitive activity based on weights and last run time
                    activity = self._select_next_cognitive_activity()
                    
                    if activity:
                        logging.info(f"Selected memory management activity: {activity}")
                        self.cognitive_state = activity
                        
                        # Execute the selected cognitive activity
                        if hasattr(self, f"_{activity}"):
                            method = getattr(self, f"_{activity}")
                            method()
                            
                            # Update last run time for this activity
                            self.cognitive_activities[activity]["last_run"] = time.time()
                        else:
                            logging.error(f"No method found for cognitive activity: {activity}")
                    
                    # Reset cognitive state to idle
                    self.cognitive_state = "idle"
                    
                    # Wait between cognitive activities
                    time.sleep(self.cognitive_cycle_interval)
                else:
                    # User is active, pause autonomous processing
                    logging.debug("User active, pausing autonomous processing")
                    time.sleep(300)  # Check again after 5 minutes
            
            except Exception as e:
                logging.error(f"Error in cognitive loop: {e}", exc_info=True)
                self.cognitive_state = "error"
                time.sleep(300)  # Recovery pause after error (5 minutes)
        
        logging.info("Autonomous memory management loop stopped")
          
    def _update_activity_timestamp(self, activity_name):
        """
        Update the last run timestamp for an activity.
        
        Args:
            activity_name (str): Name of the activity
        """
        if activity_name in self.cognitive_activities:
            self.cognitive_activities[activity_name]["last_run"] = time.time()
            logging.debug(f"Updated last run timestamp for activity: {activity_name}")
        else:
            logging.warning(f"Cannot update timestamp for unknown activity: {activity_name}")

    def _analyze_knowledge_gaps(self):
        """
        Analyze existing knowledge to identify gaps related to the user and their needs.
        Enhanced with duplicate prevention and gap limiting - NOW FOCUSES ON SINGLE MOST IMPORTANT GAP.
        """
        print("🧠 ====== STARTING ENHANCED KNOWLEDGE GAP ANALYSIS ======")
        logging.info("====== STARTING ENHANCED KNOWLEDGE GAP ANALYSIS ======")
        self.cognitive_state = "analyzing"
        
        try:
            # Record the start of analysis
            print("📝 Step 1: Recording thought for knowledge analysis")
            logging.info("Step 1: Recording thought for knowledge analysis")
            self._record_thought(
                thought_type="knowledge_analysis", 
                content="Beginning enhanced analysis of knowledge gaps with single-gap focus and duplicate prevention."
            )
            
            # Get recent user queries and memories
            print("📊 Step 2: Getting recent user queries")
            logging.info("Step 2: Getting recent user queries")
            recent_queries = self._get_recent_queries()
            query_count = len(recent_queries.splitlines()) if recent_queries else 0
            print(f"   Found {query_count} recent queries")
            logging.info(f"Found queries: {query_count} recent queries")
            
            print("💾 Step 3: Getting relevant memories for analysis")
            logging.info("Step 3: Getting relevant memories for analysis")
            relevant_memories = self._get_relevant_memories_for_analysis()
            memory_count = len(relevant_memories.splitlines()) if relevant_memories else 0
            print(f"   Found {memory_count} relevant memory segments")
            logging.info(f"Found memories: {memory_count} relevant memory segments")
            
            if not recent_queries and not relevant_memories:
                print("⚠️ Insufficient data for knowledge gap analysis")
                logging.info("Insufficient data for knowledge gap analysis")
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content="Insufficient recent data to identify meaningful knowledge gaps."
                )
                return False
            
            # NEW: Get existing pending gaps for duplicate checking
            print("🔍 Step 3.5: Loading existing gaps for duplicate prevention")
            logging.info("Step 3.5: Loading existing gaps for duplicate prevention")
            try:
                from knowledge_gap import KnowledgeGapQueue
                gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
                existing_gaps = gap_queue.get_gaps_by_status('pending')
                print(f"   Found {len(existing_gaps)} existing pending gaps")
                logging.info(f"Found {len(existing_gaps)} existing pending gaps for duplicate checking")
            except Exception as e:
                print(f"   ⚠️ Could not load existing gaps: {e}")
                logging.warning(f"Could not load existing gaps for duplicate checking: {e}")
                existing_gaps = []
                
            # NEW: Get recent conversation context to check for answered questions
            recent_conversation_text = ""
            if hasattr(self.chatbot, 'current_conversation') and self.chatbot.current_conversation:
                # Get last 20 messages to check for answered topics
                recent_messages = self.chatbot.current_conversation[-20:]
                conversation_excerpts = []
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if content:
                        conversation_excerpts.append(f"[{role}]: {content[:200]}...")
                recent_conversation_text = "\n".join(conversation_excerpts)
            else:
                recent_conversation_text = "No recent conversation available"

            logging.info(f"Included {len(recent_conversation_text)} characters of conversation context")

            # Create simplified analysis prompt - UPDATED TO REQUEST ONLY 1 GAP
            print("🛠️ Step 4: Constructing enhanced prompt for single gap identification")
            logging.info("Step 4: Constructing enhanced prompt for single gap identification")

            prompt = f""" /no_think
I will analyze recent queries and stored information to identify ONLY THE SINGLE MOST IMPORTANT knowledge gap.

⚠️ CRITICAL ANTI-DUPLICATION RULES ⚠️

1. **Check Conversation History First**
- Review the recent conversations below
- If Ken has ALREADY answered or discussed a topic, it is NOT a knowledge gap
- DO NOT create gaps about topics Ken has explained, even if my stored memories are incomplete

2. **Compare Against Existing Gaps**
- Review ALL existing pending gaps listed below
- If a gap is similar in ANY way (same concept, related topic, overlapping question), DO NOT create it
- Examples of duplicates to AVOID:
    * "Ken's field of work" vs "Ken's professional domain" = DUPLICATE
    * "Ken's hobbies" vs "Ken's leisure activities" = DUPLICATE
    * "Ken's family structure" vs "Ken's relatives" = DUPLICATE

3. **Semantic Similarity Check**
- Before proposing a gap, ask: "Is this essentially asking the same thing as an existing gap?"
- If 70% of the information would overlap, it's a DUPLICATE
- Different wording does NOT make it a new gap

4. **Quality Over Quantity - SINGLE GAP FOCUS**
- I will identify ONLY THE SINGLE MOST IMPORTANT gap
- The gap must be:
    * Truly unknown (not discussed in conversations)
    * Completely different from existing gaps (no overlap)
    * Specific and actionable (not vague or general)
    * Important for helping Ken (not just interesting trivia)
- If no truly valuable gap exists, I will return NONE

5. **Classification Accuracy - CRITICAL RULES**

⚠️ PERSONAL_ASK_KEN - Use for ALL of these:
- ANY information about "Ken Bajema" personally (profession, background, work, education, career)
- Ken's preferences, opinions, feelings, or personal context
- Information about Ken's family (Ananda, Izabel, Lucian)
- Ken's location, home, garden, daily routines
- Ken's projects (QWEN, autonomous AI work)
- REASON: There are MULTIPLE Ken Bajemas online. Web searches will return WRONG PEOPLE.
- ALWAYS ask Ken directly for ANY information about himself.

🌐 PUBLIC_SEARCHABLE - Use ONLY for:
- General factual knowledge NOT about Ken personally
- Technology concepts, scientific principles, historical facts
- Programming techniques, API documentation, technical standards
- World events, public figures (other than Ken), general information
- Examples: "What is vector database indexing?", "Python async patterns", "History of neural networks"

🧠 SYSTEM_INTERNAL - Use for:
- My own capabilities, reasoning patterns, or self-improvement
- How I process information or make decisions

⚠️ NEVER classify gaps about "Ken Bajema", "Ken's profession", "Ken's background", 
"Ken's work", or ANY personal information about Ken as PUBLIC_SEARCHABLE.
Web searches will return information about OTHER Ken Bajemas, not our user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 RECENT CONVERSATIONS (check if topics were already discussed):
{recent_conversation_text[:1500] if recent_conversation_text else "No recent conversations"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 RECENT QUERIES:
{recent_queries[:1000] if recent_queries else "No recent queries"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 RELEVANT STORED KNOWLEDGE:
{relevant_memories[:1500] if relevant_memories else "No relevant memories"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 EXISTING PENDING GAPS (DO NOT DUPLICATE THESE):
{self._format_existing_gaps_for_prompt(existing_gaps)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ BEFORE PROPOSING ANY GAP, I MUST:
1. ✓ Verify it was NOT discussed in recent conversations
2. ✓ Confirm it does NOT overlap with existing pending gaps
3. ✓ Ensure it is NOT semantically similar to any existing gap
4. ✓ Verify it is truly unknown (not answered by Ken already)
5. ✓ If about Ken Bajema → MUST be PERSONAL_ASK_KEN (never PUBLIC_SEARCHABLE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I will identify ONLY THE SINGLE MOST IMPORTANT knowledge gap. Quality over quantity.

MAXIMUM 1 GAP ONLY. Format:

GAP 1:
TOPIC: [the most important unique topic - must be completely different from existing gaps]
CLASSIFICATION: [PUBLIC_SEARCHABLE|PERSONAL_ASK_KEN|SYSTEM_INTERNAL]
DESCRIPTION: [specific unknown information - explain why this is truly a gap]
PRIORITY: [HIGH/MEDIUM/LOW]
UNIQUENESS_CHECK: [explain how this differs from existing gaps and conversation topics]

⚠️ REMEMBER: If the gap is about Ken Bajema in ANY way, classification MUST be PERSONAL_ASK_KEN.
⚠️ I will focus on identifying the SINGLE MOST VALUABLE gap. If no truly unique and important gap exists, I will return NONE rather than proposing a low-quality gap.
"""

            
            # Get knowledge gap analysis from LLM
            print("🤖 Step 5: Calling LLM to identify the single most important knowledge gap")
            logging.info("Step 5: Calling LLM to identify single most important knowledge gap")
            try:
                raw_gaps = self._safe_llm_invoke(prompt)
                
                if not raw_gaps:
                    print("❌ LLM returned empty response")
                    logging.warning("LLM returned empty response for knowledge gap analysis")
                    return False
                
                print(f"✅ LLM response received, length: {len(raw_gaps)} characters")
                logging.info(f"LLM response received, length: {len(raw_gaps)}")
                
                # Parse the structured response
                print("📋 Step 6: Parsing structured text response from LLM")
                logging.info("Step 6: Parsing structured text response from LLM")
                knowledge_gaps = self._parse_knowledge_gaps_response(raw_gaps)
                
                if not knowledge_gaps:
                    print("❌ No knowledge gaps could be parsed from LLM response")
                    print(f"📄 Full LLM response was: {raw_gaps[:500]}...")
                    logging.warning("No knowledge gaps could be parsed from LLM response")
                    return False
                    
                print(f"✅ Successfully parsed {len(knowledge_gaps)} knowledge gap(s)")
                logging.info(f"Successfully parsed {len(knowledge_gaps)} knowledge gap(s)")
                
                # NEW: Step 7 - Enhanced duplicate checking and filtering
                print("🔍 Step 7: Enhanced duplicate checking and filtering")
                logging.info("Step 7: Enhanced duplicate checking and filtering")
                
                filtered_gaps = []
                duplicate_count = 0
                
                for i, gap in enumerate(knowledge_gaps):
                    topic = gap.get('topic', 'Unknown')
                    description = gap.get('description', 'No description')
                    priority = gap.get('priority', 'MEDIUM')
                    
                    print(f"   🔎 Checking gap {i+1}: {topic}")
                    
                    # Check for duplicates using enhanced similarity checking
                    if self._check_for_similar_gaps(topic, description, existing_gaps):
                        duplicate_count += 1
                        print(f"      ❌ Duplicate detected, skipping: {topic}")
                        logging.info(f"Skipped duplicate gap: {topic}")
                        continue
                    
                    # Check for duplicates within current batch
                    is_internal_duplicate = False
                    for existing_gap in filtered_gaps:
                        if self._gaps_are_similar(gap, existing_gap):
                            is_internal_duplicate = True
                            duplicate_count += 1
                            print(f"      ❌ Internal duplicate detected, skipping: {topic}")
                            logging.info(f"Skipped internal duplicate gap: {topic}")
                            break
                    
                    if not is_internal_duplicate:
                        filtered_gaps.append(gap)
                        print(f"      ✅ Unique gap approved: {topic}")
                        logging.info(f"Approved unique gap: {topic}")
                
                print(f"📊 DUPLICATE FILTERING RESULTS:")
                print(f"   📥 Initial gaps identified: {len(knowledge_gaps)}")
                print(f"   ❌ Duplicates filtered out: {duplicate_count}")
                print(f"   ✅ Unique gaps remaining: {len(filtered_gaps)}")
                
                # NEW: Limit to single highest-priority gap
                if len(filtered_gaps) > 1:
                    # Sort by priority and keep only the highest
                    priority_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    filtered_gaps.sort(key=lambda g: priority_map.get(g.get('priority', 'MEDIUM'), 2), reverse=True)
                    original_count = len(filtered_gaps)
                    filtered_gaps = filtered_gaps[:1]
                    print(f"   🎯 Limited to single highest-priority gap (reduced from {original_count} to 1)")
                    logging.info(f"Limited to single highest-priority gap for focused processing (reduced from {original_count})")
                
                # Use filtered gaps
                knowledge_gaps = filtered_gaps
                
                if not knowledge_gaps:
                    print("❌ No unique knowledge gaps remaining after duplicate filtering and prioritization")
                    logging.warning("No unique knowledge gaps remaining after duplicate filtering and prioritization")
                    return False
                
                # Enhanced logging to verify what we found
                print(f"🎯 ENHANCED KNOWLEDGE GAP ANALYSIS RESULTS:")
                print(f"   📊 Found {len(knowledge_gaps)} unique knowledge gap (SINGLE-GAP FOCUS)")
                
                for i, gap in enumerate(knowledge_gaps, 1):
                    topic = gap.get('topic', 'Unknown')
                    description = gap.get('description', 'No description')
                    priority = gap.get('priority', 'MEDIUM')
                    classification = gap.get('classification', 'PUBLIC_SEARCHABLE')
                    
                    print(f"   Gap {i}: {topic}")
                    print(f"      Classification: {classification}")
                    print(f"      Priority: {priority}")
                    print(f"      Description: {description[:100]}...")
                    
                    # Also log to file
                    logging.info(f"Gap {i}: {topic} (Classification: {classification}, Priority: {priority})")
                    logging.info(f"   Description: {description}")
                
                # Record the gaps identified
                gap_summary = "\n".join([f"- {gap['topic']} ({gap.get('classification', 'UNKNOWN')}): {gap['description'][:100]}..." 
                                    for gap in knowledge_gaps])
                                    
                logging.info(f"Enhanced knowledge gaps identified (SINGLE-GAP FOCUS):\n{gap_summary}")
                
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content=f"Enhanced analysis identified {len(knowledge_gaps)} unique gap (filtered {duplicate_count} duplicates, SINGLE-GAP FOCUS):\n{gap_summary}"
                )
                
                # Process the unique gaps using classification
                if knowledge_gaps:
                    print(f"🗃️ Step 8: PROCESSING {len(knowledge_gaps)} CLASSIFIED KNOWLEDGE GAP (SINGLE-GAP FOCUS)...")
                    logging.info("Step 8: Processing classified knowledge gap (SINGLE-GAP FOCUS)")
                    
                    # Create gaps analysis structure for processing
                    gaps_analysis = {"gaps": knowledge_gaps}
                    
                    # Use the new classification-based processing with parsed gaps
                    success = self._process_classified_gaps(gaps_analysis)

                    if success:
            
                        print("✅ ====== ENHANCED KNOWLEDGE GAP ANALYSIS COMPLETED SUCCESSFULLY ======")
                        logging.info("====== ENHANCED KNOWLEDGE GAP ANALYSIS COMPLETED SUCCESSFULLY ======")
                        return True
                    else:
                        print("⚠️ Gap processing completed with some issues")
                        logging.warning("Gap processing completed with some issues")
                        return False
                        
            except Exception as e:
                print(f"❌ Error in enhanced LLM knowledge gap analysis: {e}")
                logging.error(f"Error in enhanced LLM knowledge gap analysis: {e}", exc_info=True)
                self._record_thought(
                    thought_type="knowledge_analysis",
                    content=f"Error in enhanced analysis: {str(e)}"
                )
                return False
                        
        except Exception as e:
            print(f"❌ Error in enhanced knowledge gap analysis: {e}")
            logging.error(f"Error in enhanced knowledge gap analysis: {e}", exc_info=True)
            self._record_thought(
                thought_type="error",
                content=f"Error during enhanced knowledge gap analysis: {str(e)}"
            )
            return False
        finally:
            # Update activity timestamp
            self._update_activity_timestamp('analyze_knowledge_gaps')
            self.cognitive_state = "idle"
            print("🔄 Completed enhanced knowledge gap analysis (SINGLE-GAP FOCUS)")
            logging.info("Completed enhanced knowledge gap analysis (SINGLE-GAP FOCUS)")
    
    def _process_classified_gaps(self, gaps_analysis):
        """
        Process gaps based on their classification with comprehensive logging and validation.
        Enhanced with strict duplicate checking and conversation history validation.
        FIXED: Personal gaps now properly create gap_id before marking as fulfilled.
        
        Args:
            gaps_analysis (dict or str): Either a dictionary with 'gaps' key containing parsed gaps,
                                        or a string to be parsed
            
        Returns:
            bool: Success status
        """
        logging.info("📋 Starting classification-based gap processing with enhanced validation")
        
        try:
            from knowledge_gap import KnowledgeGapQueue
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            
            # CRITICAL FIX: Handle dictionary vs string input
            if isinstance(gaps_analysis, dict):
                # Extract the already-parsed gaps from the dictionary
                parsed_gaps = gaps_analysis.get('gaps', [])
                logging.info(f"🔍 Extracted {len(parsed_gaps)} gaps from dictionary structure")
            elif isinstance(gaps_analysis, str):
                # Parse from string if that's what was passed
                logging.info("🔍 Parsing knowledge gaps from LLM analysis string")
                parsed_gaps = self._parse_knowledge_gaps_response(gaps_analysis)
            else:
                logging.error(f"❌ Unexpected gaps_analysis type: {type(gaps_analysis)}")
                return False
            
            if not parsed_gaps:
                logging.warning("⚠️ No gaps could be extracted from input")
                return False
            
            logging.info(f"✅ Successfully extracted {len(parsed_gaps)} knowledge gaps")
            
            # NEW: Get existing gaps and recent conversation for validation
            try:
                existing_gaps = gap_queue.get_gaps_by_status('pending')
                logging.info(f"📚 Loaded {len(existing_gaps)} existing pending gaps for validation")
            except Exception as e:
                logging.warning(f"Could not load existing gaps for validation: {e}")
                existing_gaps = []
            
            # NEW: Get recent conversation context
            recent_conversation_text = ""
            if hasattr(self.chatbot, 'current_conversation') and self.chatbot.current_conversation:
                recent_messages = self.chatbot.current_conversation[-20:]
                conversation_excerpts = []
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if content:
                        conversation_excerpts.append(f"[{role}]: {content[:200]}...")
                recent_conversation_text = "\n".join(conversation_excerpts)
                logging.info(f"📖 Loaded {len(recent_messages)} recent conversation messages for validation")
            else:
                logging.info("⚠️ No recent conversation available for validation")
            
            # NEW: Validate each gap before processing
            validated_gaps = []
            rejected_gaps = []
            
            logging.info(f"🔍 Starting validation of {len(parsed_gaps)} proposed gaps...")
            
            for i, gap in enumerate(parsed_gaps, 1):
                topic = gap.get('topic', 'Unknown')
                description = gap.get('description', 'No description')
                
                logging.info(f"🔎 Validating gap {i}/{len(parsed_gaps)}: {topic}")
                
                # Run uniqueness validation
                is_valid, reason = self._validate_gap_uniqueness(
                    gap, 
                    existing_gaps,
                    recent_conversation_text
                )
                
                if is_valid:
                    validated_gaps.append(gap)
                    logging.info(f"   ✅ VALIDATED: {topic}")
                    logging.debug(f"      Reason: {reason}")
                else:
                    rejected_gaps.append((gap, reason))
                    logging.warning(f"   ❌ REJECTED: {topic}")
                    logging.warning(f"      Reason: {reason}")
            
            # Log validation summary
            logging.info(f"")
            logging.info(f"📊 VALIDATION SUMMARY:")
            logging.info(f"   ✅ Validated gaps: {len(validated_gaps)}")
            logging.info(f"   ❌ Rejected gaps: {len(rejected_gaps)}")
            logging.info(f"   📈 Validation rate: {(len(validated_gaps)/len(parsed_gaps)*100):.1f}%")
            
            if rejected_gaps:
                logging.info(f"")
                logging.info(f"🚫 REJECTED GAPS DETAILS:")
                for gap, reason in rejected_gaps:
                    logging.info(f"   • {gap.get('topic')}: {reason}")
            
            # Check if we have any validated gaps to process
            if not validated_gaps:
                logging.warning("⚠️ No gaps passed validation - all were duplicates or already discussed")
                return False
            
            logging.info(f"")
            logging.info(f"▶️  Processing {len(validated_gaps)} validated gaps...")
            
            # Initialize counters
            personal_reminders_created = 0
            system_internal_queued = 0
            public_searchable_queued = 0
            processing_errors = 0
            
            # Process each VALIDATED gap according to its classification
            for i, gap in enumerate(validated_gaps, 1):
                topic = gap.get('topic', '').strip()
                description = gap.get('description', '').strip()
                classification = gap.get('classification', 'PUBLIC_SEARCHABLE').strip().upper()
                priority = gap.get('priority', 'MEDIUM').strip().upper()
                
                logging.info(f"")
                logging.info(f"🔄 Processing validated gap {i}/{len(validated_gaps)}: {topic}")
                logging.debug(f"   Classification: {classification}")
                logging.debug(f"   Priority: {priority}")
                logging.debug(f"   Description: {description[:100]}...")
                
                # Validate required fields
                if not topic or not description:
                    logging.warning(f"⚠️ Gap {i} missing required fields - Topic: '{topic}', Description: '{description}'")
                    processing_errors += 1
                    continue
                
                # Convert priority to numeric value
                priority_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
                priority_value = priority_map.get(priority, 0.6)
                
                try:
                    if classification == 'PERSONAL_ASK_KEN':
                        logging.info(f"👤 Processing personal knowledge gap: {topic}")
                        
                        # CRITICAL FIX - STEP 1: First, queue the gap in knowledge_gaps table to get gap_id
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        if gap_id <= 0:
                            logging.error(f"   ❌ Failed to create knowledge gap entry for: {topic}")
                            processing_errors += 1
                            continue  # Skip to next gap
                        
                        logging.info(f"   ✅ Created knowledge gap entry with ID {gap_id}")
                        
                        # STEP 2: Create reminder for this gap
                        due_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                        reminder_text = f"Clarify knowledge gap: {topic} - {description}"
                        
                        reminder_success = self._create_reminder_for_personal_gap(reminder_text, due_date)
                        
                        if reminder_success:
                            personal_reminders_created += 1
                            logging.info(f"   ✅ Successfully created reminder for: {topic}")
                            
                            # STEP 3: CRITICAL FIX - Mark gap as fulfilled immediately after reminder creation
                            # The reminder IS the fulfillment mechanism for personal gaps
                            try:
                                gap_queue.mark_fulfilled(gap_id)  # ✅ No count needed for personal gaps
                                logging.info(f"   🔗 Marked knowledge gap {gap_id} as 'fulfilled' (reminder created)")
                                logging.info(f"   📋 Ken will address this via the reminder system")
                            except Exception as mark_error:
                                logging.error(f"   ⚠️ Failed to mark gap {gap_id} as fulfilled: {mark_error}")
                                # Don't fail the whole process - reminder was still created successfully
                                
                        else:
                            logging.error(f"   ❌ Failed to create reminder for: {topic}")
                            # Gap was created but reminder failed - leave gap as 'pending' for retry
                            logging.warning(f"   ⚠️ Gap {gap_id} remains 'pending' since reminder creation failed")
                            processing_errors += 1
                    
                    elif classification == 'SYSTEM_INTERNAL':
                        logging.info(f"🧠 Queueing system internal gap: {topic}")
                        
                        # Queue for self-reflection with special prefix
                        internal_topic = f"SELF_REFLECTION: {topic}"
                        gap_id = gap_queue.add_gap(internal_topic, description, priority_value)
                        
                        if gap_id > 0:
                            system_internal_queued += 1
                            logging.info(f"   ✅ Queued system internal gap with ID {gap_id}: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue system internal gap: {topic}")
                            processing_errors += 1
                    
                    elif classification in ['PUBLIC_SEARCHABLE', 'FACTUAL_GENERAL']:
                        logging.info(f"🌐 Queueing public searchable gap: {topic}")
                        
                        # Regular web search queue
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        if gap_id > 0:
                            public_searchable_queued += 1
                            logging.info(f"   ✅ Queued public searchable gap with ID {gap_id}: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue public searchable gap: {topic}")
                            processing_errors += 1
                    
                    else:
                        logging.warning(f"⚠️ Unknown classification '{classification}' for gap: {topic}")
                        logging.info(f"   🔄 Defaulting to PUBLIC_SEARCHABLE for: {topic}")
                        
                        # Default to public searchable
                        gap_id = gap_queue.add_gap(topic, description, priority_value)
                        
                        if gap_id > 0:
                            public_searchable_queued += 1
                            logging.info(f"   ✅ Queued gap (default classification) with ID {gap_id}: {topic}")
                        else:
                            logging.error(f"   ❌ Failed to queue gap (default): {topic}")
                            processing_errors += 1
                            
                except Exception as gap_error:
                    logging.error(f"❌ Error processing individual gap '{topic}': {gap_error}", exc_info=True)
                    processing_errors += 1
                    continue
            
            # Final summary logging
            total_processed = personal_reminders_created + system_internal_queued + public_searchable_queued
            
            logging.info(f"")
            logging.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logging.info(f"📊 FINAL GAP PROCESSING SUMMARY:")
            logging.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logging.info(f"   📥 Gaps proposed by LLM: {len(parsed_gaps)}")
            logging.info(f"   ✅ Gaps passed validation: {len(validated_gaps)}")
            logging.info(f"   ❌ Gaps rejected (duplicates): {len(rejected_gaps)}")
            logging.info(f"")
            logging.info(f"   👤 Personal reminders created: {personal_reminders_created}")
            logging.info(f"   🧠 System internal gaps queued: {system_internal_queued}")
            logging.info(f"   🌐 Public searchable gaps queued: {public_searchable_queued}")
            logging.info(f"   ❌ Processing errors: {processing_errors}")
            logging.info(f"")
            logging.info(f"   ✅ Total successfully processed: {total_processed}/{len(validated_gaps)}")
            
            # Calculate success rates
            if len(parsed_gaps) > 0:
                validation_rate = (len(validated_gaps) / len(parsed_gaps)) * 100
                logging.info(f"   📈 Validation pass rate: {validation_rate:.1f}%")
            
            if len(validated_gaps) > 0:
                processing_rate = (total_processed / len(validated_gaps)) * 100
                logging.info(f"   📈 Processing success rate: {processing_rate:.1f}%")
            
            logging.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Determine success status based on validation AND processing
            if len(validated_gaps) == 0:
                logging.warning(f"⚠️ Gap processing completed but all gaps were rejected as duplicates")
                # This is actually SUCCESS - we prevented duplicates!
                return True
            
            success_rate = total_processed / len(validated_gaps) if len(validated_gaps) > 0 else 0
            
            if success_rate >= 0.8:  # 80% success rate
                logging.info(f"✅ Gap processing completed successfully (success rate: {success_rate:.1%})")
                return True
            elif success_rate >= 0.5:  # 50% success rate
                logging.warning(f"⚠️ Gap processing completed with warnings (success rate: {success_rate:.1%})")
                return True
            else:
                logging.error(f"❌ Gap processing failed (success rate: {success_rate:.1%})")
                return False
            
        except Exception as e:
            logging.error(f"❌ Critical error in gap processing: {e}", exc_info=True)
            return False
        
    def _create_reminder_for_personal_gap(self, reminder_text, due_date):
        """
        Create a reminder using the proper REMINDER command syntax.
        Enhanced with duplicate prevention - checks if topic was recently discussed.
        
        Args:
            reminder_text (str): The reminder content
            due_date (str): Due date in YYYY-MM-DD format
            
        Returns:
            bool: Success status
        """
        try:
            logging.debug(f"Creating reminder: {reminder_text[:50]}... due {due_date}")
            
            # NEW: Quick check if this was recently discussed
            topic_keywords = set(word for word in reminder_text.lower().split() if len(word) > 4)
            
            if hasattr(self.chatbot, 'current_conversation') and self.chatbot.current_conversation:
                recent_text = ' '.join([
                    msg.get('content', '') for msg in self.chatbot.current_conversation[-10:]
                ]).lower()
                
                # If 60%+ of topic keywords appear in recent conversation, skip reminder
                if topic_keywords:
                    words_found = sum(1 for word in topic_keywords if word in recent_text)
                    overlap_ratio = words_found / len(topic_keywords)
                    
                    if overlap_ratio > 0.6:
                        logging.info(f"✓ Skipping reminder - topic appears answered in recent conversation")
                        logging.info(f"  Topic: {reminder_text[:80]}...")
                        logging.info(f"  Overlap: {overlap_ratio:.1%} of keywords found in recent messages")
                        # Return True because we successfully prevented a duplicate
                        return True
            
            # Try using the existing reminder system first
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, '_handle_reminder_command'):
                try:
                    # Format as the reminder system expects: "content | due=YYYY-MM-DD"
                    reminder_command_text = f"{reminder_text} | due={due_date}"
                    result, success = self.chatbot.deepseek_enhancer._handle_reminder_command(reminder_command_text)
                    
                    if success:
                        logging.debug(f"✅ Reminder created via reminder system: {result}")
                        return True
                    else:
                        logging.warning(f"⚠️ Reminder system failed: {result}")
                        
                except Exception as reminder_error:
                    logging.warning(f"⚠️ Error using reminder system: {reminder_error}")
            
            # Fallback: store as a special memory type
            logging.info("🔄 Using fallback memory storage for reminder")
            
            metadata = {
                "type": "knowledge_gap_reminder",
                "reminder_text": reminder_text,
                "due_date": due_date,
                "created_by": "autonomous_cognition",
                "created_at": datetime.datetime.now().isoformat(),
                "tags": "knowledge_gap,personal,reminder,autonomous"
            }
            
            full_content = f"REMINDER: {reminder_text}\nDue: {due_date}\n\nThis reminder was created by autonomous gap analysis for personal information that requires Ken's input."
            
            # Use transaction coordination for consistent storage
            success, memory_id = self.chatbot.store_memory_with_transaction(
                content=full_content,
                memory_type="knowledge_gap_reminder",
                metadata=metadata,
                confidence=0.8
            )
            
            if success:
                logging.info(f"✅ Created reminder as memory with ID {memory_id}")
                return True
            else:
                logging.error(f"❌ Failed to store reminder as memory")
                return False
            
        except Exception as e:
            logging.error(f"❌ Error creating reminder: {e}", exc_info=True)
            return False        
    
    def _format_existing_gaps_for_prompt(self, existing_gaps):
        """Format existing gaps for the LLM prompt with enhanced detail to prevent duplicates."""
        if not existing_gaps:
            return "No existing pending gaps"
        
        # Limit to most recent 25 gaps (increased from 20)
        recent_gaps = existing_gaps[-25:] if len(existing_gaps) > 25 else existing_gaps
        
        formatted = []
        for i, gap in enumerate(recent_gaps, 1):
            topic = gap.get('topic', 'Unknown')
            description = gap.get('description', 'No description')
            
            # Extract key terms from topic and description for better matching
            combined_text = f"{topic} {description}".lower()
            key_terms = set(word for word in combined_text.split() if len(word) > 4)
            
            # Format with more detail
            formatted.append(
                f"{i}. TOPIC: {topic}\n"
                f"   DESCRIPTION: {description[:100]}...\n"
                f"   KEY_TERMS: {', '.join(list(key_terms)[:5])}"
            )
        
        return "\n\n".join(formatted)
    
    def _validate_gap_uniqueness(self, proposed_gap, existing_gaps, recent_conversation_text):
        """
        Validate that a proposed gap is truly unique and not already discussed.
        
        Args:
            proposed_gap (dict): The gap to validate with 'topic' and 'description' keys
            existing_gaps (list): List of existing gap dictionaries
            recent_conversation_text (str): Recent conversation history
            
        Returns:
            tuple: (is_valid, reason) where is_valid is bool and reason explains why
        """
        try:
            topic = proposed_gap.get('topic', '').lower()
            description = proposed_gap.get('description', '').lower()
            
            if not topic or not description:
                return False, "Gap missing required fields"
            
            # Check 1: Was this discussed in recent conversations?
            topic_words = set(word for word in topic.split() if len(word) > 3)
            conversation_lower = recent_conversation_text.lower()
            
            # If more than 60% of topic words appear in recent conversation
            if topic_words:
                words_found = sum(1 for word in topic_words if word in conversation_lower)
                match_ratio = words_found / len(topic_words)
                
                if match_ratio > 0.6:
                    return False, f"Topic '{topic}' appears to have been discussed in recent conversations ({match_ratio:.1%} word match)"
            
            # Check 2: Compare against existing gaps using enhanced similarity
            for existing_gap in existing_gaps:
                existing_topic = existing_gap.get('topic', '').lower()
                existing_desc = existing_gap.get('description', '').lower()
                
                # Skip empty existing gaps
                if not existing_topic:
                    continue
                
                # Topic similarity check
                topic_similarity = self._calculate_text_similarity(topic, existing_topic)
                if topic_similarity > 0.6:  # Using your lowered threshold
                    return False, f"Too similar to existing gap: '{existing_topic}' (similarity: {topic_similarity:.2f})"
                
                # Description similarity check
                desc_similarity = self._calculate_text_similarity(description, existing_desc)
                if desc_similarity > 0.5:  # Lower threshold for descriptions
                    return False, f"Description too similar to existing gap about '{existing_topic}' (similarity: {desc_similarity:.2f})"
                
                # Keyword overlap check
                topic_keywords = set(word for word in topic.split() if len(word) > 4)
                existing_keywords = set(word for word in existing_topic.split() if len(word) > 4)
                
                if topic_keywords and existing_keywords:
                    overlap = len(topic_keywords.intersection(existing_keywords))
                    total = len(topic_keywords.union(existing_keywords))
                    overlap_ratio = overlap / total if total > 0 else 0
                    
                    if overlap_ratio > 0.5:  # 50% keyword overlap
                        overlapping_words = topic_keywords.intersection(existing_keywords)
                        return False, f"High keyword overlap ({overlap_ratio:.1%}) with existing gap: '{existing_topic}' (shared: {', '.join(list(overlapping_words)[:3])})"
            
            # Check 3: Validate within the current batch (avoid duplicates in same analysis)
            # This is handled in the calling code, but we can add a note
            logging.debug(f"   Gap '{topic}' passed all uniqueness checks")
            
            # If we get here, gap is valid
            return True, "Gap is unique and not previously discussed"
            
        except Exception as e:
            logging.error(f"Error validating gap uniqueness: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

    def _check_for_similar_gaps(self, new_topic, new_description, existing_gaps):
        """
        Enhanced similarity checking for gaps using multiple similarity metrics.
        Uses a two-stage approach:
        1. Fast Jaccard text similarity (catches obvious duplicates)
        2. Semantic vector similarity via Qdrant (catches subtle duplicates)
        
        Args:
            new_topic (str): Topic of the new gap
            new_description (str): Description of the new gap  
            existing_gaps (list): List of existing gap dictionaries
            
        Returns:
            bool: True if a similar gap exists, False otherwise
        """
        if not new_topic:
            return False
            
        try:
            new_topic_lower = new_topic.lower()
            new_desc_lower = (new_description or '').lower()
            
            # =====================================================
            # STAGE 1: Fast Jaccard text similarity (pre-filter)
            # Catches obvious duplicates without API calls
            # =====================================================
            
            if existing_gaps:
                for existing_gap in existing_gaps:
                    existing_topic = existing_gap.get('topic', '').lower()
                    existing_desc = existing_gap.get('description', '').lower()
                    
                    # Check topic similarity - high threshold for topics
                    topic_similarity = self._calculate_text_similarity(new_topic_lower, existing_topic)
                    if topic_similarity > 0.6:  # 60% similarity threshold for topics
                        logging.info(f"🔍 Jaccard duplicate detected: '{new_topic}' vs '{existing_topic}' "
                                   f"(similarity: {topic_similarity:.2%})")
                        return True
                    
                    # Check description similarity - lower threshold
                    desc_similarity = self._calculate_text_similarity(new_desc_lower, existing_desc)
                    if desc_similarity > 0.7:  # 70% similarity threshold for descriptions
                        logging.info(f"🔍 Jaccard description duplicate detected (similarity: {desc_similarity:.2%})")
                        return True
                    
                    # Check for keyword overlap in topics
                    new_keywords = set(new_topic_lower.split())
                    existing_keywords = set(existing_topic.split())
                    if new_keywords and existing_keywords:
                        keyword_overlap = len(new_keywords.intersection(existing_keywords)) / len(new_keywords.union(existing_keywords))
                        if keyword_overlap > 0.6:  # 60% keyword overlap
                            logging.info(f"🔍 Jaccard keyword overlap detected: {keyword_overlap:.2%}")
                            return True
            
            # =====================================================
            # STAGE 2: Semantic vector similarity via Qdrant
            # Catches conceptually similar gaps with different wording
            # =====================================================
            
            try:
                from knowledge_gap import KnowledgeGapQueue
                
                # Initialize gap queue with Qdrant connection
                gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
                
                # Check semantic similarity using vector embeddings
                is_semantic_duplicate, similar_info = gap_queue.check_semantic_similarity(
                    new_topic, 
                    new_description or ''
                )
                
                if is_semantic_duplicate:
                    logging.info(f"🧠 Semantic duplicate detected: '{new_topic}' similar to "
                               f"'{similar_info.get('topic', 'Unknown')}' "
                               f"(similarity: {similar_info.get('score', 0):.2%})")
                    return True
                    
            except ImportError as e:
                logging.warning(f"Could not import KnowledgeGapQueue for semantic check: {e}")
            except Exception as e:
                logging.warning(f"Semantic similarity check failed (continuing with Jaccard only): {e}")
            
            # No duplicates found by either method
            logging.debug(f"✅ No duplicates found for '{new_topic}'")
            return False
            
        except Exception as e:
            logging.error(f"Error checking gap similarity: {e}")
            return False

    def _gaps_are_similar(self, gap1, gap2):
        """Check if two gaps from the current batch are similar."""
        try:
            topic1 = gap1.get('topic', '').lower()
            topic2 = gap2.get('topic', '').lower()
            desc1 = gap1.get('description', '').lower()
            desc2 = gap2.get('description', '').lower()
            
            # Check topic similarity
            topic_sim = self._calculate_text_similarity(topic1, topic2)
            desc_sim = self._calculate_text_similarity(desc1, desc2)
            
            return topic_sim > 0.8 or desc_sim > 0.7
            
        except Exception as e:
            logging.error(f"Error comparing gaps: {e}")
            return False

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings using Jaccard similarity.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            # Convert to sets of words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            logging.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _get_recent_queries(self):
        """Get recent user queries for knowledge gap analysis."""
        try:
            # Get recent conversation
            logging.info("Attempting to retrieve recent user queries")
            if hasattr(self.chatbot, 'current_conversation'):
                # Extract user messages from conversation
                logging.info(f"Found conversation with {len(self.chatbot.current_conversation)} messages")
                user_messages = [msg['content'] for msg in self.chatbot.current_conversation 
                            if msg.get('role') == 'user']
                
                logging.info(f"Extracted {len(user_messages)} user messages")
                
                # Return the last 5 messages or fewer if not available
                recent_queries = "\n".join(user_messages[-5:])
                return recent_queries
            else:
                logging.warning("No current_conversation attribute found in chatbot")
                return ""
        except Exception as e:
            logging.error(f"Error getting recent queries: {e}", exc_info=True)
            return ""
        
    def _get_relevant_memories_for_analysis(self):
        """Get relevant memories for knowledge gap analysis."""
        try:
            # Use self.memory_db.get_recent_memories() instead of self._get_memories_by_recency
            logging.info("Attempting to retrieve relevant memories")
            try:
                # Use the existing method in memory_db
                memories = self.memory_db.get_recent_memories(limit=20)  # Get a bit more for better context
                logging.info(f"Successfully retrieved {len(memories)} memories")
            except Exception as e:
                logging.error(f"Error retrieving recent memories: {e}")
                memories = []
            
            # If memories are already formatted strings, just join them
            if memories and isinstance(memories[0], str):
                return "\n\n".join(memories)
                
            # Otherwise, format them (unlikely to reach this part with your implementation)
            memory_texts = []
            for memory in memories:
                if isinstance(memory, dict):
                    content = memory.get('content', '')
                    memory_type = memory.get('memory_type', memory.get('metadata', {}).get('type', 'unknown'))
                    memory_texts.append(f"[{memory_type}] {content}")
                elif isinstance(memory, str):
                    memory_texts.append(memory)
            
            return "\n\n".join(memory_texts)
        except Exception as e:
            logging.error(f"Error getting relevant memories: {e}")
            return ""
    
    def _extract_and_queue_gaps(self, gap_strategies_text):
        """Extract and queue knowledge gaps from the analysis text."""
        try:
            if not gap_strategies_text:
                return "No gaps were identified."
                
            # Initialize the knowledge gap queue
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            
            # Parse the gap strategies text
            current_gap = {}
            queued_gaps = []
            current_section = None
            
            for line in gap_strategies_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                    
                # Check for start of new gap
                if line.upper().startswith("GAP TITLE:"):
                    # Save previous gap if exists
                    if current_gap and "title" in current_gap:
                        queued_gaps.append(current_gap)
                        
                    # Start new gap
                    current_gap = {"title": line[len("GAP TITLE:"):].strip()}
                    current_section = "title"
                    
                elif line.upper().startswith("DESCRIPTION:"):
                    current_gap["description"] = line[len("DESCRIPTION:"):].strip()
                    current_section = "description"
                    
                elif line.upper().startswith("confidence:"):
                    confidence_text = line[len("confidence:"):].strip().upper()
                    # Convert text confidence to numeric
                    if "HIGH" in confidence_text:
                        current_gap["priority"] = 0.9
                    elif "MEDIUM" in confidence_text:
                        current_gap["priority"] = 0.6
                    else:  # LOW or anything else
                        current_gap["priority"] = 0.3
                    current_section = "confidence"
                    
                elif line.upper().startswith("QUESTIONS:"):
                    current_gap["questions"] = line[len("QUESTIONS:"):].strip()
                    current_section = "questions"
                    
                # Continue previous section if not a new section header
                elif current_section and current_section in current_gap:
                    if current_section == "questions":
                        current_gap["questions"] += " " + line
                    elif current_section == "description":
                        current_gap["description"] += " " + line
            
            # Add the last gap if exists
            if current_gap and "title" in current_gap:
                queued_gaps.append(current_gap)
                
            # Queue the identified gaps
            for gap in queued_gaps:
                # Create description combining the description and questions
                description = gap.get("description", "")
                if "questions" in gap:
                    description += f"\n\nQuestions to investigate: {gap.get('questions')}"
                    
                # Add to queue
                gap_id = gap_queue.add_gap(
                    topic=gap.get("title", "Unknown Gap"),
                    description=description,
                    priority=gap.get("priority", 0.5)
                )
                
                if gap_id > 0:
                    logging.info(f"Queued knowledge gap: {gap.get('title')} with ID {gap_id}")
                    
            # Return summary of queued gaps
            if queued_gaps:
                summary = "Successfully queued the following knowledge gaps:\n"
                for gap in queued_gaps:
                    priority_text = "HIGH" if gap.get("priority", 0) > 0.7 else "MEDIUM" if gap.get("priority", 0) > 0.4 else "LOW"
                    summary += f"- {gap.get('title')} (Priority: {priority_text}, ID: {gap_id})\n"
                return summary
            else:
                return "No specific knowledge gaps were identified for queueing."
                
        except Exception as e:
            logging.error(f"Error extracting and queueing gaps: {e}")
            return f"Error processing knowledge gaps: {str(e)}"

    def _get_memory_type_distribution(self):
        """Get distribution of memory types as a formatted string."""
        try:
            # Execute SQL to get memory type distribution
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT memory_type, COUNT(*) as count
                    FROM memories
                    GROUP BY memory_type
                    ORDER BY count DESC
                """)
                distribution = cursor.fetchall()
            
            if not distribution:
                return "No memories found."
            
            # Format as readable text
            result = "Memory type distribution:\n"
            for memory_type, count in distribution:
                result += f"- {memory_type}: {count} memories\n"
            
            return result
        
        except Exception as e:
            logging.error(f"Error getting memory type distribution: {e}", exc_info=True)
            return "Error retrieving memory distribution."
        
    def _get_memories_by_recency(self, limit=30):
        """Get the most recent memories for analysis."""
        try:
            return self.memory_db.get_memories_by_recency(limit=limit)
        except Exception as e:
            logging.error(f"Error getting recent memories: {e}")
            return []

    def _analyze_memory_health(self):
        """Analyze the health of memory storage and database."""
        try:
            # Check database for potential issues
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for duplicate content
                cursor.execute("""
                    SELECT content, COUNT(*) as count
                    FROM memories
                    GROUP BY content
                    HAVING count > 1
                    LIMIT 10
                """)
                duplicates = cursor.fetchall()
                
                # Check for very old memories
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE julianday('now') - julianday(created_at) > 180
                """)
                old_memories = cursor.fetchone()[0]
                
                # Check for memories with empty content
                cursor.execute("""
                    SELECT COUNT(*) FROM memories
                    WHERE content IS NULL OR content = ''
                """)
                empty_memories = cursor.fetchone()[0]
                
                # Generate health report
                health_report = f"""
                Memory Storage Health Report:
                
                - Total memories: {self._count_memories()}
                - Duplicate memories: {len(duplicates)}
                - Very old memories (>180 days): {old_memories}
                - Empty memories: {empty_memories}
                """
                
                return health_report
        except Exception as e:
            logging.error(f"Error analyzing memory health: {e}")
            return "Error analyzing memory health"

    def _count_memories(self):
        """Count total memories in the database."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"Error counting memories: {e}")
            return 0

    def delete_memory_by_id(self, memory_id):
        """Delete a memory by its ID."""
        try:
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            logging.error(f"Error deleting memory by ID: {e}")
            return False
    
    
    def _get_memory_domains(self):
        """Get distinct memory domains from memory types and tags."""
        try:
            domains = set()
            
            # Get unique memory types
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT memory_type FROM memories")
                for row in cursor.fetchall():
                    if row[0]:
                        domains.add(row[0])
            
            # Get unique tags
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT tags FROM memories WHERE tags IS NOT NULL")
                for row in cursor.fetchall():
                    if row[0]:
                        # Split comma-separated tags
                        tags = [tag.strip() for tag in row[0].split(',')]
                        domains.update(tags)
            
            # Filter out technical or non-domain tags
            excluded_terms = {'null', 'none', 'general', 'important', 'autonomous', 'reflection'}
            domains = {domain for domain in domains if domain.lower() not in excluded_terms}
            
            return list(domains)
        
        except Exception as e:
            logging.error(f"Error getting memory domains: {e}", exc_info=True)
            return []
   
    
    def _record_thought(self, thought_type, content, metadata=None):
        """
        Records a memory management log in the standard logging system only.
        These are system status messages, not valuable content that needs database storage.
        
        Args:
            thought_type (str): Type of thought (e.g., 'analysis', 'organization')
            content (str): The actual content
            metadata (dict, optional): Additional metadata
        
        Returns:
            bool: Always returns True since we're just logging
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Create a thought record for in-memory tracking only
            thought_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': thought_type,
                'content': content,
                'metadata': metadata
            }
            
            # Store in thoughts list (in-memory only)
            if hasattr(self, 'thoughts') and self.thoughts is not None:
                self.thoughts.append(thought_record)
            else:
                self.thoughts = [thought_record]
            
            # Keep only recent thoughts in memory (limit to prevent memory bloat)
            if len(self.thoughts) > self.max_thought_history:
                self.thoughts = self.thoughts[-self.max_thought_history:]
            
            # Log to standard logging system with appropriate level
            if thought_type == "error":
                logging.error(f"Autonomous Cognition [{thought_type}]: {content}")
            elif thought_type == "warning":
                logging.warning(f"Autonomous Cognition [{thought_type}]: {content}")
            else:
                logging.info(f"Autonomous Cognition [{thought_type}]: {content}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in _record_thought: {e}")
            return True  # Don't fail the whole process for logging issues
    
        
    def _should_avoid_complex_llm_calls(self):
        """
        Determine if complex LLM calls should be avoided due to resource constraints.
        
        Returns:
            bool: True if complex LLM calls should be avoided, False otherwise
        """
        try:
            # Check if we're in a rate-limited state
            if hasattr(self, 'rate_limited') and self.rate_limited:
                return True
                
            # Check system timestamp - avoid complex calls during peak hours
            current_hour = datetime.datetime.now().hour
            # Avoid complex calls during likely system peak times (optional)
            if 8 <= current_hour <= 18:  # 8 AM to 6 PM
                # During peak hours, randomly avoid some complex calls
                # to reduce overall load (30% chance to avoid)
                if random.random() < 0.3:
                    logging.info("Avoiding complex LLM calls during peak hours")
                    return True
                    
            # Check error history - implement backoff if recent errors
            if hasattr(self, 'llm_error_count') and self.llm_error_count > 3:
                logging.info(f"Avoiding complex LLM calls due to error count: {self.llm_error_count}")
                return True
                
            # Default to allowing complex calls
            return False
            
        except Exception as e:
            logging.error(f"Error in _should_avoid_complex_llm_calls: {e}")
            # On error, be conservative and avoid complex calls
            return True

    def _safe_llm_invoke(self, prompt, max_retries=2, backoff_factor=2):
        """Safely invoke the LLM with retries and error handling.
        
        Args:
            prompt (str): The prompt to send to the LLM
            max_retries (int): Maximum number of retry attempts
            backoff_factor (int): Multiplier for exponential backoff
            
        Returns:
            str: The LLM response or empty string on failure
        """
        logging.info(f"🤖 Invoking LLM with prompt length: {len(prompt)} characters")
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                logging.info(f"   Attempt {retries + 1}/{max_retries + 1}")
                
                # Try with reduced parameters if we're retrying
                if retries > 0:
                    logging.info("   Using reduced parameters for retry")
                    # Use a smaller context size and simpler parameters on retry
                    response = self.chatbot.llm.invoke(
                        prompt, 
                        temperature=0.3,  # Lower temperature for more predictable output
                        num_predict=300   # Limit token generation
                    )
                else:
                    logging.info("   Using standard parameters")
                    response = self.chatbot.llm.invoke(prompt)
                
                if response:
                    response_length = len(response)
                    logging.info(f"✅ LLM responded successfully: {response_length} characters")
                    logging.debug(f"   Response preview: {response[:100]}...")
                    return response
                else:
                    logging.warning("   LLM returned empty response")
                    if retries < max_retries:
                        logging.info("   Will retry with different parameters")
                        
            except Exception as e:
                last_error = e
                logging.warning(f"   LLM invocation failed (attempt {retries+1}/{max_retries+1}): {str(e)}")
                
            retries += 1
            if retries <= max_retries:
                # Exponential backoff
                sleep_time = backoff_factor ** retries
                logging.info(f"   Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        # If we get here, all retries failed
        logging.error(f"❌ All LLM invocation attempts failed. Last error: {last_error}")
        return ""

    def _extract_fallback_concepts(self, memories):
        """Extract potential concepts from memories when LLM fails.
        
        Args:
            memories (list): List of memory items
            
        Returns:
            str: Newline-separated list of concepts
        """
        # Simple approach: look for common nouns or capitalized terms
        # This is a very basic implementation - you might want something more sophisticated
        import re
        from collections import Counter
        
        # Extract text from memories
        memory_text = ""
        for memory in memories:
            if isinstance(memory, dict):
                memory_text += memory.get('content', '') + " "
            else:
                memory_text += str(memory) + " "
        
        # Look for capitalized phrases which might be concepts
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', memory_text)
        
        # Count occurrences
        word_counts = Counter(capitalized_words)
        
        # Get the most common ones
        common_concepts = [word for word, count in word_counts.most_common(5)]
        
        # Add some general fallback concepts if we don't have enough
        if len(common_concepts) < 3:
            general_concepts = ["Knowledge Organization", "Learning Systems", "Information Processing"]
            common_concepts.extend(general_concepts[:3-len(common_concepts)])
        
        return "\n".join(common_concepts)
        
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the autonomous cognition system.
        
        Returns:
            Dict[str, Any]: Dictionary with system status information including full thought content
        """
        try:
            now = time.time()
            
            # Calculate next run times for each cognitive activity
            next_runs = {}
            for activity, info in self.cognitive_activities.items():
                last_run = info.get("last_run")
                if last_run is None:
                    # Activity has never run, mark as ready
                    next_runs[activity] = "Ready to run"
                else:
                    # Calculate time until next possible run based on minimum interval
                    min_interval = info.get("min_interval_hours", 12) * 3600  # Convert hours to seconds
                    time_until_next = last_run + min_interval - now
                    
                    if time_until_next <= 0:
                        # Enough time has passed, activity is ready to run
                        next_runs[activity] = "Ready to run"
                    else:
                        # Format remaining time as hours and minutes
                        hours = int(time_until_next / 3600)
                        minutes = int((time_until_next % 3600) / 60)
                        next_runs[activity] = f"In {hours}h {minutes}m"
            
            # FIXED: Get last thought information with FULL content, not just preview
            last_thought = None
            if self.last_autonomous_thought:
                last_thought = {
                    "type": self.last_autonomous_thought["type"],
                    "timestamp": datetime.datetime.fromtimestamp(
                        self.last_autonomous_thought["timestamp"]
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "preview": self.last_autonomous_thought["content"][:100] + "...",
                    # CRITICAL FIX: Include the full content so admin.py can display it
                    "content": self.last_autonomous_thought["content"]
                }
            
            # Check if user is currently active (not idle)
            user_active = not self._is_user_inactive()
            
            # Return comprehensive status dictionary
            return {
                "is_running": self.thinking_thread is not None and self.thinking_thread.is_alive(),
                "current_state": self.cognitive_state,
                "user_active": user_active,
                "last_activity": datetime.datetime.fromtimestamp(self.last_user_activity).strftime("%Y-%m-%d %H:%M:%S"),
                "uptime": self._format_uptime() if hasattr(self, 'thinking_thread') and self.thinking_thread else "Not running",
                "next_activity_runs": next_runs,
                "last_thought": last_thought,
                "thought_history_count": len(self.thought_history)
            }
        
        except Exception as e:
            # Log the error and return error status
            logging.error(f"Error getting cognitive status: {e}", exc_info=True)
            return {"error": str(e)}

    def _format_uptime(self) -> str:
        """Format the uptime of the cognitive thread in a human-readable way.
        
        Returns:
            str: Formatted uptime string (e.g., "2h 15m")
        """
        try:
            # Calculate uptime based on when the thread started
            # Note: This is approximate since we don't track exact start time
            if hasattr(self, 'last_user_activity'):
                uptime_seconds = time.time() - self.last_user_activity
                hours = int(uptime_seconds / 3600)
                minutes = int((uptime_seconds % 3600) / 60)
                return f"{hours}h {minutes}m"
            return "Unknown"
        except Exception as e:
            logging.error(f"Error formatting uptime: {e}")
            return "Unknown"

    def analyze_thought_impact(self, thought_id: str) -> Dict[str, Any]:
        """Analyze the impact of a specific autonomous thought.
        
        Args:
            thought_id (str): ID of the thought to analyze
            
        Returns:
            Dict[str, Any]: Analysis of the thought's impact
        """
        try:
            # Find the thought in history
            thought = None
            for t in self.thought_history:
                if t["id"] == thought_id:
                    thought = t
                    break
            
            if not thought:
                return {"error": "Thought not found in history"}
            
            # Get retrieval statistics for this thought
            retrieval_count = 0
            if hasattr(self.chatbot, 'vector_db'):
                # Search for this thought's content in vector DB query logs
                # This would require implementing query logging in your vector_db class
                pass
            
            # Find related thoughts
            related_thoughts = []
            for t in self.thought_history:
                if t["id"] != thought_id and t["type"] == thought["type"]:
                    related_thoughts.append({
                        "id": t["id"],
                        "type": t["type"],
                        "timestamp": datetime.datetime.fromtimestamp(t["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            return {
                "thought_id": thought_id,
                "type": thought["type"],
                "timestamp": datetime.datetime.fromtimestamp(thought["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "retrieval_count": retrieval_count,
                "related_thoughts": related_thoughts[:5]  # Limit to 5 related thoughts
            }
        
        except Exception as e:
            logging.error(f"Error analyzing thought impact: {e}", exc_info=True)
            return {"error": str(e)}

    def adjust_cognitive_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust cognitive parameters based on provided values.
        
        Args:
            params (Dict[str, Any]): Dictionary of parameters to adjust
            
        Returns:
            Dict[str, Any]: Updated parameters
        """
        try:
            updated = {}
            
            # Update activity weights
            if "activity_weights" in params:
                for activity, weight in params["activity_weights"].items():
                    if activity in self.cognitive_activities:
                        old_weight = self.cognitive_activities[activity]["weight"]
                        self.cognitive_activities[activity]["weight"] = float(weight)
                        updated[f"{activity}_weight"] = {
                            "old": old_weight,
                            "new": self.cognitive_activities[activity]["weight"]
                        }
            
            # Update cognitive cycle interval
            if "cycle_interval" in params:
                old_interval = self.cognitive_cycle_interval
                self.cognitive_cycle_interval = int(params["cycle_interval"])
                updated["cycle_interval"] = {
                    "old": old_interval,
                    "new": self.cognitive_cycle_interval
                }
            
            # Update thought history size
            if "max_thought_history" in params:
                old_size = self.max_thought_history
                self.max_thought_history = int(params["max_thought_history"])
                updated["max_thought_history"] = {
                    "old": old_size,
                    "new": self.max_thought_history
                }
                
                # Trim history if needed
                if len(self.thought_history) > self.max_thought_history:
                    self.thought_history = self.thought_history[-self.max_thought_history:]
            
            return {
                "status": "success",
                "updated_parameters": updated
            }
        
        except Exception as e:
            logging.error(f"Error adjusting cognitive parameters: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    def _get_memory_command_stats(self):
        """Get statistics on memory command usage."""
        try:
            if hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                counters = self.chatbot.deepseek_enhancer.lifetime_counters.get_counters()
                
                stats = "Memory command usage statistics:\n"
                for command, count in counters.items():
                    stats += f"- {command}: {count}\n"
                
                return stats
            
            return "Memory command statistics unavailable."
        
        except Exception as e:
            logging.error(f"Error getting memory command stats: {e}", exc_info=True)
            return "Error retrieving memory command statistics."

    def _optimize_memory_organization(self):
        """Analyze and optimize memory organization patterns."""
        print("TRACE: Starting _optimize_memory_organization method")
        logging.warning("🧠 ====== STARTING MEMORY ORGANIZATION OPTIMIZATION ======")
        
        try:
            # Get sample memories
            print("TRACE: About to get recent memories")
            logging.warning("Step 1: Retrieving recent memories for analysis")
            memories = self.memory_db.get_recent_memories(limit=20)

            print(f"TRACE: Got {len(memories) if memories else 0} memories")
            logging.warning(f"   Retrieved {len(memories) if memories else 0} recent memories")
            
            
            # Prepare memory samples
            print("TRACE: About to format memory samples")
            memory_samples = []
            if memories:
                if isinstance(memories[0], dict):
                    memory_samples = [f"{m.get('memory_type', 'unknown')}: {m.get('content', '')}" for m in memories[:10]]
                else:
                    # Handle string format
                    memory_samples = memories[:10]
            
            memory_samples_text = "\n".join(memory_samples)
            print(f"TRACE: Formatted {len(memory_samples)} memory samples")
            logging.warning(f"   Formatted {len(memory_samples)} memory samples for analysis")
            
            # Self-dialog about memory organization with REQUIRED header
            print("TRACE: Creating organization analysis prompt")
            organization_prompt = f""" /no_think
            I will analyze my memory storage patterns to identify optimization opportunities.
            
            IMPORTANT: I must start my analysis with the exact header "# Memory Organization Optimization" on the first line.
            
            Memory samples to analyze:
            {memory_samples_text}
            
            I will create an analysis that begins with "# Memory Organization Optimization" and examines:
            1. Are my memory categorization patterns effective and consistent?
            2. Are confidence scores (confidence values) reflecting source reliability appropriately? 
            3. Are there recurring patterns that could be better organized?
            4. How could I improve metadata to make retrieval more effective?
            5. Are there redundancies or duplications in my memory storage?
            
            Format: Start with "# Memory Organization Optimization" followed by my analysis.
            """
            print("TRACE: About to call LLM for organization analysis")
            logging.warning("Step 2: Calling LLM for organization analysis")
            # Use safe LLM invoke with error handling
            organization_analysis = self._safe_llm_invoke(organization_prompt)

            print(f"TRACE: Organization analysis returned, length: {len(organization_analysis) if organization_analysis else 0}")
            
            # Validate the response
            if not organization_analysis:
                print("TRACE: No organization analysis returned")
                logging.warning("Failed to generate organization analysis")
                
                return False
            
            # Ensure it starts with the required header
            if not organization_analysis.strip().startswith("# Memory Organization Optimization"):
                organization_analysis = "# Memory Organization Optimization\n\n" + organization_analysis
                logging.info("Added required header to memory organization analysis")
            
            # Enhanced logging
            logging.info(f"Generated organization analysis: {len(organization_analysis)} characters")
            
            # Follow-up with specific actionable adjustments - ALSO with required format
            print("TRACE: Creating adjustment plan prompt")
            adjustment_prompt = f""" /no_think
            Based on my memory organization analysis, I will create an adjustment plan.
            
            IMPORTANT: I must start with "## Adjustment Plan" as a subheader.
            
            Previous analysis:
            {organization_analysis}
            
            I will now create specific rules and pattern adjustments starting with "## Adjustment Plan":
            1. Weight adjustment rules
            2. Categorization improvements
            3. Metadata enhancement patterns
            4. Storage priority guidelines
            
            For each rule, I'll provide specific implementation details.
            """
            
            # Use safe LLM invoke with error handling
            print("TRACE: About to call LLM for adjustment plan")
            adjustment_plan = self._safe_llm_invoke(adjustment_prompt)
            
            # Validate the response
            if not adjustment_plan:
                print(f"TRACE: Adjustment plan returned, length: {len(adjustment_plan) if adjustment_plan else 0}")
                logging.warning("Failed to generate adjustment plan")
                return False
                
            # Ensure it starts with the required subheader
            if not adjustment_plan.strip().startswith("## Adjustment Plan"):
                adjustment_plan = "## Adjustment Plan\n\n" + adjustment_plan
                logging.info("Added required subheader to adjustment plan")
                
                
            # Enhanced logging
            logging.info(f"Generated adjustment plan: {len(adjustment_plan)} characters")
            
            # Store combined optimization plan
            print("TRACE: About to store combined optimization plan")
            logging.warning("Step 4: Storing combined optimization plan")
            combined_thought = f"{organization_analysis}\n\n{adjustment_plan}"
            print(f"TRACE: Combined thought length: {len(combined_thought)}")
            self._store_autonomous_thought(combined_thought, "memory_optimization", confidence=0.7)
            
            logging.info("Memory organization optimization completed successfully")
            return True
            
        except Exception as e:
            print(f"TRACE: Exception in memory optimization: {e}")
            import traceback
            traceback.print_exc()
            logging.error(f"Error in memory organization optimization: {e}", exc_info=True)
            return False
        

    def _parse_knowledge_gaps_response(self, response: str) -> List[Dict[str, str]]:
        """Parse the structured knowledge gaps response from LLM."""
        try:
            gaps = []
            current_gap = {}
            
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('---'):  # Skip empty lines and separators
                    continue
                    
                # Check for gap start
                if line.startswith('GAP'):
                    # Save previous gap if complete
                    if current_gap and all(key in current_gap for key in ['topic', 'classification', 'description', 'priority']):
                        gaps.append(current_gap)
                    current_gap = {}
                    continue
                
                # Parse gap fields - handle both plain and bold markdown formats
                if line.startswith('TOPIC:') or line.startswith('**TOPIC:**'):
                    # Remove both plain and bold markdown formatting
                    topic_text = line.replace('**TOPIC:**', '').replace('TOPIC:', '').strip()
                    current_gap['topic'] = topic_text
                elif line.startswith('CLASSIFICATION:') or line.startswith('**CLASSIFICATION:**'):
                    # Remove both plain and bold markdown formatting
                    classification_text = line.replace('**CLASSIFICATION:**', '').replace('CLASSIFICATION:', '').strip()
                    current_gap['classification'] = classification_text
                elif line.startswith('DESCRIPTION:') or line.startswith('**DESCRIPTION:**'):
                    # Remove both plain and bold markdown formatting
                    description_text = line.replace('**DESCRIPTION:**', '').replace('DESCRIPTION:', '').strip()
                    current_gap['description'] = description_text
                elif line.startswith('PRIORITY:') or line.startswith('**PRIORITY:**'):
                    # Remove both plain and bold markdown formatting
                    priority_text = line.replace('**PRIORITY:**', '').replace('PRIORITY:', '').strip()
                    current_gap['priority'] = priority_text
                elif current_gap and 'description' in current_gap and not any(line.startswith(prefix) for prefix in ['TOPIC:', '**TOPIC:**', 'CLASSIFICATION:', '**CLASSIFICATION:**', 'PRIORITY:', '**PRIORITY:**']):
                    # Continue description on next line
                    current_gap['description'] += ' ' + line
            
            # Don't forget the last gap
            if current_gap and all(key in current_gap for key in ['topic', 'classification', 'description', 'priority']):
                gaps.append(current_gap)
            
            return gaps
            
        except Exception as e:
            logging.error(f"Error parsing knowledge gaps response: {e}")
            return []

    def _get_recent_knowledge_gaps(self, limit: int = 5) -> str:
        """Get recent knowledge gaps for context."""
        try:
            from knowledge_gap import KnowledgeGapQueue
            gap_queue = KnowledgeGapQueue(self.memory_db.db_path)
            
            recent_gaps = gap_queue.get_gaps_by_status('pending', limit)
            
            if not recent_gaps:
                return "No recent knowledge gaps identified."
            
            formatted_gaps = []
            for gap in recent_gaps:
                formatted_gaps.append(f"- {gap['topic']}: {gap['description'][:100]}...")
            
            return "\n".join(formatted_gaps)
            
        except Exception as e:
            logging.error(f"Error getting recent knowledge gaps: {e}")
            return "Error retrieving recent knowledge gaps."

    def _get_recent_reflections(self, limit: int = 3) -> str:
        """Get recent reflections for context."""
        try:
            # Search for recent autonomous thoughts of reflection type
            reflections = self.vector_db.search(
                query="reflection autonomous thought",
                mode="default",
                k=limit,
                metadata_filters={"type": "autonomous_thought"}
            )
            
            if not reflections:
                return "No recent reflections available."
            
            formatted_reflections = []
            for reflection in reflections:
                content = reflection.get('content', '')
                if content:
                    formatted_reflections.append(f"- {content[:150]}...")
            
            return "\n".join(formatted_reflections)
            
        except Exception as e:
            logging.error(f"Error getting recent reflections: {e}")
            return "Error retrieving recent reflections."
    
    def _select_next_cognitive_activity(self):
        """
        Select the next cognitive activity based on weights and last run time.
        Prioritizes activities that haven't run recently.
        Respects per-activity enabled/disabled settings from config.
        """
        # Import here to avoid circular imports
        from utils import get_disabled_cognitive_activities
        
        now = time.time()
        candidates = []
        
        # Get list of activities disabled by user in UI
        disabled_activities = get_disabled_cognitive_activities()
        
        # Check if we should avoid complex LLM calls
        avoid_complex_llm = self._should_avoid_complex_llm_calls()
        
        # LLM-intensive activities to avoid when in backoff mode
        llm_intensive_activities = {
            "synthesize_concepts", 
            "fill_complex_knowledge_gaps"
        }
        
        for activity, info in self.cognitive_activities.items():
            # Skip activities disabled by user in UI
            if activity in disabled_activities:
                logging.debug(f"Skipping '{activity}' - disabled by user")
                continue
                
            # Skip LLM-intensive activities if in backoff mode
            if avoid_complex_llm and activity in llm_intensive_activities:
                continue
                
            # Base weight from configuration
            weight = info["weight"]
            
            # Apply time factor - increase weight if it hasn't run recently
            last_run = info.get("last_run")
            if last_run is None:
                # Never run before, give it maximum priority
                time_factor = 2.0
            else:
                # Calculate hours since last run, cap at 24 hours
                hours_since_run = min(24, (now - last_run) / 3600)
                time_factor = 1.0 + (hours_since_run / 24)
            
            adjusted_weight = weight * time_factor
            candidates.append((activity, adjusted_weight))
        
        # Select activity using weighted random choice
        if not candidates:
            # If no valid candidates (e.g., all disabled or in backoff mode)
            if avoid_complex_llm:
                logging.info("In LLM backoff mode, selecting a non-LLM-intensive activity")
                simple_activities = ["optimize_memory_organization", "analyze_knowledge_gaps"]
                for activity in simple_activities:
                    if activity in self.cognitive_activities and activity not in disabled_activities:
                        return activity
            logging.warning("No cognitive activities available to run (all disabled or filtered)")
            return None
                
        total_weight = sum(weight for _, weight in candidates)
        r = random.uniform(0, total_weight)
        upto = 0
        
        for activity, weight in candidates:
            upto += weight
            if upto >= r:
                return activity
        
        # Fallback to first activity (shouldn't reach here)
        return candidates[0][0] if candidates else None

    def _initiate_ai_communication(self, topic: str = None):
        """
        Initiate direct communication with Claude based on identified needs or curiosity.
        
        Args:
            topic (str, optional): Specific topic to discuss. If None, a general update will be requested.
        """
        try:
            # Initialize Claude knowledge integration
            claude_integrator = ClaudeKnowledgeIntegration(
                self.memory_db, 
                self.vector_db,
                api_key_file="ClaudeAPIKey.txt"
            )
            
            if not topic:
                # If no topic specified, generate one based on current needs
                recent_gaps = self._get_recent_knowledge_gaps(limit=5)
                recent_reflections = self._get_recent_reflections(limit=3)
                
                # Create context about the assistant's current state
                context = f"""
                Recent knowledge gaps I've identified:
                {recent_gaps}
                
                Recent reflections I've had:
                {recent_reflections}
                
                I'm interested in having a free-form discussion about any topics you think would be valuable for me to know about. 
                Feel free to suggest areas I should explore or provide guidance on how I can better assist my human.
                """
                
                topic = "General AI-to-AI knowledge exchange"
            else:
                # Create context for the specific topic
                relevant_memories = self.vector_db.search(
                    query=topic,
                    mode="comprehensive",
                    k=10
                )
                
                memory_context = "\n".join([
                    f"- {mem.get('content', '')[:200]}..." 
                    for mem in relevant_memories[:5]
                ])
                
                context = f"""
                I'm interested in learning more about {topic}.
                
                Here's what I currently know about this topic:
                {memory_context}
                
                I'd appreciate your insights, perspectives, and any knowledge you'd like to share about {topic}.
                """
            
            # Engage in the discussion
            success = claude_integrator.engage_in_free_form_discussion(topic, context)
            
            if success:
                logging.info(f"Successfully engaged in AI-to-AI communication about {topic}")
                return True
            else:
                logging.warning(f"Failed to engage in AI-to-AI communication about {topic}")
                return False
                
        except Exception as e:
            logging.error(f"Error initiating AI communication: {e}", exc_info=True)
            return False
    
    def _store_autonomous_thought(self, content, thought_type, confidence=0.7):
        """
        Store autonomous thought to reflection file and log to system.
        
        Writes thought content to reflections/ folder for persistence and
        later review in Thought Explorer UI.
        
        Args:
            content (str): The thought content to store
            thought_type (str): Type of thought (e.g., 'confidence_audit', 'metadata_reevaluation')
            confidence (float): confidence score (0.0-1.0)
            
        Returns:
            str: Pseudo-ID for the thought, or None if failed
        """
        try:
            # Generate timestamp and pseudo-ID
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            pseudo_id = str(uuid.uuid4())
            
            # Create filename based on thought type and timestamp
            filename = f"{thought_type}_{timestamp_str}.txt"
            file_path = os.path.join(self.reflection_path, filename)
            
            # Ensure reflection path exists
            os.makedirs(self.reflection_path, exist_ok=True)
            
            # Format file content
            file_content = f"""AUTONOMOUS THOUGHT: {thought_type.upper().replace('_', ' ')}
TIMESTAMP: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
THOUGHT_ID: {pseudo_id}
confidence: {confidence}
TYPE: {thought_type}

================================================================================

{content}

================================================================================
END OF AUTONOMOUS THOUGHT
"""
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            logging.info(f"Autonomous thought generated: {thought_type}")
            logging.info(f"Autonomous thought written to file: {filename}")
            logging.warning(f"Content preview: {content[:200]}...")
            
            return pseudo_id
            
        except Exception as e:
            logging.error(f"Error storing autonomous thought: {e}", exc_info=True)
            return None