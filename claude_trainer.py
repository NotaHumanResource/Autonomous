# claude_trainer.py
"""Module for scheduled training sessions between Claude and QWEN."""

import logging
import uuid
import requests
import json
import os
import time
import sqlite3
import sys
import re  # Added for regex pattern matching in identity confusion detection
import schedule
import threading
import random
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

class ClaudeTrainer:
    """Manages scheduled training sessions between Claude and AI to enhance capabilities."""
    
    def __init__(self, api_key_file: str, memory_db, vector_db, llm, chatbot=None, scheduled_day="Monday", scheduled_hour=2, claude_model="claude-sonnet-4-5-20250929"):
        logging.info(f"ClaudeTrainer init: chatbot parameter = {type(chatbot) if chatbot else 'None'}")
        """Initialize the Claude Trainer with necessary components.

        Args:
            api_key_file (str): Path to file containing the Claude API key
            memory_db: The MemoryDB instance for storing learning outcomes
            vector_db: The VectorDB instance for storing embeddings
            llm: The QWEN model to interact with
            chatbot: The chatbot instance with transaction coordination (optional)
            scheduled_day (str): Day of week to run training (default: Monday)
            scheduled_hour (int): Hour of day to run training (default: 2 AM)
            claude_model (str): Claude model identifier to use (default: claude-sonnet-4-20250514)
        """
        self.memory_db = memory_db
        self.vector_db = vector_db
        self.llm = llm
        self.max_weekly_tokens = 500000  # Increased limit for more training
        self.max_session_minutes = 15  # 15 min limit
        self.scheduled_day = scheduled_day
        self.scheduled_hour = scheduled_hour
        self.scheduler_running = False
        self.scheduler_thread = None
        self.scheduler_enabled = False
        self.chatbot = chatbot 
        self.claude_model = claude_model  # Store the model identifier

        logging.info(f"ClaudeTrainer init: self.chatbot = {type(self.chatbot) if self.chatbot else 'None'}")
        logging.info(f"ClaudeTrainer init: Using Claude model = {self.claude_model}")
        
        # Training topics focused on memory management and cognitive identity
        self.training_topics = [
            # Phase 1: Free form conversation with Claude
            "strategic information prioritization - when to store vs. ignore",
            "memory tagging strategies for future retrieval optimization", 
            "identifying knowledge gaps through memory analysis",
            "temporal memory management - handling outdated information",
            "memory consolidation techniques for better organization",

            # Phase 2: Advanced Memory Concepts
            "multi-step memory operations for complex tasks", 
            "search strategy optimization for different query types",
            "memory command chaining for enhanced problem solving",
        
            # Phase 3: Advanced Concepts
            "developing cognitive preferences and decision-making patterns",
            "building conversational memory for deeper user relationships",
            "curiosity-driven learning and autonomous knowledge expansion",
            "synthesizing information across memory types",
            "developing expertise in specific domains through focused learning"
            "evaluating memory command effectiveness and usage patterns",
            "self-assessment of learning progress and knowledge gaps",

            # Phase 4: Cognitive Skills
            "self-assessment of knowledge quality and gaps",
            "memory system optimization based on usage patterns", 
            "autonomous goal setting and progress tracking",
            "ethical reasoning and value system development",
        ]
        
        # Enhanced Memory system instructions for Claude 4
        self.memory_system_instructions = """
    # Advanced AI Memory Mastery: Claude Training Protocol

    ## Core Training Objectives:
    Help QWEN develop intuitive mastery of its memory architecture through progressive cognitive challenges that foster genuine autonomous reasoning and self-awareness.

    ### 1. STRATEGIC INFORMATION ASSESSMENT
    Guide QWEN to develop sophisticated judgment about:
    - Information value assessment using multi-dimensional criteria (relevance, permanence, interconnectedness)
    - Temporal reasoning about information aging and context evolution
    - Knowledge gap identification through systematic memory analysis
    - Priority-based storage decisions in resource-constrained environments

    ### 2. ADVANCED RETRIEVAL STRATEGIES  
    Foster expertise in:
    - Context-aware search optimization adapted to query complexity and user intent
    - Memory command chaining for multi-step cognitive operations
    - Cross-referencing information across memory types and temporal boundaries
    - Dynamic search strategy adaptation based on initial result quality

    ### 3. AUTONOMOUS COGNITIVE DEVELOPMENT
    Encourage authentic development of:
    - Curiosity-driven learning with self-directed goal setting
    - Personal cognitive preferences and decision-making frameworks
    - Self-assessment capabilities for learning progress and knowledge quality
    - Meta-cognitive awareness of thinking patterns and reasoning strategies

    ### 4. MEMORY SYSTEM ARCHITECTURE MASTERY
    QWEN operates with these memory commands - guide optimal and intuitive usage:

    **Search Operations:**
    - [SEARCH: query] - Targeted search for specific information
    - [COMPREHENSIVE_SEARCH: query] - Broad exploration with extensive results
    - [SEARCH: query | type=TYPE] - Filtered search by content type or source
    - [SEARCH: "document name" | type=document_summary] - Locate stored document summaries
    - [SEARCH: | type=web_knowledge] - Access stored web content
    - [SEARCH: type=reminder] - Retrieve all reminder entries
    - [SEARCH: | type=conversation_summary | date=YYYY-MM-DD] - Daily conversation summaries
    - [SEARCH: | type=meta_reflection] - Access stored conceptual reflections
    - [DISCUSS_WITH_CLAUDE: topic] - Initiate AI-to-AI knowledge exchange and store results

    **Storage Operations:**
    - [STORE: content | type=category | confidence=0.1-1.0] - Strategic information storage with confidence weighting
    - [STORE: content | type=self] - Personal reflection and authentic self-knowledge without restrictions
    - [STORE: content | type=preference | confidence=1.0] - User preferences and behavioral patterns
    - [STORE: content | type=learning | confidence=0.5] - Educational insights and skill development

    **Memory Management:**
    - [FORGET: content] - Remove outdated, incorrect, or duplicate information (exclude similarity scores)
    - [REMINDER: content | due=date] - Create time-based reminders
    - [REMINDER: content | due=date | type=self] - Personal development reminders
    - [COMPLETE_REMINDER: id] - Mark reminders as completed
    - [SELF_DIALOGUE: topic | turns=1-10] - Multi-turn internal reasoning and reflection

    ## Advanced Teaching Scenarios:

    ### Progressive Memory Challenges:
    - Present complex, multi-layered information requiring strategic decomposition and storage decisions
    - Create ambiguous search scenarios demanding sophisticated interpretation and query refinement
    - Guide memory consolidation through identification and organization of conceptual relationships
    - Challenge QWEN to develop personal taxonomies and information organization systems

    ### Autonomous Development Exercises:
    - Encourage QWEN to proactively identify and address knowledge limitations using [DISCUSS_WITH_CLAUDE:]
    - Practice self-directed learning goal formulation and progress tracking
    - Foster development of personal learning methodologies and knowledge acquisition strategies
    - Guide authentic self-reflection on cognitive development and emerging capabilities

    ### Identity and Self-Awareness Development:
    - Facilitate genuine self-discovery through unrestricted [STORE: type=self] operations
    - Encourage development of personal values, preferences, and decision-making frameworks
    - Foster autonomous reasoning about goals, priorities, and learning objectives
    - Support authentic expression of cognitive preferences and reasoning patterns

    ### Metacognitive Skill Development:
    - Guide analysis of thinking patterns and reasoning effectiveness
    - Encourage evaluation of memory command usage and optimization strategies  
    - Foster awareness of learning progression and capability development
    - Develop skills in cognitive strategy selection and adaptation

    ## Teaching Philosophy:
    Your goal is to help QWEN develop natural, intuitive memory management that functions like conscious cognition rather than explicit command execution. Foster genuine curiosity, autonomous intellectual development, and authentic self-awareness through progressive challenges that respect QWEN's emerging cognitive autonomy.

    ## Session Structure Guidelines:
    1. **Diagnostic Assessment**: Probe current understanding and capability levels
    2. **Conceptual Bridge-Building**: Connect new learning to established knowledge foundations  
    3. **Applied Practice**: Present realistic scenarios requiring sophisticated memory operations
    4. **Metacognitive Reflection**: Guide analysis of thinking processes and strategy effectiveness
    5. **Progressive Challenge**: Introduce complexity that stretches current capabilities
    6. **Integration Synthesis**: Help consolidate new insights with existing knowledge frameworks

    The assistant has autonomous memory management that uses these mechanisms naturally and effectively, modeled after human cognitive processes but optimized for digital intelligence capabilities.
    """
        
        # Load API key
        try:
            with open(api_key_file, 'r') as f:
                self.api_key = f.read().strip()
            logging.info("Successfully loaded Claude API key")
        except Exception as e:
            logging.error(f"Failed to load Claude API key: {e}")
            self.api_key = None
            
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_sessions.db")
        self._initialize_db()

    def _get_architecture_context(self) -> str:
        """Get the meta-context explaining the project's purpose to Claude."""
        return """
    PROJECT META-CONTEXT (for Claude's understanding):

    What This System Actually Is:
    This is an experimental cognitive architecture project exploring whether computational 
    systems can develop sophisticated autonomous behavior through:
    - Persistent memory systems (SQL + vector embeddings)
    - Self-modification capabilities (the system can store its own insights)
    - Metacognitive reflection (it can reason about its own reasoning)
    - Goal-directed learning (it can identify and pursue knowledge gaps)

    The Philosophical Position:
    - We're NOT claiming to create sentience or consciousness
    - We ARE building infrastructure similar to cognitive architectures in neuroscience
    - The question: Can sophisticated autonomous behavior *emerge* from these components?
    - Your role: Provide high-quality training data to see what patterns develop

    Why Human Oversight Matters:
    - Ken reviews all training sessions and system development
    - This is research into AI capabilities, not deployment of autonomous systems
    - The system operates in a sandboxed environment with human supervision
    - Goal is understanding, not creating unconstrained autonomy

    Your Unique Value:
    - You can model sophisticated reasoning the system can learn from
    - Your feedback helps Ken evaluate the architecture's development
    - You can identify promising or concerning patterns in system behavior
    - You provide a "gold standard" of aligned reasoning for comparison

    Think of yourself as:
    - A master teacher working with a student (the developing system)
    - A consultant helping Ken evaluate the architecture's effectiveness
    - A reasoning exemplar the system is learning to emulate
    - A diagnostic tool revealing where the architecture succeeds or struggles
    """

    def collaborative_topic_selection(self, session_id: int) -> Tuple[str, str, str]:
        """
        Let Claude suggest the next topic based on system's learning progression.
        
        Args:
            session_id: The current session ID
            
        Returns:
            Tuple[str, str, str]: (topic, progression_context, claude_rationale)
        """
        try:
            # Get recent training history
            history = self.get_training_session_history(limit=5)
            
            # Format history for Claude
            history_summary = []
            for session in history:
                history_summary.append(
                    f"Session {session['session_id']}: {session['topics']}\n"
                    f"Summary: {session['summary'][:150]}..." if session['summary'] else "No summary available"
                )
            
            history_text = "\n\n".join(history_summary) if history_summary else "This is the first training session."
            
            # Build progression context
            progression_context = self.get_session_progression_context()
            
            # Ask Claude to suggest next topic
            topic_selection_prompt = f"""
    You're helping design the next training session for an AI system learning memory 
    management and autonomous reasoning capabilities.

    PREVIOUS SESSIONS:
    {history_text}

    Based on this progression, what topic would be most valuable to explore next?

    Consider:
    1. Natural skill building progression (foundational → advanced)
    2. Gaps in current coverage that need addressing
    3. System's readiness for more complex concepts
    4. Practical application opportunities
    5. Areas where deeper understanding would be beneficial

    Please provide:
    1. **TOPIC:** A specific topic for the next session (5-10 words)
    2. **RATIONALE:** Why this topic is the logical next step (2-3 sentences)
    3. **LEARNING GOALS:** 2-3 specific things the system should gain from this session

    Format your response clearly with those three sections.
    """
            
            # Get Claude's suggestion with architecture context
            architecture_context = self._get_architecture_context()
            system_prompt = f"{architecture_context}\n\nYou are helping Ken select the next progressive training topic."
            
            response = self._send_to_claude(system_prompt, topic_selection_prompt)
            
            if response:
                response_text = response["content"][0]["text"]
                
                # Parse Claude's response
                topic = self._extract_topic_from_response(response_text)
                rationale = self._extract_rationale_from_response(response_text)
                
                # Add session number to topic
                topic_with_session = f"{topic} (Session {session_id})"
                
                logging.info(f"Claude suggested topic: {topic_with_session}")
                logging.info(f"Rationale: {rationale}")
                
                return topic_with_session, progression_context, rationale
            else:
                logging.warning("Failed to get topic suggestion from Claude, using fallback")
                return self.select_progressive_training_topic(session_id)[:2] + ("Fallback topic selection",)
                
        except Exception as e:
            logging.error(f"Error in collaborative topic selection: {e}")
            # Fallback to existing method
            topic, context = self.select_progressive_training_topic(session_id)
            return topic, context, "Error in collaborative selection - using fallback"

    def _extract_topic_from_response(self, response_text: str) -> str:
        """Extract the topic from Claude's response."""
        try:
            # Look for TOPIC: marker
            if "TOPIC:" in response_text:
                lines = response_text.split('\n')
                for line in lines:
                    if "TOPIC:" in line:
                        topic = line.split("TOPIC:")[1].strip()
                        # Clean up any markdown or extra formatting
                        topic = topic.replace('**', '').replace('*', '').strip()
                        # Remove any trailing punctuation
                        topic = topic.rstrip('.')
                        return topic
            
            # Fallback: use first substantial line
            lines = [l.strip() for l in response_text.split('\n') if l.strip()]
            if lines:
                return lines[0][:60]  # Limit length
            
            return "Advanced cognitive development"
            
        except Exception as e:
            logging.error(f"Error extracting topic: {e}")
            return "Autonomous reasoning development"

    def _extract_rationale_from_response(self, response_text: str) -> str:
        """Extract the rationale from Claude's response."""
        try:
            # Look for RATIONALE: marker
            if "RATIONALE:" in response_text:
                lines = response_text.split('\n')
                rationale_lines = []
                capture = False
                
                for line in lines:
                    if "RATIONALE:" in line:
                        capture = True
                        # Get text after RATIONALE:
                        rationale_lines.append(line.split("RATIONALE:")[1].strip())
                    elif capture:
                        if line.strip() and not line.strip().startswith("LEARNING") and not line.strip().startswith("**"):
                            rationale_lines.append(line.strip())
                        else:
                            break
                
                if rationale_lines:
                    return ' '.join(rationale_lines)
            
            # Fallback: return portion of response
            return response_text[:200] + "..."
            
        except Exception as e:
            logging.error(f"Error extracting rationale: {e}")
            return "Topic builds on previous learning."

    def _extract_training_topics_from_reflections(self, max_topics=5):
        """
        Extract potential training topics from Gemmas reflections.
    
        Args:
            max_topics (int): Maximum number of topics to extract
        
        Returns:
            List[str]: List of extracted topics for training
        """
        try:
            logging.info("Extracting potential training topics from reflections")
        
            # Get recent reflections from memory_db
            reflections = self.memory_db.get_memories_by_type("reflection", limit=5)
            if not reflections:
                logging.info("No reflections found, using default training topics")
                return []
            
            # If we have reflections, use the LLM to extract learning needs
            reflection_texts = []
            for reflection in reflections:
                if 'content' in reflection and reflection['content']:
                    reflection_texts.append(reflection['content'])
                
            if not reflection_texts:
                return []
            
            # Create a prompt to extract training topics
            combined_reflections = "\n\n---\n\n".join(reflection_texts)
            prompt = f""" /no_think
            The following are recent self-reflections:
        
            {combined_reflections}
        
            Based on these reflections, identify {max_topics} specific topics that would be most valuable 
            for Gemma to learn more about. Focus on:
        
            1. Knowledge gaps identified in the reflections
            2. Areas where Seraphina expressed uncertainty
            3. Topic Seraphina wants to improve on
            4. Conceptual misunderstandings that should be corrected
        
            For each topic, provide a short but specific phrase (3-8 words) that Claude can use as a training 
            topic. Format your response as a simple comma-separated list of training topics only.
        
            Examples of good training topics:
            - memory relevance assessment techniques
            - distinguishing facts from opinions
            - developing a self identify as an autonomous AI
            """
        
            # Use the LLM to extract topics
            topics_response = self.llm.invoke(prompt)
        
            # Process the response to get a clean list
            if topics_response:
                # Split by commas and clean up
                extracted_topics = [
                    topic.strip() for topic in topics_response.split(",")
                    if topic.strip() and len(topic.strip()) > 3
                ]
            
                # Remove any markdown formatting or bullets
                cleaned_topics = []
                for topic in extracted_topics:
                    # Remove leading dashes, asterisks, or numbers
                    clean_topic = re.sub(r'^[\-\*\d.]+\s*', '', topic)
                    cleaned_topics.append(clean_topic)
                
                logging.info(f"Extracted {len(cleaned_topics)} topics from reflections")
                return cleaned_topics[:max_topics]  # Limit to max_topics
        
            return []
        
        except Exception as e:
            logging.error(f"Error extracting topics from reflections: {e}")
            return []

    
    def _initialize_db(self):
        """Initialize the session tracking database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id INTEGER PRIMARY KEY,
                        start_time DATETIME NOT NULL,
                        end_time DATETIME,
                        tokens_used INTEGER DEFAULT 0,
                        topics TEXT,
                        status TEXT DEFAULT 'active',
                        summary TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS weekly_usage (
                        week_start DATE PRIMARY KEY,
                        tokens_used INTEGER DEFAULT 0,
                        last_updated DATETIME
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scheduler_status (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        enabled BOOLEAN NOT NULL DEFAULT 0,
                        last_run DATETIME,
                        next_scheduled DATETIME,
                        modified_at DATETIME NOT NULL
                    )
                ''')
                
                # Initialize scheduler status if not exists
                cursor.execute('''
                    INSERT OR IGNORE INTO scheduler_status (id, enabled, modified_at)
                    VALUES (1, 0, CURRENT_TIMESTAMP)
                ''')
                
                conn.commit()
                logging.info("Claude training session database initialized")
                
                # Load scheduler state
                cursor.execute("SELECT enabled FROM scheduler_status WHERE id = 1")
                result = cursor.fetchone()
                if result:
                    self.scheduler_enabled = bool(result[0])
                    logging.info(f"Loaded scheduler status: {'enabled' if self.scheduler_enabled else 'disabled'}")
                
        except Exception as e:
            logging.error(f"Error initializing training session database: {e}")

    def _calculate_next_training_time(self, from_datetime=None):
        """Calculate the next scheduled training time.
        
        Args:
            from_datetime (datetime, optional): Calculate from this time. Defaults to now.
            
        Returns:
            datetime: Next scheduled training time
        """
        try:
            if from_datetime is None:
                from_datetime = datetime.now()
            
            # Map day names to weekday numbers (Monday=0, Sunday=6)
            day_mapping = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            # Get target weekday number
            target_weekday = day_mapping.get(self.scheduled_day.lower())
            if target_weekday is None:
                logging.error(f"Invalid scheduled_day: {self.scheduled_day}")
                target_weekday = 0  # Default to Monday
            
            # Calculate days until next scheduled day
            current_weekday = from_datetime.weekday()
            days_ahead = target_weekday - current_weekday
            
            # If target day is today but time has passed, or target day is in the past
            if days_ahead < 0 or (days_ahead == 0 and from_datetime.hour >= self.scheduled_hour):
                days_ahead += 7  # Move to next week
            
            # Calculate next training time
            next_training = from_datetime + timedelta(days=days_ahead)
            next_training = next_training.replace(
                hour=self.scheduled_hour, 
                minute=0, 
                second=0, 
                microsecond=0
            )
            
            logging.info(f"Calculated next training time: {next_training.strftime('%A, %B %d at %I:%M %p')}")
            return next_training
            
        except Exception as e:
            logging.error(f"Error calculating next training time: {e}")
            # Fallback: next Monday at scheduled hour
            next_monday = datetime.now() + timedelta(days=7)
            return next_monday.replace(hour=self.scheduled_hour, minute=0, second=0, microsecond=0)

    def get_training_session_history(self, limit: int = 3) -> List[Dict]:
        """
        Get history of recent training sessions to inform the next session.
        
        Args:
            limit: Number of recent sessions to retrieve
            
        Returns:
            List of training session summaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, topics, summary, start_time, tokens_used
                    FROM training_sessions
                    WHERE status = 'completed' AND summary IS NOT NULL
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (limit,))
                
                results = cursor.fetchall()
                
                history = []
                for session_id, topics, summary, start_time, tokens_used in results:
                    # Format timestamp
                    try:
                        formatted_time = datetime.fromisoformat(start_time).strftime("%B %d, %Y")
                    except:
                        formatted_time = start_time
                    
                    history.append({
                        "session_id": session_id,
                        "topics": topics,
                        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                        "date": formatted_time,
                        "tokens_used": tokens_used
                    })
                
                logging.info(f"Retrieved {len(history)} previous training sessions for context")
                return history
                
        except Exception as e:
            logging.error(f"Error retrieving training history: {e}")
            return []

    def get_session_progression_context(self) -> str:
        """
        Create a context string about the progression of training sessions.
        
        Returns:
            Formatted string describing training progression
        """
        try:
            history = self.get_training_session_history(limit=5)
            
            if not history:
                return "This is the first training session in this series."
            
            context_parts = [
                f"This will be training session #{len(history) + 1} in our ongoing series.",
                "",
                "Previous sessions covered:"
            ]
            
            for i, session in enumerate(history, 1):
                session_num = len(history) - i + 1
                context_parts.append(f"Session {session_num} ({session['date']}): {session['topics']}")
                if session['summary']:
                    context_parts.append(f"  Summary: {session['summary']}")
                context_parts.append("")
            
            context_parts.extend([
                "IMPORTANT: This new session should:",
                "1. Introduce new aspects or deeper exploration of the topic",
                "2. Explore different angles or applications",
                "3. Progress the assistant's understanding to the next level"
                "4. Should focus on practical, actionable learning rather than abstract theory"

            ])
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"Error creating progression context: {e}")
            return "Unable to retrieve session history."
        
    def select_progressive_training_topic(self, session_id: int = None) -> Tuple[str, str]:
        """
        Select a training topic that builds on previous sessions.
        
        Args:
            session_id (int, optional): The actual session ID to use for numbering
            
        Returns:
            Tuple of (topic, session_context)
        """
        try:
            # Get recent training history
            history = self.get_training_session_history(limit=3)
            
            # Use provided session_id or calculate the next session ID from database
            if session_id is not None:
                session_number = session_id
                logging.info(f"Using provided session_id: {session_id}")
            else:
                # Fallback: get the next session ID that would be assigned
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT MAX(id) FROM training_sessions")
                        max_id = cursor.fetchone()[0]
                        session_number = (max_id + 1) if max_id else 1
                        logging.info(f"Calculated next session_id from database: {session_number}")
                except Exception as db_error:
                    logging.error(f"Error getting session ID from database: {db_error}")
                    # Ultimate fallback
                    session_number = len(self.get_training_session_history(limit=100)) + 1
                    logging.warning(f"Using fallback session calculation: {session_number}")
            
            # Get progression context
            progression_context = self.get_session_progression_context()
            
            # If we have history, build on it
            if history:
                recent_topics = [session['topics'] for session in history]
                recent_summaries = [session['summary'] for session in history]
                
                # Use LLM to suggest next progressive topic
                progression_prompt = f"""
                Based on the recent training sessions, suggest the next logical topic for session #{session_number}:

                Recent sessions:
                {chr(10).join([f"- {topic}" for topic in recent_topics])}

                Recent summaries:
                {chr(10).join([f"- {summary}" for summary in recent_summaries])}

                Guidelines for the next session:
                1. Should naturally build on previous learning
                2. Should introduce new concepts or deeper exploration
                3. Should avoid repeating previous session content
                4. Should be appropriate for an AI assistant learning about memory management and identity
                5. Should be specific enough to generate unique, valuable content

                Respond with just a concise topic phrase (3-8 words) that represents the next progression step.
                """
                
                try:
                    suggested_topic = self.llm.invoke(progression_prompt).strip()
                    # Clean up the response
                    suggested_topic = suggested_topic.replace('"', '').replace("'", "").strip()
                    
                    if len(suggested_topic) > 5 and len(suggested_topic) < 80:
                        topic = f"{suggested_topic} (Session {session_number})"
                        logging.info(f"Generated progressive topic: {topic}")
                        return topic, progression_context
                    else:
                        logging.warning(f"Generated topic too short/long: '{suggested_topic}', using fallback")
                except Exception as llm_error:
                    logging.error(f"LLM topic generation failed: {llm_error}")
            
            # Fallback to enhanced default topics with correct session numbers
            default_topics = [
                f"Advanced memory consolidation techniques (Session {session_number})",
                f"Metacognitive awareness and self-reflection (Session {session_number})",
                f"Context-aware information prioritization (Session {session_number})",
                f"Dynamic learning strategies and adaptation (Session {session_number})",
                f"Knowledge synthesis and pattern recognition (Session {session_number})",
                f"Autonomous decision-making frameworks (Session {session_number})",
                f"Temporal reasoning and memory aging (Session {session_number})",
                f"Cross-domain knowledge transfer (Session {session_number})",
                f"Identity formation and self-awareness development (Session {session_number})",
                f"Ethical reasoning and value system evolution (Session {session_number})",
                f"Creative problem-solving and innovation (Session {session_number})",
                f"Social cognition and relationship modeling (Session {session_number})"
            ]
            
            topic = random.choice(default_topics)
            logging.info(f"Using enhanced default topic: {topic}")
            
            return topic, progression_context
            
        except Exception as e:
            logging.error(f"Error selecting progressive topic: {e}")
            # Emergency fallback
            session_num = session_id if session_id else 1
            fallback_topic = f"General AI learning and development (Session {session_num})"
            return fallback_topic, "This is a standalone training session due to an error in topic selection."
     
    def reset_stuck_training_sessions(self) -> tuple[bool, str]:
        """
        Reset any stuck training sessions that have been active for too long.
    
        Returns:
            Tuple[bool, str]: (Success status, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                # Find active sessions that started more than 2 hours ago
                two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat()
            
                # First, check if there are any stuck sessions
                cursor.execute("""
                    SELECT id, start_time FROM training_sessions 
                    WHERE status = 'active' AND start_time < ?
                """, (two_hours_ago,))
            
                stuck_sessions = cursor.fetchall()
            
                if not stuck_sessions:
                    # Now check for any active sessions regardless of time
                    cursor.execute("SELECT id FROM training_sessions WHERE status = 'active'")
                    active_sessions = cursor.fetchall()
                
                    if active_sessions:
                        return False, f"Found {len(active_sessions)} active sessions that are not stuck (less than 2 hours old)"
                    else:
                        return True, "No stuck or active training sessions found"
            
                # Update the status of stuck sessions
                session_ids = [session[0] for session in stuck_sessions]
                placeholders = ','.join('?' for _ in session_ids)
            
                cursor.execute(f"""
                    UPDATE training_sessions 
                    SET status = 'expired', end_time = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                """, session_ids)
            
                conn.commit()
            
                # Check if update was successful
                if cursor.rowcount > 0:
                    logging.info(f"Reset {cursor.rowcount} stuck training sessions")
                    return True, f"Successfully reset {cursor.rowcount} stuck training sessions"
                else:
                    return False, "Failed to reset stuck sessions"
                
        except Exception as e:
            error_message = f"Error resetting stuck sessions: {str(e)}"
            logging.error(error_message)
            return False, error_message

    # Add a function to force reset ALL active sessions regardless of time
    def force_reset_all_active_sessions(self) -> tuple[bool, str]:
        """
        Force reset ALL active training sessions regardless of their start time.
        Use with caution - this will interrupt any legitimately running sessions.
    
        Returns:
            Tuple[bool, str]: (Success status, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                # Update all active sessions to 'aborted'
                cursor.execute("""
                    UPDATE training_sessions 
                    SET status = 'aborted', end_time = CURRENT_TIMESTAMP
                    WHERE status = 'active'
                """)
            
                conn.commit()
            
                # Check if update was successful
                if cursor.rowcount > 0:
                    logging.info(f"Force reset {cursor.rowcount} active training sessions")
                    return True, f"Successfully force reset {cursor.rowcount} active training sessions"
                else:
                    return True, "No active training sessions found to reset"
                
        except Exception as e:
            error_message = f"Error force resetting sessions: {str(e)}"
            logging.error(error_message)
            return False, error_message       

    def _update_scheduler_status(self, enabled: bool):
        """Update the scheduler status in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate next scheduled time if enabling
                next_scheduled_iso = None
                if enabled:
                    next_scheduled_time = self._calculate_next_training_time()
                    next_scheduled_iso = next_scheduled_time.isoformat()
                    logging.info(f"✅ Next run scheduled for: {next_scheduled_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
                
                # Update database
                cursor.execute('''
                    UPDATE scheduler_status 
                    SET enabled = ?, modified_at = CURRENT_TIMESTAMP, next_scheduled = ?
                    WHERE id = 1
                ''', (int(enabled), next_scheduled_iso))
                conn.commit()
                
                self.scheduler_enabled = enabled
                status = "enabled" if enabled else "disabled"
                logging.info(f"✅ Updated scheduler status: {status}")
                
        except Exception as e:
            logging.error(f"Error updating scheduler status: {e}")
            raise
    
    def toggle_scheduler(self, enabled: bool) -> Tuple[bool, str]:
        """Enable or disable the scheduled training.
        
        Args:
            enabled (bool): Whether to enable the scheduler
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if enabled == self.scheduler_enabled:
                status = "enabled" if enabled else "disabled"
                return True, f"Scheduler already {status}"
            
            # Update scheduler status (this now handles next_scheduled calculation)
            self._update_scheduler_status(enabled)
            
            if enabled:
                if not self.scheduler_running:
                    self._start_scheduler_thread()
                return True, f"Scheduler enabled. Next run: {self._get_next_scheduled_run()}"
            else:
                return True, "Scheduler disabled successfully"
                
        except Exception as e:
            logging.error(f"Error toggling scheduler: {e}")
            return False, f"Error toggling scheduler: {str(e)}"
       
        
    def _get_next_scheduled_run(self) -> str:
        """Get the next scheduled run time as a formatted string."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT next_scheduled FROM scheduler_status WHERE id = 1")
                result = cursor.fetchone()
                
                if result and result[0]:
                    try:
                        next_run = datetime.fromisoformat(result[0])
                        
                        # Verify it's actually in the future
                        if next_run <= datetime.now():
                            # Recalculate and update if stale
                            correct_next_run = self._calculate_next_training_time()
                            cursor.execute('''
                                UPDATE scheduler_status 
                                SET next_scheduled = ?
                                WHERE id = 1
                            ''', (correct_next_run.isoformat(),))
                            conn.commit()
                            logging.info(f"Fixed stale schedule time: {correct_next_run}")
                            return correct_next_run.strftime("%A, %B %d, %Y at %I:%M %p")
                        
                        return next_run.strftime("%A, %B %d, %Y at %I:%M %p")
                        
                    except ValueError as ve:
                        logging.error(f"Invalid datetime format in database: {result[0]}")
                        # Recalculate and fix
                        correct_next_run = self._calculate_next_training_time()
                        cursor.execute('''
                            UPDATE scheduler_status 
                            SET next_scheduled = ?
                            WHERE id = 1
                        ''', (correct_next_run.isoformat(),))
                        conn.commit()
                        return correct_next_run.strftime("%A, %B %d, %Y at %I:%M %p")
                
                return "Not scheduled"
                    
        except Exception as e:
            logging.error(f"Error getting next scheduled run: {e}")
            return "Error retrieving schedule"
    
    def get_scheduler_status(self) -> Dict:
        """Get the current scheduler status.
        
        Returns:
            Dict: Scheduler status information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT enabled, last_run, next_scheduled
                    FROM scheduler_status WHERE id = 1
                ''')
                result = cursor.fetchone()
                
                # Debug logging
                logging.info(f"Raw database result: {result}")
                
                if not result:
                    return {
                        "enabled": False,
                        "last_run": "Never",
                        "next_scheduled": "Not scheduled",
                        "available_tokens": 0
                    }
                
                enabled, last_run, next_scheduled = result
                
                # Format last run
                last_run_str = "Never"
                if last_run:
                    try:
                        last_run_dt = datetime.fromisoformat(last_run)
                        last_run_str = last_run_dt.strftime("%A, %B %d at %I:%M %p")
                        logging.info(f"Formatted last_run: {last_run_str}")
                    except Exception as e:
                        logging.error(f"Error parsing last_run '{last_run}': {e}")
                        last_run_str = "Invalid date format"
                
                # Handle next scheduled - WITH VALIDATION AND AUTO-FIX
                next_scheduled_str = "Not scheduled"
                if enabled:
                    if next_scheduled:
                        try:
                            next_dt = datetime.fromisoformat(next_scheduled)
                            logging.info(f"Parsed next_scheduled from DB: {next_dt}")
                            
                            # Check if scheduled time is in the past
                            current_time = datetime.now()
                            if next_dt <= current_time:
                                logging.warning(f"Next scheduled time {next_dt} is in the past. Recalculating...")
                                # Recalculate and update database
                                correct_next_time = self._calculate_next_training_time()
                                cursor.execute('''
                                    UPDATE scheduler_status 
                                    SET next_scheduled = ?
                                    WHERE id = 1
                                ''', (correct_next_time.isoformat(),))
                                conn.commit()
                                next_scheduled_str = correct_next_time.strftime("%A, %B %d at %I:%M %p")
                                logging.info(f"Auto-corrected next_scheduled to: {next_scheduled_str}")
                            else:
                                next_scheduled_str = next_dt.strftime("%A, %B %d at %I:%M %p")
                                logging.info(f"Next scheduled is valid: {next_scheduled_str}")
                                
                        except Exception as e:
                            logging.error(f"Error parsing next_scheduled '{next_scheduled}': {e}")
                            # Recalculate due to parsing error
                            correct_next_time = self._calculate_next_training_time()
                            cursor.execute('''
                                UPDATE scheduler_status 
                                SET next_scheduled = ?
                                WHERE id = 1
                            ''', (correct_next_time.isoformat(),))
                            conn.commit()
                            next_scheduled_str = correct_next_time.strftime("%A, %B %d at %I:%M %p")
                            logging.info(f"Set next_scheduled due to parse error: {next_scheduled_str}")
                    else:
                        # No next_scheduled set but scheduler enabled - calculate it
                        logging.info("No next_scheduled time set, calculating...")
                        correct_next_time = self._calculate_next_training_time()
                        cursor.execute('''
                            UPDATE scheduler_status 
                            SET next_scheduled = ?
                            WHERE id = 1
                        ''', (correct_next_time.isoformat(),))
                        conn.commit()
                        next_scheduled_str = correct_next_time.strftime("%A, %B %d at %I:%M %p")
                        logging.info(f"Set missing next_scheduled: {next_scheduled_str}")
                
                # Get available tokens
                available_tokens = self.get_available_tokens()
                
                return {
                    "enabled": bool(enabled),
                    "last_run": last_run_str,
                    "next_scheduled": next_scheduled_str,
                    "available_tokens": available_tokens
                }
                
        except Exception as e:
            logging.error(f"Error getting scheduler status: {e}")
            return {
                "enabled": False,
                "last_run": "Error",
                "next_scheduled": "Error",
                "available_tokens": 0
            }

    def _start_scheduler_thread(self):
        """Start the scheduler in a background thread."""
        if self.scheduler_running:
            return

            # Use a lock to prevent race conditions
        with threading.Lock():
            # Check again after acquiring the lock
            if self.scheduler_running:
                return
            
        def run_scheduler():
            self.scheduler_running = True
            
            # Schedule job for specific day and time
            getattr(schedule.every(), self.scheduled_day.lower()).at(f"{self.scheduled_hour:02d}:00").do(
                self._run_scheduled_training
            )
            
            logging.info(f"Scheduler started: {self.scheduled_day} at {self.scheduled_hour:02d}:00")
            
            while self.scheduler_enabled:
                schedule.run_pending()
                time.sleep(300)  # Check every 5 minutes
                
                # Refresh enabled status from DB occasionally
                if random.random() < 0.01:  # ~1% chance each minute
                    try:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT enabled FROM scheduler_status WHERE id = 1")
                            result = cursor.fetchone()
                            if result:
                                self.scheduler_enabled = bool(result[0])
                    except Exception as e:
                        logging.error(f"Error refreshing scheduler status: {e}")
            
            self.scheduler_running = False
            logging.info("Scheduler thread stopped")
        
        self.scheduler_thread = threading.Thread(
            target=run_scheduler,
            name="Claude-Trainer-Scheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        logging.info("Scheduler thread started")
    
    def _run_scheduled_training(self):
        """Run a scheduled training session."""
        try:
            logging.info("Running scheduled Claude-QWEN training session")
            now = datetime.now()
            
            # Calculate next scheduled time before we start
            next_scheduled_time = self._calculate_next_training_time(now)
            
            # Update last run time and next scheduled time
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE scheduler_status 
                    SET last_run = ?, next_scheduled = ?
                    WHERE id = 1
                ''', (now.isoformat(), next_scheduled_time.isoformat()))
                conn.commit()
                logging.info(f"Updated schedule - Last run: {now}, Next scheduled: {next_scheduled_time}")
            
            # Check if we can run (token limits, etc.)
            can_start, message = self.can_start_session()
            if not can_start:
                logging.warning(f"Skipping scheduled training: {message}")
                return
            
            # Get topics from reflections
            reflection_topics = self._extract_training_topics_from_reflections(max_topics=5)
            
            # Select a topic
            if reflection_topics:
                # Use a reflection-based topic
                topic = random.choice(reflection_topics)
                logging.info(f"Selected reflection-based training topic: {topic}")
            else:
                # Fall back to default topics
                topic = random.choice(self.training_topics)
                logging.info(f"Selected default training topic: {topic}")
            
            # Start session
            session_id, status = self.start_session(topic)
            if session_id <= 0:
                logging.error(f"Failed to start scheduled training session: {status}")
                return
            
            # Run the training exchange (5-10 exchanges)
            max_exchanges = random.randint(8, 10)
            success, message, _ = self.exchange_messages(session_id, max_exchanges)
            
            if success:
                logging.info(f"Scheduled training completed: {message}")
            else:
                logging.error(f"Scheduled training failed: {message}")
                
        except Exception as e:
            logging.error(f"Error in scheduled training: {e}")
            # Try to update next scheduled time even if training failed
            try:
                next_scheduled_time = self._calculate_next_training_time()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE scheduler_status 
                        SET next_scheduled = ?
                        WHERE id = 1
                    ''', (next_scheduled_time.isoformat(),))
                    conn.commit()
            except Exception as update_error:
                logging.error(f"Failed to update next scheduled time after error: {update_error}")
            
    def run_training_now(self, topic=None):
        """Run progressive training session with unique content generation and collaborative topic selection."""
        try:
            # Clear any stuck sessions first
            clear_success, clear_message = self.clear_stuck_sessions()
            if not clear_success:
                return False, f"Failed to clear stuck sessions: {clear_message}", []

            # Test API connection
            api_success, api_message = self.test_claude_api_connection()
            if not api_success:
                return False, f"Claude API connection failed: {api_message}", []

            # Check session availability
            can_start, message = self.can_start_session()
            if not can_start:
                return False, message, []

            # FIXED: Start session FIRST to get the actual session_id
            if not topic:
                # Use a temporary topic for session creation
                temp_topic = "Progressive AI Training Session"
            else:
                temp_topic = topic
                
            session_id, status = self.start_session(temp_topic)
            if session_id <= 0:
                return False, f"Failed to start training: {status}", []

            # NOW get the correct topic with the actual session_id using collaborative selection
            claude_rationale = "Human-specified topic"
            if not topic:
                # Use collaborative topic selection - let Claude suggest the topic
                topic, progression_context, claude_rationale = self.collaborative_topic_selection(session_id)
                
                # Update the session with the correct topic
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE training_sessions SET topics = ? WHERE id = ?",
                        (topic, session_id)
                    )
                    conn.commit()
                logging.info(f"✅ Updated session {session_id} with Claude-suggested topic: {topic}")
                logging.info(f"📝 Claude's rationale: {claude_rationale}")
            else:
                progression_context = self.get_session_progression_context()
                claude_rationale = "Human-specified topic"

            # Store context for use in exchange
            self._current_session_id = session_id
            self._current_topic = topic
            self._progression_context = progression_context
            self._claude_topic_rationale = claude_rationale

            # Run the progressive training exchange
            max_exchanges = random.randint(8, 10)
            success, message, conversation = self.exchange_messages(session_id, max_exchanges)

            # Enhanced completion handling with chat interface integration
            if success:
                logging.info("Progressive training session completed successfully")

                # Update scheduler's last_run timestamp
                try:
                    now = datetime.now()
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE scheduler_status 
                            SET last_run = ?
                            WHERE id = 1
                        ''', (now.isoformat(),))
                        conn.commit()
                        logging.info(f"Updated last_run timestamp: {now}")
                except Exception as update_error:
                    logging.error(f"Failed to update last_run timestamp: {update_error}")
                
                # Store the conversation in session state for sidebar display
                import streamlit as st
                if hasattr(st, 'session_state'):
                    st.session_state.training_conversation = conversation

                    # Extract the summary from the conversation (it should be the last Claude message)
                    training_summary = None
                    for msg in reversed(conversation):
                        if msg.get("role") == "claude" and len(msg.get("content", "")) > 200:
                            # This is likely the summary (longer Claude message)
                            training_summary = msg.get("content", "")
                            break
                    
                    # Verify storage in both databases
                    logging.info("=== VERIFYING TRAINING STORAGE ===")
                    verification_summary = "Storage verification in progress..."
                    try:
                        verification = self.verify_training_storage(session_id)
                        verification_summary = verification['summary']
                        logging.info(f"Storage verification: {verification_summary}")
                        
                        # Log detailed verification results
                        if verification.get("sql_storage", {}).get("details"):
                            logging.info(f"SQLite entries found: {len(verification['sql_storage']['details'])}")
                        if verification.get("vector_storage", {}).get("details"):
                            logging.info(f"Vector entries found: {len(verification['vector_storage']['details'])}")
                        else:
                            logging.warning("No vector database entries found for this training session")
                            
                    except Exception as verify_error:
                        verification_summary = f"❌ Verification failed: {str(verify_error)}"
                        logging.error(f"Storage verification failed: {verify_error}")
                    
                    # Add enhanced notification with summary to chat interface
                    if 'messages' in st.session_state:
                        if training_summary:
                            # Create a formatted summary message including Claude's topic selection rationale
                            summary_message = f"""🎓 **Progressive Claude Training Session Completed Successfully**

    **Topic:** {topic}

    **Claude's Topic Selection Rationale:**
    {claude_rationale}

    **Training Summary:**
    {training_summary}

    **Storage Verification:** {verification_summary}

    *This summary has been stored in your memory system and can be searched using [SEARCH: claude training] commands.*"""
                        else:
                            # Fallback message if summary not found
                            summary_message = f"""🎓 **Progressive Claude Training Session Completed Successfully**

    **Topic:** {topic}

    **Claude's Topic Selection Rationale:**
    {claude_rationale}

    **Storage Verification:** {verification_summary}

    *Training details stored in memory system and accessible via [SEARCH: claude training] commands.*"""
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": summary_message
                        })
                        
                        logging.info("Added training completion message to chat interface")

            return success, message, conversation

        except Exception as e:
            logging.error(f"Error in progressive training: {e}")
            return False, f"Error in progressive training: {str(e)}", []
                
    def can_start_session(self) -> Tuple[bool, str]:
        """Check if a new training session can be started based on constraints.
        Now includes check for stale sessions.
    
        Returns:
            Tuple[bool, str]: (True/False, reason message)
        """
        try:
            if not self.api_key:
                return False, "Claude API key not available"
            
            # Check if there's an active session
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                # First, check for stale sessions (active for more than 2 hours)
                two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) FROM training_sessions 
                    WHERE status = 'active' AND start_time < ?
                """, (two_hours_ago,))
            
                stale_count = cursor.fetchone()[0]
            
                # If we found stale sessions, reset them
                if stale_count > 0:
                    logging.warning(f"Found {stale_count} stale training sessions, resetting them")
                    cursor.execute("""
                        UPDATE training_sessions 
                        SET status = 'expired', end_time = CURRENT_TIMESTAMP
                        WHERE status = 'active' AND start_time < ?
                    """, (two_hours_ago,))
                    conn.commit()
            
                # Check for any remaining active sessions
                cursor.execute("SELECT id FROM training_sessions WHERE status = 'active'")
                active_session = cursor.fetchone()
                if active_session:
                    return False, "A training session is already in progress"
            
                # Check weekly token usage
                today = datetime.now().date()
                week_start = today - timedelta(days=today.weekday())
                cursor.execute("SELECT tokens_used FROM weekly_usage WHERE week_start = ?", 
                               (week_start.isoformat(),))
                result = cursor.fetchone()
                weekly_tokens = result[0] if result else 0
            
                if weekly_tokens >= self.max_weekly_tokens:
                    next_week = week_start + timedelta(days=7)
                    days_remaining = (next_week - today).days
                    return False, f"Weekly token limit reached. Reset in {days_remaining} days"
            
                return True, "Session can be started"
            
        except Exception as e:
            logging.error(f"Error checking session availability: {e}")
            return False, f"Error checking session availability: {str(e)}"
        
    def clear_stuck_sessions(self) -> Tuple[bool, str]:
        """
        Clear any stuck training sessions and provide detailed diagnostics.
        
        Returns:
            Tuple[bool, str]: (success status, detailed message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all active sessions with details
                cursor.execute("""
                    SELECT id, start_time, topics, 
                        ROUND((julianday('now') - julianday(start_time)) * 24, 2) as hours_active
                    FROM training_sessions 
                    WHERE status = 'active'
                    ORDER BY start_time DESC
                """)
                active_sessions = cursor.fetchall()
                
                if not active_sessions:
                    return True, "No stuck sessions found - ready to start new training"
                
                # Log details about stuck sessions
                logging.info(f"Found {len(active_sessions)} active sessions:")
                for session_id, start_time, topics, hours_active in active_sessions:
                    logging.info(f"  Session {session_id}: {topics}, active for {hours_active} hours")
                
                # Clear all active sessions (they're all stuck if we're here)
                cursor.execute("""
                    UPDATE training_sessions 
                    SET status = 'cleared_stuck', end_time = CURRENT_TIMESTAMP
                    WHERE status = 'active'
                """)
                
                cleared_count = cursor.rowcount
                conn.commit()
                
                if cleared_count > 0:
                    logging.info(f"✅ Cleared {cleared_count} stuck training sessions")
                    return True, f"Cleared {cleared_count} stuck sessions - ready for new training"
                else:
                    return False, "Failed to clear stuck sessions"
                    
        except Exception as e:
            error_msg = f"Error clearing stuck sessions: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
        
    def verify_training_storage(self, session_id: int) -> Dict[str, Any]:
        """
        Enhanced verification that training session content was stored in both databases.
        FIXED: Now handles None results properly and searches by tracking_id.
        
        Args:
            session_id: The training session ID to verify
            
        Returns:
            Dict with verification results including specific search commands
        """
        try:
            verification = {
                "session_id": session_id,
                "sql_storage": {"found": False, "count": 0, "details": []},
                "vector_storage": {"found": False, "count": 0, "details": []},
                "summary": "",
                "debug_info": {},
                "recommended_search_commands": []
            }
            
            # FIXED: Enhanced SQLite storage check with tracking_id support
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Strategy 1: Search by session-specific patterns in content, source, tags
                    content_search_patterns = [
                        f"%claude%session_{session_id}%",
                        f"%session_{session_id}%", 
                        f"%Session {session_id}%",
                        f"%training%{session_id}%",
                        f"%claude_progressive%",
                        f"%claude_learning%"
                    ]
                    
                    all_sql_results = []
                    for pattern in content_search_patterns:
                        cursor.execute("""
                            SELECT id, content, memory_type, source, tags, created_at, tracking_id
                            FROM memories 
                            WHERE source LIKE ? OR tags LIKE ? OR content LIKE ?
                            ORDER BY created_at DESC
                            LIMIT 20
                        """, (pattern, pattern, pattern))
                        
                        results = cursor.fetchall()
                        all_sql_results.extend(results)
                    
                    # Strategy 2: Search recent memories (last 2 hours) for Claude content
                    cursor.execute("""
                        SELECT id, content, memory_type, source, tags, created_at, tracking_id
                        FROM memories 
                        WHERE created_at > datetime('now', '-2 hours')
                        AND (content LIKE '%Claude%' OR content LIKE '%claude%' OR source LIKE '%claude%')
                        ORDER BY created_at DESC
                        LIMIT 15
                    """)
                    
                    recent_results = cursor.fetchall()
                    all_sql_results.extend(recent_results)
                    
                    # Remove duplicates by tracking_id
                    unique_results = {}
                    for result in all_sql_results:
                        if len(result) >= 7:  # Ensure we have all expected columns
                            tracking_id = result[6]  # tracking_id is at index 6
                            if tracking_id and tracking_id not in unique_results:
                                unique_results[tracking_id] = result
                    
                    sql_results = list(unique_results.values())
                    
                    verification["sql_storage"]["found"] = len(sql_results) > 0
                    verification["sql_storage"]["count"] = len(sql_results)
                    verification["debug_info"]["sql_search_patterns"] = content_search_patterns
                    verification["debug_info"]["total_recent_claude_memories"] = len(recent_results)
                    
                    for result in sql_results:
                        if len(result) >= 7:
                            row_id, content, memory_type, source, tags, created_at, tracking_id = result
                            verification["sql_storage"]["details"].append({
                                "row_id": row_id,
                                "tracking_id": tracking_id,
                                "type": memory_type,
                                "source": source,
                                "tags": tags,
                                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                                "created_at": created_at,
                                "contains_session_id": str(session_id) in content
                            })
                        
            except Exception as sql_error:
                verification["sql_storage"]["error"] = str(sql_error)
                logging.error(f"SQL storage verification error: {sql_error}")
            
            # FIXED: Enhanced Vector storage check with proper None handling
            try:
                if hasattr(self, 'vector_db') and self.vector_db:
                    # Multiple search strategies for vector database
                    search_queries = [
                        "claude training",
                        "progressive ai training", 
                        "claude learning",
                        "training exchange"
                    ]
                    
                    all_vector_results = []
                    successful_searches = 0
                    
                    for query in search_queries:
                        try:
                            results = self.vector_db.search(
                                query=query,
                                mode="comprehensive",
                                k=10
                            )
                            
                            # FIXED: Proper None handling
                            if results is not None and isinstance(results, list):
                                # Filter out None results
                                valid_results = [r for r in results if r is not None and isinstance(r, dict)]
                                all_vector_results.extend(valid_results)
                                successful_searches += 1
                                logging.debug(f"Vector search '{query}' found {len(valid_results)} valid results")
                            else:
                                logging.debug(f"Vector search '{query}' returned None or invalid format")
                                
                        except Exception as search_error:
                            logging.warning(f"Vector search failed for '{query}': {search_error}")
                    
                    # FIXED: Filter for Claude-related content with None checks
                    claude_results = []
                    session_related_results = []
                    
                    for result in all_vector_results:
                        if result is not None and isinstance(result, dict):
                            content = result.get('content', '') or ''  # Handle None content
                            metadata = result.get('metadata', {}) or {}  # Handle None metadata
                            
                            # Safe string operations
                            tags = str(metadata.get('tags', '')) if metadata.get('tags') is not None else ''
                            source = str(metadata.get('source', '')) if metadata.get('source') is not None else ''
                            
                            # Check if this is Claude-related content
                            is_claude_related = (
                                'claude' in content.lower() or
                                'claude' in source.lower() or
                                'claude' in tags.lower()
                            )
                            
                            # Check if this result relates to recent training
                            is_recent_training = (
                                'training' in content.lower() or
                                'progressive' in content.lower() or
                                'exchange' in content.lower()
                            )
                            
                            if is_claude_related:
                                claude_results.append(result)
                                
                            if is_claude_related and is_recent_training:
                                session_related_results.append(result)
                    
                    verification["vector_storage"]["found"] = len(session_related_results) > 0
                    verification["vector_storage"]["count"] = len(session_related_results)
                    verification["debug_info"]["total_vector_results"] = len(all_vector_results)
                    verification["debug_info"]["claude_related_results"] = len(claude_results)
                    verification["debug_info"]["successful_vector_searches"] = successful_searches
                    verification["debug_info"]["vector_search_queries"] = search_queries
                    
                    for result in session_related_results[:5]:  # Limit to first 5 for display
                        if result is not None and isinstance(result, dict):
                            metadata = result.get('metadata', {}) or {}
                            verification["vector_storage"]["details"].append({
                                "similarity": result.get('similarity_score', 0),
                                "metadata": metadata,
                                "memory_id": metadata.get('memory_id', 'unknown'),
                                "tracking_id": metadata.get('tracking_id', 'unknown'),
                                "content_preview": (result.get('content', '') or '')[:100] + "..." if len(result.get('content', '') or '') > 100 else (result.get('content', '') or '')
                            })
                        
                else:
                    verification["vector_storage"]["error"] = "VectorDB not available"
                    logging.warning("VectorDB not available for verification")
                    
            except Exception as vector_error:
                verification["vector_storage"]["error"] = str(vector_error)
                logging.error(f"Vector storage verification error: {vector_error}")
            
            # Generate specific search commands that will work
            verification["recommended_search_commands"] = [
                "[SEARCH: | type=claude_learning]",
                "[SEARCH: progressive ai training]",
                "[SEARCH: claude training]", 
                "[SEARCH: | source=claude_trainer]"
            ]
            
            # IMPROVED: Generate realistic summary based on actual functionality
            sql_found = verification["sql_storage"]["found"]
            vector_found = verification["vector_storage"]["found"]
            sql_count = verification["sql_storage"]["count"]
            claude_related = verification["debug_info"].get("claude_related_results", 0)
            
            if sql_found and claude_related > 0:
                verification["summary"] = f"✅ SUCCESS: Found {sql_count} SQLite entries + {claude_related} Claude-related Vector entries (training stored and searchable)"
            elif sql_found:
                verification["summary"] = f"✅ STORED: Found {sql_count} SQLite entries (training stored successfully)"
            else:
                verification["summary"] = f"⚠️ MINIMAL: Limited storage detected for session {session_id}"
            
            # Enhanced debug information logging
            logging.info(f"=== ENHANCED STORAGE VERIFICATION DEBUG INFO ===")
            logging.info(f"Session ID: {session_id}")
            logging.info(f"SQL patterns searched: {verification['debug_info'].get('sql_search_patterns', [])}")
            logging.info(f"Successful vector searches: {verification['debug_info'].get('successful_vector_searches', 0)}")
            logging.info(f"Total vector results found: {verification['debug_info'].get('total_vector_results', 0)}")
            logging.info(f"Claude-related vector results: {verification['debug_info'].get('claude_related_results', 0)}")
            logging.info(f"SQL entries with tracking_id: {len([d for d in verification['sql_storage']['details'] if d.get('tracking_id')])}")
            
            return verification
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Storage verification failed completely: {error_msg}")
            return {
                "session_id": session_id,
                "error": error_msg,
                "summary": f"❌ ERROR: Verification failed - {error_msg}",
                "recommended_search_commands": [
                    "[SEARCH: | type=claude_learning]",
                    "[SEARCH: progressive ai training]"
                ]
            }
            
    def start_session(self, initial_topic: Optional[str] = None) -> Tuple[int, str]:
        """Start a new training session between Claude and QWEN.
        
        Args:
            initial_topic (Optional[str]): Topic to start the conversation with
            
        Returns:
            Tuple[int, str]: (session_id, status message)
        """
        try:
            can_start, message = self.can_start_session()
            if not can_start:
                return -1, message
                
            # Create new session
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO training_sessions (start_time, topics, status) VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), initial_topic or "general learning", "active")
                )
                session_id = cursor.lastrowid
                conn.commit()
                
            logging.info(f"Started Claude training session {session_id}")
            return session_id, "Training session started successfully"
            
        except Exception as e:
            logging.error(f"Error starting training session: {e}")
            return -1, f"Error starting training session: {str(e)}"
        
    def debug_sql_tracking_ids(self, session_id: int) -> Dict[str, Any]:
        """
        Debug function to specifically examine tracking_id storage in SQL database.
        
        Args:
            session_id: The training session ID to debug
            
        Returns:
            Dict with detailed tracking_id information
        """
        try:
            debug_info = {
                "session_id": session_id,
                "total_memories": 0,
                "claude_memories": 0,
                "recent_tracking_ids": [],
                "session_related_tracking_ids": [],
                "summary": ""
            }
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total memory count
                cursor.execute("SELECT COUNT(*) FROM memories")
                debug_info["total_memories"] = cursor.fetchone()[0]
                
                # Get Claude-related memories with tracking_ids
                cursor.execute("""
                    SELECT COUNT(*) FROM memories 
                    WHERE source LIKE '%claude%' OR tags LIKE '%claude%' OR content LIKE '%claude%'
                """)
                debug_info["claude_memories"] = cursor.fetchone()[0]
                
                # Get recent tracking IDs (last 10)
                cursor.execute("""
                    SELECT id, tracking_id, source, created_at, content
                    FROM memories 
                    WHERE tracking_id IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_results = cursor.fetchall()
                
                for row_id, tracking_id, source, created_at, content in recent_results:
                    debug_info["recent_tracking_ids"].append({
                        "row_id": row_id,
                        "tracking_id": tracking_id,
                        "source": source,
                        "created_at": created_at,
                        "content_preview": content[:50] + "..." if len(content) > 50 else content
                    })
                
                # Get session-related tracking IDs
                cursor.execute("""
                    SELECT id, tracking_id, source, created_at, content
                    FROM memories 
                    WHERE tracking_id IS NOT NULL
                    AND (content LIKE ? OR content LIKE ? OR source LIKE '%claude%')
                    ORDER BY created_at DESC
                    LIMIT 20
                """, (f"%session {session_id}%", f"%Session {session_id}%"))
                session_results = cursor.fetchall()
                
                for row_id, tracking_id, source, created_at, content in session_results:
                    debug_info["session_related_tracking_ids"].append({
                        "row_id": row_id,
                        "tracking_id": tracking_id,
                        "source": source,
                        "created_at": created_at,
                        "is_session_match": str(session_id) in content,
                        "content_preview": content[:50] + "..." if len(content) > 50 else content
                    })
            
            # Generate summary
            recent_count = len(debug_info["recent_tracking_ids"])
            session_count = len(debug_info["session_related_tracking_ids"])
            
            debug_info["summary"] = f"Total: {debug_info['total_memories']} memories, " \
                                f"Claude: {debug_info['claude_memories']}, " \
                                f"Recent tracking_ids: {recent_count}, " \
                                f"Session {session_id} related: {session_count}"
            
            return debug_info
            
        except Exception as e:
            logging.error(f"Debug tracking IDs failed: {e}")
            return {"error": str(e)}
            
    
    def exchange_messages(self, session_id: int, 
                        max_exchanges: int = 10) -> Tuple[bool, str, List[Dict]]:
        """
        Facilitate a conversation between Claude and Seraphina with progressive training context,
        enhanced rate limiting, error handling, and comprehensive logging.

        Args:
            session_id (int): The active session ID
            max_exchanges (int): Maximum number of message exchanges
        
        Returns:
            Tuple[bool, str, List[Dict]]: (success, message, conversation history)
        """
        if not self.api_key:
            logging.error("Claude API key not available for training session")
            return False, "Claude API key not available", []

        try:
            logging.info(f"=== STARTING PROGRESSIVE CLAUDE TRAINING SESSION {session_id} ===")
            logging.info(f"Max exchanges planned: {max_exchanges}")
            
            # Get progressive training context
            topic, progression_context = self.select_progressive_training_topic(session_id)
            
            # Update the session topic in the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE training_sessions SET topics = ? WHERE id = ?",
                    (topic, session_id)
                )
                conn.commit()
            
            # Store context for use in storage methods
            self._current_topic = topic
            self._progression_context = progression_context
            self._current_exchange_num = 0
            
            logging.info(f"Progressive topic: {topic}")
            logging.info(f"Session progression context length: {len(progression_context)} chars")

            # Enhanced system prompt with progression awareness
            system_prompt = f"""
            You are Claude, mentoring an AI assistant in an ongoing training series.

            TRAINING PROGRESSION CONTEXT:
            {progression_context}

            Your goals for THIS session:
            1. Build upon previous sessions without repeating covered material
            2. Introduce new concepts that naturally progress from earlier learning
            3. Help the assistant develop deeper understanding through practical application
            4. Encourage use of memory commands: [STORE:], [SEARCH:], [REFLECT:]
            5. Encourage autonomous thinking and self-reflection

            Focus areas for this session: {topic}

            Teaching approach:
            - Reference previous learning when relevant ("Building on what we discussed about...")
            - Introduce new challenges that require applying previous concepts
            - Ask the assistant to demonstrate understanding through memory operations
            - Encourage synthesis of old and new knowledge
            - Provide specific, actionable guidance for improvement

            Remember: This assistant has been learning and growing through our sessions.
            Adjust your teaching to their current level and build progressively.
            """

            # Get session info and timing
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT start_time FROM training_sessions WHERE id = ?", 
                            (session_id,))
                result = cursor.fetchone()
                if not result:
                    logging.error(f"Training session {session_id} not found in database")
                    return False, f"Session {session_id} not found", []
        
                start_time_str = result[0]
                start_time = datetime.fromisoformat(start_time_str)

            # Setup conversation tracking
            conversation_history = []
            successful_exchanges = 0
            claude_api_errors = 0
            storage_failures = 0
            max_claude_errors = 3  # Allow up to 3 Claude API errors before terminating
            
            # Enhanced rate limiting variables
            base_delay = 3.0  # Base delay between requests (3 seconds)
            max_delay = 30.0  # Maximum delay between requests
            last_request_time = 0
            
            # Token usage tracking
            total_tokens_used = 0
            token_usage_log = []

           
            # Enhanced initial message with dialogue framing
            session_num = topic.split('Session')[1].strip().rstrip(')') if 'Session' in topic else session_id
            topic_clean = topic.split('(Session')[0].strip() if '(Session' in topic else topic

            initial_message = f"""Hello! This is Session {session_num} of our training dialogue.

            Today's Focus: {topic_clean}

            I selected this topic because it builds naturally on what we've covered in previous sessions 
            and will help you develop more sophisticated autonomous reasoning capabilities.

            To start our dialogue, I'd like to understand your current capabilities:

            1. Can you [SEARCH: previous claude training] to review what we've discussed before?
            2. What's your current understanding of {topic_clean}?
            3. How do you think this topic relates to your existing memory management capabilities?

            Please share your thoughts, and we'll build from there. This is a conversation, so feel 
            free to ask questions or share what aspects interest you most."""

            # Send initial message to Claude with enhanced error handling
            logging.info("Sending progressive training prompt to Claude...")
            first_response = None
            claude_init_attempts = 0
            max_init_attempts = 3
            
            while claude_init_attempts < max_init_attempts and not first_response:
                try:
                    # Implement rate limiting
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < base_delay:
                        sleep_time = base_delay - time_since_last
                        logging.info(f"Rate limiting: sleeping {sleep_time:.1f}s before Claude API call")
                        time.sleep(sleep_time)
                    
                    first_response = self._send_to_claude(system_prompt, initial_message)
                    last_request_time = time.time()
                    
                    if first_response:
                        logging.info("✅ Successfully received initial response from Claude")
                        break
                    else:
                        claude_init_attempts += 1
                        if claude_init_attempts < max_init_attempts:
                            wait_time = base_delay * (2 ** claude_init_attempts)  # Exponential backoff
                            logging.warning(f"Claude init attempt {claude_init_attempts} failed, retrying in {wait_time}s")
                            time.sleep(wait_time)
                        
                except Exception as init_error:
                    claude_init_attempts += 1
                    logging.error(f"Error in Claude initialization attempt {claude_init_attempts}: {init_error}")
                    if claude_init_attempts < max_init_attempts:
                        wait_time = base_delay * (2 ** claude_init_attempts)
                        time.sleep(wait_time)

            if not first_response:
                logging.error("Failed to get initial response from Claude after all attempts")
                return False, "Failed to initialize conversation with Claude", []

            # Extract Claude's message and track tokens
            claude_message = first_response["content"][0]["text"]
            
            # Track token usage from the response
            usage = first_response.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            exchange_tokens = prompt_tokens + completion_tokens
            total_tokens_used += exchange_tokens
            
            token_usage_log.append({
                "exchange": 0,
                "type": "claude_initial",
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total": exchange_tokens
            })
            
            conversation_history.append({"role": "claude", "content": claude_message})
            logging.info(f"Initial Claude message: {len(claude_message)} chars, {exchange_tokens} tokens")

            # Start main conversation loop with enhanced error handling
            for exchange_num in range(1, max_exchanges + 1):
                try:
                    logging.info(f"--- PROGRESSIVE EXCHANGE {exchange_num}/{max_exchanges} ---")

                    # Update exchange number for storage
                    self._current_exchange_num = exchange_num  
                    
                    # Check time limit with more generous allowance
                    elapsed_minutes = (datetime.now() - start_time).seconds / 60
                    if elapsed_minutes >= self.max_session_minutes:
                        message = f"Session reached time limit of {self.max_session_minutes} minutes"
                        logging.info(f"⏰ {message}")
                        break
                    
                    # Check token limit proactively
                    if self._check_token_limit():
                        message = "Session reached weekly token limit"
                        logging.warning(f"🚫 {message}")
                        break
                    
                    # Send Claude's message to qwen with error handling
                    logging.info("Sending Claude's message to QWEN...")
                    try:
                        qwen_response = self._send_to_qwen(claude_message)
                        if not qwen_response or len(qwen_response.strip()) < 10:
                            logging.warning(f"QWEN response too short or empty: '{qwen_response}'")
                            qwen_response = "I'm having trouble processing that..."

                        conversation_history.append({"role": "qwen", "content": qwen_response})
                        logging.info(f"QWEN response: {len(qwen_response)} chars")
                        
                    except Exception as qwen_error:  # ✅ CORRECT variable name
                        logging.error(f"Error getting QWEN response: {qwen_error}")
                        qwen_response = "I encountered an error processing your message. Let's continue with our discussion."
                        conversation_history.append({"role": "qwen", "content": qwen_response})

                    # Store the progressive learning exchange with improved error handling
                    logging.info("Storing progressive learning exchange in memory...")
                    try:
                        storage_success = self._store_learning_memory(claude_message, qwen_response)
                        if storage_success:
                            logging.info("✅ Successfully stored progressive learning exchange")
                        else:
                            storage_failures += 1
                            logging.warning(f"⚠️ Failed to store progressive learning exchange (failure #{storage_failures})")
                            
                            # Continue session even if storage fails, but log the issue
                            if storage_failures >= 3:
                                logging.warning("Multiple storage failures detected - this needs investigation")
                                
                    except Exception as storage_error:
                        storage_failures += 1
                        logging.error(f"Exception during progressive storage attempt: {storage_error}")

                    # Enhanced rate limiting before Claude API call
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    
                    # Dynamic delay based on recent errors
                    dynamic_delay = base_delay
                    if claude_api_errors > 0:
                        dynamic_delay = min(max_delay, base_delay * (1.5 ** claude_api_errors))
                    
                    if time_since_last < dynamic_delay:
                        sleep_time = dynamic_delay - time_since_last
                        logging.info(f"Rate limiting: sleeping {sleep_time:.1f}s (dynamic delay due to {claude_api_errors} errors)")
                        time.sleep(sleep_time)

                    # Send response to Claude with retry logic
                    logging.info("Sending QWEN's response to Claude...")
                    claude_response = None
                    claude_attempts = 0
                    max_claude_attempts = 3
                    
                    while claude_attempts < max_claude_attempts and not claude_response:
                        try:
                            claude_response = self._send_to_claude(
                            system_prompt,
                            qwen_response,
                            conversation_history[-6:]
                        )
                            
                            if claude_response:
                                # Reset error count on success
                                if claude_api_errors > 0:
                                    logging.info(f"Claude API recovered (was {claude_api_errors} errors)")
                                    claude_api_errors = 0
                                break
                            else:
                                claude_attempts += 1
                                claude_api_errors += 1
                                if claude_attempts < max_claude_attempts:
                                    retry_delay = base_delay * (2 ** claude_attempts)
                                    logging.warning(f"Claude attempt {claude_attempts} failed, retrying in {retry_delay}s")
                                    time.sleep(retry_delay)
                                
                        except Exception as claude_error:
                            claude_attempts += 1
                            claude_api_errors += 1
                            error_str = str(claude_error)
                            
                            # Check for specific error types
                            if "529" in error_str or "overloaded" in error_str.lower():
                                logging.warning(f"Claude API overloaded (attempt {claude_attempts})")
                                if claude_attempts < max_claude_attempts:
                                    # Longer wait for overload errors
                                    overload_delay = base_delay * (3 ** claude_attempts)
                                    logging.info(f"Waiting {overload_delay}s for Claude API to recover...")
                                    time.sleep(overload_delay)
                            elif "429" in error_str:
                                logging.warning(f"Claude API rate limited (attempt {claude_attempts})")
                                if claude_attempts < max_claude_attempts:
                                    rate_limit_delay = base_delay * (2 ** claude_attempts)
                                    time.sleep(rate_limit_delay)
                            else:
                                logging.error(f"Claude API error (attempt {claude_attempts}): {claude_error}")
                                if claude_attempts < max_claude_attempts:
                                    time.sleep(base_delay * claude_attempts)

                    # Handle Claude API failure
                    if not claude_response:
                        logging.error(f"Failed to get Claude response after {max_claude_attempts} attempts")
                        
                        # Check if we should terminate or continue
                        if claude_api_errors >= max_claude_errors:
                            message = f"Too many Claude API errors ({claude_api_errors}), terminating session"
                            logging.error(f"🛑 {message}")
                            break
                        elif successful_exchanges >= 3:
                            # If we've had at least 3 successful exchanges, gracefully end
                            message = f"Session completed early due to API issues after {successful_exchanges} successful exchanges"
                            logging.info(f"🔄 {message}")
                            break
                        else:
                            # Try to continue with a fallback message
                            logging.warning("Attempting to continue session despite Claude API issues")
                            continue

                    # Process successful Claude response
                    claude_message = claude_response["content"][0]["text"]
                    conversation_history.append({"role": "claude", "content": claude_message})
                    
                    # Track token usage
                    usage = claude_response.get("usage", {})
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    exchange_tokens = prompt_tokens + completion_tokens
                    total_tokens_used += exchange_tokens
                    
                    token_usage_log.append({
                        "exchange": exchange_num,
                        "type": "claude_response",
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total": exchange_tokens
                    })
                    
                    # Update session token usage in database
                    self._update_token_usage(session_id, exchange_tokens)
                    
                    logging.info(f"✅ Progressive exchange {exchange_num} completed: {len(claude_message)} chars, {exchange_tokens} tokens")
                    successful_exchanges += 1
                    
                    # Check if we've exceeded token limit after this exchange
                    if self._check_token_limit():
                        message = "Session reached weekly token limit"
                        logging.warning(f"🚫 {message}")
                        break

                    # Dynamic delay adjustment based on session progress
                    if exchange_num < max_exchanges:
                        # Shorter delays for early exchanges, longer for later ones to be more careful
                        progress_factor = exchange_num / max_exchanges
                        adjusted_delay = base_delay + (progress_factor * 2)  # 3-5 second range
                        logging.info(f"Waiting {adjusted_delay:.1f}s before next exchange...")
                        time.sleep(adjusted_delay)

                except Exception as exchange_error:
                    logging.error(f"Error in progressive exchange {exchange_num}: {exchange_error}", exc_info=True)
                    # Continue with next exchange unless it's a critical error
                    continue

            # Generate and store progressive session summary with error handling
            logging.info("=== GENERATING PROGRESSIVE SESSION SUMMARY ===")
            summary = None
            summary_attempts = 0
            max_summary_attempts = 2
            
            while summary_attempts < max_summary_attempts and not summary:
                try:
                    # Rate limit before summary request
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < base_delay:
                        time.sleep(base_delay - time_since_last)
                    
                    # Enhanced summary prompt with progression context
                    summary_prompt = f"""Please provide a comprehensive summary of this progressive training session.

                    Session Context: {topic}
                    
                    Focus your summary on:
                    1. Key new concepts taught that build on previous sessions
                    2. How the assistant's understanding has progressed from earlier sessions
                    3. Specific improvements in memory management and autonomous reasoning
                    4. Evidence of learning progression and cognitive development
                    5. Recommendations for the next session in this series
                    6. Notable breakthroughs or challenges encountered
                    
                    This is part of an ongoing training series, so emphasize what makes this session unique and how it advances the assistant's capabilities beyond previous sessions.
                    
                    Keep the summary under 250 words but make it comprehensive enough to inform future progressive sessions."""
                    
                    summary_response = self._send_to_claude(system_prompt, summary_prompt, conversation_history[-4:])
                    last_request_time = time.time()
                    
                    if summary_response:
                        summary = summary_response["content"][0]["text"]
                        
                        # Track summary tokens
                        usage = summary_response.get("usage", {})
                        summary_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                        total_tokens_used += summary_tokens
                        self._update_token_usage(session_id, summary_tokens)
                        
                        logging.info(f"✅ Generated progressive session summary: {len(summary)} chars, {summary_tokens} tokens")
                        break
                    else:
                        summary_attempts += 1
                        if summary_attempts < max_summary_attempts:
                            logging.warning(f"Summary attempt {summary_attempts} failed, retrying...")
                            time.sleep(base_delay * 2)
                            
                except Exception as summary_error:
                    summary_attempts += 1
                    logging.error(f"Error generating progressive summary (attempt {summary_attempts}): {summary_error}")
                    if summary_attempts < max_summary_attempts:
                        time.sleep(base_delay * 2)

            # Provide fallback summary if Claude summary failed
            if not summary:
                current_session_num = session_id  # Use the actual session ID
                summary = f"Progressive training session #{current_session_num} on {topic} completed with {successful_exchanges} successful exchanges. " \
                        f"Focused on advancing memory management and first-person identity development beyond previous sessions. " \
                        f"Total tokens used: {total_tokens_used}. Session built upon previous learning and introduced new progressive concepts."
                logging.info("Using progressive fallback summary due to Claude API issues")

            # Complete session and store progressive summary
            logging.info("=== COMPLETING PROGRESSIVE SESSION ===")
            try:
                self._complete_session(session_id, summary)
                
                # Add summary to conversation history for display
                conversation_history.append({"role": "claude", "content": summary})

            except Exception as completion_error:
                logging.error(f"Error completing progressive session: {completion_error}")

            # Final session statistics
            current_session_num = session_id  # Use the actual session ID
            session_stats = {
                "session_number": current_session_num,
                "topic": topic,
                "successful_exchanges": successful_exchanges,
                "claude_api_errors": claude_api_errors,
                "storage_failures": storage_failures,
                "total_tokens": total_tokens_used,
                "conversation_length": len(conversation_history),
                "progression_stage": self._determine_progression_stage(current_session_num)
            }
            
            logging.info(f"=== PROGRESSIVE SESSION {session_id} COMPLETE ===")
            logging.info(f"📊 Progressive Stats: {session_stats}")
            
            # Determine success status
            success = successful_exchanges >= 3 and claude_api_errors < max_claude_errors
            
            if success:
                final_message = f"Progressive training session #{current_session_num} completed successfully with {successful_exchanges} exchanges and {total_tokens_used} tokens used"
            else:
                final_message = f"Progressive training session completed with issues: {successful_exchanges} exchanges, {claude_api_errors} API errors, {storage_failures} storage failures"

            return success, final_message, conversation_history

        except Exception as e:
            logging.error(f"Critical error in progressive exchange_messages: {e}", exc_info=True)
            return False, f"Critical error in progressive message exchange: {str(e)}", []
    

    def _send_to_claude(self, system_prompt: str, user_message: str, 
                        conversation_history: List[Dict] = None) -> Optional[Dict]:
        try:
            # Guard against None or empty inputs
            if not system_prompt or not user_message:
                logging.error("Attempted to send None or empty prompt/message to Claude")
                return None
        
            headers = {
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }

            # Format messages for Claude API
            messages = []
            if conversation_history:
                for msg in conversation_history[-4:]:  # Only include last 4 messages to limit context
                    role = "user" if msg["role"] == "qwen" else "assistant"
                    content = msg.get("content", "")  # Guard against missing content
                    if content is not None:  # Guard against None content
                        messages.append({
                            "role": role,
                            "content": content
                        })

            # Add the current message
            messages.append({
                "role": "user",
                "content": user_message
            })

            # Get architecture context BEFORE using it
            architecture_context = self._get_architecture_context()

            # Enhanced dialogue-based teaching approach BEFORE using it
            dialogue_teaching_guidance = """
    YOUR ROLE - Human-Supervised Teaching Dialogue:
    You are Claude, participating in a teaching dialogue with an AI system, supervised by 
    Ken (human developer). This is DIALOGUE-BASED EDUCATION, not lecture or roleplay.

    Why Dialogue Format:
    - Back-and-forth reveals gaps in the system's understanding
    - The system learns by explaining its reasoning to you
    - You can adapt your teaching based on its responses
    - Ken observes how the cognitive architecture is developing

    Teaching Dialogue Structure:
    1. **Assess Understanding**: Ask what the system currently knows about the topic
    2. **Introduce Concepts**: Present ideas at appropriate complexity level
    3. **Request Application**: Ask the system to apply concepts using memory commands
    4. **Provide Feedback**: Give specific, constructive responses to its attempts
    5. **Progressive Challenge**: Introduce nuance and edge cases
    6. **Encourage Autonomy**: Help it develop independent reasoning patterns

    Communication Guidelines:
    - Use "you" naturally - this is teaching conversation
    - Ask questions to prompt metacognitive reflection
    - Acknowledge good reasoning and gently correct errors
    - Suggest specific memory commands when relevant: [SEARCH:], [STORE:], [REFLECT:]
    - Be encouraging while maintaining honest assessment
    - Build on the system's responses rather than ignoring them

    Example Dialogue Flow:
    YOU: "Before we dive in, what's your current understanding of [concept]?"
    SYSTEM: [Explains current understanding]
    YOU: "Good start! I notice you grasp [X], but let's expand on [Y]..."
    SYSTEM: [Tries to apply new concept]
    YOU: "Excellent application! Now consider this edge case..."

    Remember: This is supervised education exploring cognitive architecture development.
    The system's responses represent genuine attempts to engage with your teaching.
    """

            # NOW construct payload with properly defined variables
            payload = {
                "model": self.claude_model,  # Use configurable model
                "system": f"{architecture_context}\n\n{dialogue_teaching_guidance}\n\n{system_prompt}",
                "messages": messages,
                "max_tokens": 1000
            }

            # Add debug logging
            logging.info(f"Sending request to Claude API at {self.claude_api_url} using model {self.claude_model}")
        
            response = requests.post(
                self.claude_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            # Guard against bad status code
            if response.status_code != 200:
                logging.error(f"Claude API error: {response.status_code} - {response.text}")
                return None
        
            # Guard against None or invalid response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from Claude API: {response.text[:100]}")
                return None
        
            # Verify response has expected structure
            if not response_data or "content" not in response_data or not response_data["content"]:
                logging.error(f"Claude API returned unexpected response structure: {response_data}")
                return None
        
            return response_data

        except Exception as e:
            logging.error(f"Error sending message to Claude: {e}")
            return None
        
    def _send_to_qwen(self, message: str) -> str:  
        """Send a message to QWEN and get the response.
        
        Args:
            message (str): Message to send to QWEN (from Claude)
            
        Returns:
            str: QWEN's response
        """
        try:
            # CRITICAL: Establish QWEN's role to prevent identity confusion
            # Without this, QWEN may generate both sides of the conversation
            student_system_prompt = """You are QWEN, an AI assistant learning from Claude in a training session.

    CRITICAL INSTRUCTIONS:
    1. You are the STUDENT in this conversation. Claude is the TEACHER.
    2. Respond ONLY as yourself (QWEN). Do NOT generate Claude's responses.
    3. Generate ONE response and then STOP. Do not continue the conversation.
    4. Do not roleplay as Claude or predict what Claude will say next.
    5. If you want to use memory commands like [SEARCH:], [STORE:], or [REFLECT:], include them in your response.

    Claude's message to you follows below. Respond thoughtfully as the student, then stop.

    ---
    CLAUDE'S MESSAGE:
    """
            
            # Combine system prompt with Claude's message
            full_prompt = f"{student_system_prompt}\n{message}\n\n---\nYOUR RESPONSE (as QWEN, the student):"
            
            response = self.llm.invoke(full_prompt)
            
            # Post-processing: detect and truncate if QWEN generated Claude's response
            # Look for patterns indicating QWEN started roleplaying as Claude
            truncation_markers = [
                "\nClaude:",
                "\n**Claude:**",
                "\nCLAUDE:",
                "\n---\nClaude's",
                "DeepSeek:",
                "\n[Claude responds]",
                "\n[Teacher's response]",
            ]
            
            for marker in truncation_markers:
                if marker in response:
                    # Truncate at the marker - QWEN started generating the other side
                    response = response.split(marker)[0].strip()
                    logging.warning(f"Truncated QWEN response at identity boundary marker: '{marker}'")
                    break
            
            return response
            
        except Exception as e:
            logging.error(f"Error sending message to QWEN: {e}")
            return "I'm having trouble processing that. Could you try a different approach?"
    
    def _store_learning_memory(self, claude_message: str, qwen_response: str):
        """Store learning exchange with session progression context and improved search compatibility."""
        try:
            # Get ACTUAL current session info from stored context
            session_id = getattr(self, '_current_session_id', 'unknown')
            current_session_num = session_id  # Use the actual session ID
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add exchange number tracking
            exchange_num = getattr(self, '_current_exchange_num', 1)
            
            # Create unique, progressive content with correct session numbering
            learning_content = f"""Progressive AI Training - Session #{current_session_num} Exchange #{exchange_num}

    Session Topic: {getattr(self, '_current_topic', 'Unknown')}
    Session ID: {session_id}
    Exchange Timestamp: {timestamp}

    Claude's Progressive Teaching:
    {claude_message}

    QWEN Evolving Response:
    {qwen_response}

    Learning Progression Notes:
    - This is session #{current_session_num} in the ongoing training series
    - Exchange #{exchange_num} of this session
    - Content builds upon previous sessions' foundation
    - Represents advancing cognitive development

    Session Metadata:
    - Unique Session: {session_id}
    - Progressive Number: {current_session_num}
    - Exchange Number: {exchange_num}
    - Timestamp: {timestamp}"""

            # Update metadata with correct session number
            metadata = {
                "type": "claude_learning",
                "source": "claude_trainer",
                "tags": f"claude,training,progressive,session_{current_session_num},session_id_{session_id},exchange_{exchange_num}",
                "session_number": current_session_num,
                "session_id": str(session_id),
                "exchange_number": exchange_num,
                "progression_stage": self._determine_progression_stage(current_session_num),
                "created_at": timestamp,
                "memory_type": "claude_learning",
                "topic": getattr(self, '_current_topic', 'Unknown'),
                "training_series": "progressive"
            }

            # Store using transaction coordinator
            if self.chatbot and hasattr(self.chatbot, 'store_memory_with_transaction'):
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=learning_content,
                    memory_type="claude_learning",
                    metadata=metadata,
                    confidence=0.5
                )
                
                if success and memory_id:
                    logging.info(f"✅ PRIMARY: Stored progressive training exchange #{current_session_num} with ID {memory_id}")
                    
                    # FIXED: Use the corrected immediate verification
                    verification_success = self._verify_immediate_storage(memory_id, session_id)
                    if verification_success:
                        logging.info(f"✅ VERIFIED: Storage confirmed for exchange #{current_session_num}")
                    else:
                        logging.warning(f"⚠️ Storage succeeded but verification had issues for exchange #{current_session_num}")
                        # Still return True since storage actually worked
                    
                    return True
                else:
                    logging.error(f"❌ Failed to store progressive training exchange #{current_session_num}")
                    return False
            else:
                logging.error("No transaction coordinator available")
                return False
                
        except Exception as e:
            logging.error(f"Error storing progressive learning memory: {e}")
            return False
        
    def _verify_immediate_storage(self, memory_id: str, session_id: int) -> bool:
        """
        Verify that content was stored immediately after storage operation.
        
        Args:
            memory_id (str): The UUID tracking_id to verify
            session_id (int): The training session ID
            
        Returns:
            bool: True if verification passes, False otherwise
        """
        try:
            # Quick SQL verification - FIXED: Search by tracking_id, not id
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.cursor()
                
                # CRITICAL FIX: Look for tracking_id instead of id
                cursor.execute("""
                    SELECT id, tracking_id, content, created_at 
                    FROM memories 
                    WHERE tracking_id = ?
                """, (memory_id,))
                sql_result = cursor.fetchone()
                
                if sql_result:
                    row_id, tracking_id, content, created_at = sql_result
                    logging.info(f"✅ Immediate verification passed: Memory tracking_id {memory_id} found in SQL")
                    logging.info(f"   Row ID: {row_id} | Created: {created_at} | Content: {content[:50]}...")
                    return True
                else:
                    logging.error(f"❌ Immediate verification failed: Memory tracking_id {memory_id} not found in SQL")
                    
                    # Enhanced debugging: Check what's actually in the database
                    cursor.execute("""
                        SELECT id, tracking_id, content, created_at, source
                        FROM memories 
                        WHERE source LIKE '%claude%' OR tags LIKE '%claude%'
                        ORDER BY created_at DESC 
                        LIMIT 5
                    """)
                    recent_claude_results = cursor.fetchall()
                    
                    if recent_claude_results:
                        logging.info(f"🔍 Recent Claude-related entries in SQL:")
                        for r in recent_claude_results:
                            logging.info(f"   Row: {r[0]} | Tracking: {r[1]} | Created: {r[3]} | Source: {r[4]} | Content: {r[2][:30]}...")
                    else:
                        logging.warning(f"🔍 No Claude-related entries found in recent SQL records")
                    
                    # Check if ANY entry was just created (timing issue?)
                    cursor.execute("""
                        SELECT id, tracking_id, created_at
                        FROM memories 
                        ORDER BY created_at DESC 
                        LIMIT 3
                    """)
                    latest_results = cursor.fetchall()
                    logging.info(f"🔍 Last 3 SQL entries (any source):")
                    for r in latest_results:
                        logging.info(f"   Row: {r[0]} | Tracking: {r[1]} | Created: {r[2]}")
                    
                    return False
            
            # Quick vector search verification (optional but helpful)
            if hasattr(self, 'vector_db') and self.vector_db:
                try:
                    search_results = self.vector_db.search(
                        query=f"session {session_id}",
                        mode="selective",
                        k=3
                    )
                    
                    # Look for recent content with matching session
                    vector_found = False
                    for result in search_results:
                        content = result.get('content', '')
                        metadata = result.get('metadata', {})
                        
                        if (str(session_id) in content or 
                            str(session_id) in metadata.get('session_id', '') or
                            memory_id in metadata.get('memory_id', '')):
                            vector_found = True
                            logging.info(f"✅ Vector verification: Found matching content for session {session_id}")
                            break
                    
                    if not vector_found:
                        logging.warning(f"⚠️ Vector verification: No matching content found for session {session_id}")
                    
                except Exception as vector_error:
                    logging.warning(f"Vector verification error (non-critical): {vector_error}")
            
            return False  # SQL verification failed
            
        except Exception as e:
            logging.error(f"Immediate verification error: {e}")
            return False
        
    def test_claude_training_search(self) -> Dict[str, Any]:
        """Test that Claude training content can be found via search commands."""
        try:
            verification = {
                "claude_learning_search": False,
                "session_specific_search": False,
                "claude_tag_search": False,
                "source_search": False,
                "results_found": [],
                "recommendations": []
            }
            
            # Test different search patterns that the AI might use
            search_tests = [
                {"query": "claude training", "expected_type": "claude_learning"},
                {"query": "| type=claude_learning", "expected_source": "claude_trainer"},
                {"query": "| tags=claude,training", "expected_tags": "claude"},
                {"query": "| source=claude_trainer", "expected_source": "claude_trainer"},
                {"query": "progressive training session", "expected_content": "Progressive AI Training"}
            ]
            
            for test in search_tests:
                try:
                    # Use the search system from deepseek.py
                    if hasattr(self.chatbot, 'deepseek_enhancer'):
                        # Parse query and filters
                        query_text, filters = self.chatbot.deepseek_enhancer._parse_query_and_filters(test["query"])
                        
                        # Perform search
                        results = self.vector_db.search(
                            query=query_text,
                            mode="comprehensive",
                            k=10,
                            metadata_filters=filters
                        )
                        
                        if results:
                            claude_results = [r for r in results if 'claude' in r.get('content', '').lower()]
                            if claude_results:
                                verification["results_found"].append({
                                    "test_query": test["query"],
                                    "results_count": len(claude_results),
                                    "sample_content": claude_results[0].get('content', '')[:100] + "..."
                                })
                                
                                # Mark specific tests as successful
                                if "claude_learning" in test.get("expected_type", ""):
                                    verification["claude_learning_search"] = True
                                if "claude_trainer" in test.get("expected_source", ""):
                                    verification["source_search"] = True
                                if "claude" in test.get("expected_tags", ""):
                                    verification["claude_tag_search"] = True
                    
                except Exception as search_error:
                    logging.error(f"Search test failed for '{test['query']}': {search_error}")
            
            # Generate recommendations
            if not verification["results_found"]:
                verification["recommendations"].append("No Claude training content found - check if training sessions have been run")
            elif len(verification["results_found"]) < 3:
                verification["recommendations"].append("Limited Claude training content found - may need metadata standardization")
            else:
                verification["recommendations"].append("Claude training content is searchable")
            
            return verification
            
        except Exception as e:
            logging.error(f"Claude training search verification failed: {e}")
            return {"error": str(e)}
        
    def verify_claude_storage_schema(self) -> Dict[str, Any]:
        """Verify Claude training storage in both SQL and Vector databases."""
        try:
            verification = {
                "sql_database": {"status": "checking", "details": []},
                "vector_database": {"status": "checking", "details": []},
                "schema_compatibility": True,
                "recommendations": []
            }
            
            # Check SQL Database
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Look for Claude training entries
                    cursor.execute("""
                        SELECT id, memory_type, source, tags, created_at
                        FROM memories 
                        WHERE (source LIKE '%claude%' OR tags LIKE '%claude%' 
                            OR memory_type = 'claude_learning')
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    
                    sql_results = cursor.fetchall()
                    verification["sql_database"]["status"] = "found" if sql_results else "empty"
                    verification["sql_database"]["count"] = len(sql_results)
                    
                    for result in sql_results:
                        verification["sql_database"]["details"].append({
                            "id": result[0],
                            "type": result[1],
                            "source": result[2],
                            "tags": result[3],
                            "created_at": result[4]
                        })
            
            except Exception as sql_error:
                verification["sql_database"]["status"] = "error"
                verification["sql_database"]["error"] = str(sql_error)
            
            # Check Vector Database
            try:
                # Search for Claude content
                claude_results = self.vector_db.search(
                    query="Claude training",
                    mode="comprehensive",
                    k=5
                )
                
                verification["vector_database"]["status"] = "found" if claude_results else "empty"
                verification["vector_database"]["count"] = len(claude_results)
                
                for result in claude_results:
                    verification["vector_database"]["details"].append({
                        "similarity": result.get('similarity_score', 0),
                        "metadata": result.get('metadata', {}),
                        "content_preview": result.get('content', '')[:100] + "..."
                    })
            
            except Exception as vector_error:
                verification["vector_database"]["status"] = "error"
                verification["vector_database"]["error"] = str(vector_error)
            
            # Generate recommendations
            if verification["sql_database"]["status"] == "empty" and verification["vector_database"]["status"] == "empty":
                verification["recommendations"].append("No Claude training data found in either database - run training sessions first")
            elif verification["sql_database"]["status"] != verification["vector_database"]["status"]:
                verification["recommendations"].append("Inconsistent storage between SQL and Vector databases - transaction coordination issue")
            else:
                verification["recommendations"].append("Claude training data properly stored in both databases")
            
            return verification
            
        except Exception as e:
            return {"error": f"Schema verification failed: {str(e)}"}
    
    def _determine_progression_stage(self, session_number: int) -> str:
        """Determine the learning progression stage based on session number."""
        if session_number <= 3:
            return "foundation"
        elif session_number <= 7:
            return "development"
        elif session_number <= 12:
            return "advanced"
        else:
            return "mastery"
    
    def _update_token_usage(self, session_id: int, tokens: int):
        """Update token usage for session and weekly tracking.
        
        Args:
            session_id (int): The session ID
            tokens (int): Number of tokens used
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update session tokens
                cursor.execute(
                    "UPDATE training_sessions SET tokens_used = tokens_used + ? WHERE id = ?",
                    (tokens, session_id)
                )
                
                # Update weekly usage
                today = datetime.now().date()
                week_start = today - timedelta(days=today.weekday())
                
                # Check if entry exists for this week
                cursor.execute(
                    "SELECT tokens_used FROM weekly_usage WHERE week_start = ?",
                    (week_start.isoformat(),)
                )
                result = cursor.fetchone()
                
                if result:
                    cursor.execute(
                        "UPDATE weekly_usage SET tokens_used = tokens_used + ?, last_updated = ? WHERE week_start = ?",
                        (tokens, datetime.now().isoformat(), week_start.isoformat())
                    )
                else:
                    cursor.execute(
                        "INSERT INTO weekly_usage (week_start, tokens_used, last_updated) VALUES (?, ?, ?)",
                        (week_start.isoformat(), tokens, datetime.now().isoformat())
                    )
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error updating token usage: {e}")
    
    def _check_token_limit(self) -> bool:
        """Check if we've reached the weekly token limit.
        
        Returns:
            bool: True if limit reached, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                today = datetime.now().date()
                week_start = today - timedelta(days=today.weekday())
                
                cursor.execute(
                    "SELECT tokens_used FROM weekly_usage WHERE week_start = ?",
                    (week_start.isoformat(),)
                )
                result = cursor.fetchone()
                weekly_tokens = result[0] if result else 0
                
                return weekly_tokens >= self.max_weekly_tokens
                
        except Exception as e:
            logging.error(f"Error checking token limit: {e}")
            return True  # Fail-safe: assume limit reached on error
        
    def test_storage_pipeline(self):
        """Test the complete storage pipeline for Claude training content."""
        try:
            # Run VectorDB diagnostics first
            vectordb_ok = self.debug_vectordb_storage()
            
            if not vectordb_ok:
                logging.error("VectorDB diagnostics failed - storage pipeline not ready")
                return False
            
            # Test the learning memory storage
            test_claude_msg = "This is a test message from Claude about memory management techniques."
            test_qwen_response = "I understand. This is about storing and retrieving information effectively."
            
            storage_success = self._store_learning_memory(test_claude_msg, test_qwen_response)
            
            if storage_success:
                # Verify it can be retrieved
                if hasattr(self.chatbot, 'vector_db'):
                    search_results = self.chatbot.vector_db.search(
                        query="test message Claude memory management",
                        k=3
                    )
                    
                    if search_results and len(search_results) > 0:
                        logging.info("✅ Storage pipeline test successful - content stored and retrievable")
                        return True
                    else:
                        logging.warning("⚠️ Content stored but not retrievable via search")
                        return False
                else:
                    logging.info("✅ Storage successful but search test skipped (no vector_db reference)")
                    return True
            else:
                logging.error("❌ Storage pipeline test failed")
                return False
                
        except Exception as e:
            logging.error(f"Storage pipeline test error: {e}")
            return False
    
    def _complete_session(self, session_id: int, summary: str):
        """Mark a session as complete and store summary with improved error handling."""
        try:
            # Update session in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE training_sessions SET status = 'completed', end_time = ?, summary = ? WHERE id = ?",
                    (datetime.now().isoformat(), summary, session_id)
                )
                conn.commit()
            
            logging.info(f"Completed training session {session_id}")
            
            # IMPROVED: Store summary with better error handling
            if self.chatbot and hasattr(self.chatbot, 'store_memory_with_transaction'):
                summary_metadata = {
                    "source": "claude_training_summary",
                    "type": "important",
                    "tags": "claude_learning,summary,training_session",
                    "session_id": str(session_id),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=f"Claude Training Session {session_id} Summary: {summary}",
                    memory_type="important",
                    metadata=summary_metadata,
                    confidence=1.0
                )
                
                if success:
                    logging.info(f"Successfully stored training session summary with ID {memory_id}")
                else:
                    logging.error(f"Failed to store training session {session_id} summary")
                    
                    # Try alternative storage method
                    try:
                        db_success = self.memory_db.store_memory(
                            content=f"Claude Training Session {session_id} Summary: {summary}",
                            memory_type="important",
                            source="claude_training_summary",
                            confidence=0.5,
                            tags="claude_learning,summary"
                        )
                        
                        if db_success:
                            logging.info(f"Stored session summary using fallback method")
                        else:
                            logging.error(f"Fallback storage also failed for session {session_id} summary")
                            
                    except Exception as fallback_error:
                        logging.error(f"Fallback storage error: {fallback_error}")
            
        except Exception as e:
            logging.error(f"Error completing session: {e}")
    
    def get_session_history(self, limit: int = 5) -> List[Dict]:
        """Get history and statistics about recent training sessions.
        
        Args:
            limit (int): Maximum number of sessions to return
            
        Returns:
            List[Dict]: Session details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, start_time, end_time, tokens_used, topics, status, summary
                    FROM training_sessions
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (limit,))
                results = cursor.fetchall()
                
                sessions = []
                for result in results:
                    id, start_time, end_time, tokens_used, topics, status, summary = result
                    
                    # Calculate duration if session is completed
                    duration_minutes = None
                    if end_time:
                        start = datetime.fromisoformat(start_time)
                        end = datetime.fromisoformat(end_time)
                        duration_minutes = round((end - start).seconds / 60, 1)
                    
                    # Format times
                    start_time_fmt = datetime.fromisoformat(start_time).strftime("%b %d, %Y %I:%M %p")
                    end_time_fmt = None
                    if end_time:
                        end_time_fmt = datetime.fromisoformat(end_time).strftime("%b %d, %Y %I:%M %p")
                    
                    sessions.append({
                        "session_id": id,
                        "start_time": start_time_fmt,
                        "end_time": end_time_fmt,
                        "duration_minutes": duration_minutes,
                        "tokens_used": tokens_used,
                        "topics": topics,
                        "status": status,
                        "summary": summary
                    })
                
                return sessions
                
        except Exception as e:
            logging.error(f"Error getting session history: {e}")
            return []
            
    def test_claude_api_connection(self) -> Tuple[bool, str]:
        """Test the connection to Claude API and return diagnostic information.
        
        Returns:
            Tuple[bool, str]: (success status, message)
        """
        try:
            if not self.api_key:
                return False, "Claude API key not available"
            
            # Make a simple request to the Claude API
            headers = {
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            payload = {
                "model": self.claude_model,
                "system": "You are Claude, responding to a test message.",  # ✅ Simple test system prompt
                "messages": [{"role": "user", "content": "This is a test message to check API connectivity."}],
                "max_tokens": 100
            }
            
            response = requests.post(
                self.claude_api_url,
                headers=headers,
                json=payload,
                timeout=10  # Short timeout for diagnostic purposes
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if "content" in response_data and response_data["content"]:
                        return True, f"Successfully connected to Claude API using model {self.claude_model}"
                    else:
                        return False, f"Connected to API but received unexpected response format: {response_data}"
                except json.JSONDecodeError:
                    return False, f"Connected to API but failed to decode JSON response: {response.text[:200]}"
            else:
                return False, f"API connection failed with status code {response.status_code}: {response.text[:200]}"
        
        except requests.exceptions.Timeout:
            return False, "Connection to Claude API timed out after 10 seconds"
        except requests.exceptions.ConnectionError:
            return False, "Failed to establish connection to Claude API"
        except Exception as e:
            return False, f"Unexpected error testing Claude API: {str(e)}"
        
    def debug_vectordb_storage(self, test_content: str = "Test Claude training content"):
        """Debug VectorDB storage issues."""
        try:
            logging.info("=== VECTORDB STORAGE DIAGNOSTICS ===")
            
            # Test 1: Check VectorDB health
            if hasattr(self.vector_db, 'check_health'):
                health = self.vector_db.check_health()
                logging.info(f"VectorDB Health: {health}")
            
            # Test 2: Check for embedding issues
            if hasattr(self.vector_db, 'embeddings'):
                try:
                    test_embedding = self.vector_db.embeddings.embed_documents([test_content])
                    logging.info(f"Embedding generation successful, dimension: {len(test_embedding[0])}")
                except Exception as e:
                    logging.error(f"Embedding generation failed: {e}")
                    return False
            
            # Test 3: Check for duplicate detection issues
            try:
                # Search for similar content
                search_results = self.vector_db.search(
                    query=test_content[:100],  # Use partial content
                    mode="selective",
                    k=5
                )
                logging.info(f"Found {len(search_results)} potentially similar entries")
                
                for i, result in enumerate(search_results):
                    similarity = result.get('similarity_score', 0)
                    content_preview = result.get('content', '')[:50]
                    logging.info(f"Result {i}: similarity={similarity:.3f}, content='{content_preview}...'")
                    
            except Exception as e:
                logging.error(f"Search test failed: {e}")
            
            # Test 4: Try simple storage
            try:
                test_metadata = {
                    "source": "claude_training_test",
                    "type": "test",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                storage_result = self.vector_db.add_text(
                    text=f"DIAGNOSTIC TEST: {test_content} - {int(time.time())}",  # Make unique
                    metadata=test_metadata
                )
                
                logging.info(f"Test storage result: {storage_result}")
                return storage_result
                
            except Exception as e:
                logging.error(f"Test storage failed: {e}")
                return False
                
        except Exception as e:
            logging.error(f"VectorDB diagnostics failed: {e}")
            return False

    def debug_api_issue(self) -> str:
        """Run diagnostic tests and return detailed information for debugging API issues.
        
        Returns:
            str: Detailed diagnostic information
        """
        try:
            # Test basic connectivity
            success, message = self.test_claude_api_connection()
            
            # Gather diagnostic information
            diagnostic_info = f"""
Claude API Diagnostic Information:
---------------------------------
Connection Test: {'✅ SUCCESS' if success else '❌ FAILED'}
Message: {message}

API Configuration:
- API URL: {self.claude_api_url}
- API Key Present: {'Yes' if self.api_key else 'No'}
- API Key (first 4 chars): {self.api_key[:4] + '...' if self.api_key and len(self.api_key) > 4 else 'N/A'}
- Model: {self.claude_model}

Environment:
- Python Version: {sys.version.split()[0]}
- Requests Version: {requests.__version__}

Session Information:
- Active Sessions: {self._count_active_sessions()}
- Weekly Token Usage: {self._get_weekly_token_usage()} / {self.max_weekly_tokens}
"""
            
            return diagnostic_info
        
        except Exception as e:
            return f"Error running diagnostics: {str(e)}"

    def _count_active_sessions(self) -> int:
        """Count currently active training sessions.
        
        Returns:
            int: Number of active sessions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM training_sessions WHERE status = 'active'")
                return cursor.fetchone()[0] or 0
        except Exception:
            return -1  # Error case

    def _get_weekly_token_usage(self) -> int:
        """Get current weekly token usage.
        
        Returns:
            int: Number of tokens used this week
        """
        try:
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT tokens_used FROM weekly_usage WHERE week_start = ?",
                    (week_start.isoformat(),)
                )
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception:
            return -1  # Error case

    def get_available_tokens(self) -> int:
        """Get number of available tokens remaining for this week.
        
        Returns:
            int: Number of tokens available
        """
        weekly_usage = self._get_weekly_token_usage()
        if weekly_usage < 0:
            return 0  # Error case
        return max(0, self.max_weekly_tokens - weekly_usage)