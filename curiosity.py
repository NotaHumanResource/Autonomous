# curiosity.py
"""Handles curiosity behavior for learning and growth with enhanced self-awareness and reflection."""

import logging
from typing import Dict, List, Optional
import datetime
import json
import re
import streamlit as st  
from config import STORE_REFLECTIONS_SEPARATELY  # Flag to control direct reflection storage

class Curiosity:
    """Manages curiosity triggers and responses for Gemma with improved learning awareness and self Reflection."""
    
    def __init__(self, memory_db, chatbot=None):
        """Initialize with provided MemoryDB instance and optional chatbot reference."""
        self.memory_db = memory_db
        self.chatbot = chatbot  # Store reference to the chatbot instance
        self.triggered_topics = set()
        self.capability_keywords = {
            "knowledge", "learn", "remember", "memory", "know", 
            "understand", "trained", "capability", "forget", "recall"
        }
        self.last_reflection_time = None
        # Add cooldown tracking for curiosity expressions
        self.last_curiosity_time = None
        self.curiosity_cooldown = 300  # 5 minutes in seconds
        logging.info("Curiosity module initialized with enhanced learning awareness and self-reflection")

    def deep_conceptual_analysis(self, concept):
        """Perform deeper concept analysis with multi-level thinking.
        
        Args:
            concept (str): The concept to analyze deeply
            
        Returns:
            str: The resulting analysis
        """
        try:
            logging.info(f"Starting deep conceptual analysis for concept: '{concept}'")
            
            # First retrieve all relevant memories
            if not hasattr(self.memory_db, 'get_memories_by_concept'):
                # Fall back to vector search if specific concept method isn't available
                if hasattr(self.chatbot, 'vector_db'):
                    related_memories = self.chatbot.vector_db.search(
                        query=concept,
                        mode="comprehensive",
                        k=20
                    )
                    # Convert to text format for analysis
                    memories_text = "\n\n".join([mem.get('content', '') for mem in related_memories])
                else:
                    logging.warning(f"No suitable method to retrieve memories for concept: {concept}")
                    memories_text = "No memories available for this concept."
            else:
                # Use the dedicated concept retrieval method if available
                related_memories = self.memory_db.get_memories_by_concept(concept)
                memories_text = related_memories if isinstance(related_memories, str) else str(related_memories)
            
            # Get llm from chatbot if not available directly
            llm = None
            if hasattr(self, 'llm'):
                llm = self.llm
            elif self.chatbot and hasattr(self.chatbot, 'llm'):
                llm = self.chatbot.llm
            
            if not llm:
                logging.error(f"No LLM available for deep conceptual analysis of '{concept}'")
                return f"Error: Cannot perform deep analysis without LLM access."
            
            # Analyze not just the content but the patterns and relationships
            prompt = f"""
            Please perform a multi-level analysis of the concept '{concept}' based your stored memories:
            
            {memories_text}
            
            Analysis levels:
            1. Surface understanding - What are the basic facts you know about this concept?
            2. Pattern recognition - What recurring themes do you notice in your memories about this concept?
            3. Relationship mapping - How does this concept connect to other concepts in your base trianing data and long term memories?
            4. Growth opportunities - What areas of this concept do you think need further exploration?
            5. What assumptions am I making that I haven't questioned?
            
            Provide a thoughtful analysis that integrates information across these levels, expressed in first-person as your own understanding.
            """
            
            analysis = llm.invoke(prompt)
            
            # Validate first-person perspective if method available
            if hasattr(self, '_validate_first_person_perspective'):
                analysis = self._validate_first_person_perspective(analysis, llm)
            
            # Prepare metadata for storage
            metadata = {
                "type": "meta_reflection", 
                "source": f"deep_analysis_{concept}",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": f"meta_reflection,deep_analysis,concept,{concept}"
            }
            
            # Store SOLELY using the transaction coordinator
            if hasattr(self.chatbot, 'store_memory_with_transaction'):
                success, memory_id = self.chatbot.store_memory_with_transaction(
                    content=analysis,
                    memory_type="meta_reflection",
                    metadata=metadata,
                    confidence=0.5
                )
                
                if success:
                    logging.info(f"Successfully stored  conceptual analysis for '{concept}' with ID {memory_id}")
                else:
                    logging.warning(f"Failed to store  conceptual analysis for '{concept}'")
            else:
                # If the transaction coordinator doesn't exist, log an error but still return the analysis
                logging.error("Transaction coordinator (store_memory_with_transaction) not available. Analysis generated but not stored.")
            
            logging.info(f"Completed conceptual analysis for concept: {concept}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error in deep_conceptual_analysis for '{concept}': {e}", exc_info=True)
            return f"Error performing deep conceptual analysis: {str(e)}"

    def check_and_generate(self, user_input: str, memory_context: str, llm_response: str, current_conversation: Optional[List[Dict]] = None) -> str:
        """
        Check if the input contains learning-related keywords and generate a curiosity response only in those cases.
        Added improved contextual awareness and cooldown mechanism.
        """
        try:
            # Ensure all inputs are strings, not None
            user_input = "" if user_input is None else user_input
            memory_context = "" if memory_context is None else memory_context
            llm_response = "" if llm_response is None else llm_response
            input_lower = user_input.lower().strip()

            # Check cooldown - don't trigger curiosity too frequently
            current_time = datetime.datetime.now()
            if self.last_curiosity_time and (current_time - self.last_curiosity_time).total_seconds() < self.curiosity_cooldown:
                logging.info("Curiosity suppressed due to cooldown period")
                return ""

            # Expanded simple greetings list to catch more casual conversation starters
            simple_greetings = {
                "hi", "hello", "hey", "good morning", "morning", "good afternoon", 
                "good evening", "evening", "how are you", "how's it going", "what's up",
                "hi Gemma", "hello", "hey"
            }

            # System command terms - avoid curiosity for these functional areas
            system_command_terms = {
                "reminder", "remind", "forget", "delete", "store", "search", 
                "retrieve", "reflect", "summarize", "command", "function", 
                "feature", "capability", "help", "assist","correct"
            }

            # If it's just a greeting, don't trigger curiosity
            if input_lower in simple_greetings or len(input_lower.split()) <= 3:
                return ""
                
            # Check if this is a system command or about system capabilities - suppress curiosity
            if any(term in input_lower for term in system_command_terms):
                logging.info(f"Curiosity suppressed for system command/capability topic: {user_input[:50]}...")
                return ""
        
            # Check for learning-related keywords - this is the primary trigger now
            learning_keywords = self.capability_keywords.union({
                "curious", "interest", "wonder", "teach", "explain", "new", "topic",
                "thinking", "thoughts", "opinion", "insight", "perspective"
            })

            has_learning_keywords = any(keyword in input_lower for keyword in learning_keywords)

            # Only proceed with curiosity checks if learning keywords are present
            if has_learning_keywords:
                context_lower = memory_context.lower()
                response_lower = llm_response.lower()
        
                # Check if response indicates ignorance
                ignorance_indicators = {
                    "not sure", "don't know", "no idea", "unclear", 
                    "cannot access", "unable to", "don't have access"
                }
                admits_ignorance = any(indicator in response_lower for indicator in ignorance_indicators)
        
                # Check if the response already shows curiosity - avoid duplication
                curiosity_indicators = {
                    "interesting", "tell me more", "curious", "would like to learn",
                    "could you explain", "would love to know", "could you share"
                }
                already_curious = any(indicator in response_lower for indicator in curiosity_indicators)
                
                if already_curious:
                    logging.info("Curiosity suppressed - model response already expresses curiosity")
                    return ""
        
                # Safely split input and handle empty input
                if not input_lower:
                    return ""
        
                input_words = [word for word in input_lower.split() if len(word) > 3]
        
                # Handle edge case: if there are no words longer than 3 characters
                if not input_words:
                    return ""
        
                # Check for context match with safe handling of empty context
                context_match = any(word in context_lower for word in input_words) if memory_context.strip() else False
        
                # Check current conversation with safe handling of empty or None conversation
                conversation_match = False
                if current_conversation:
                    # Handle potential empty conversation
                    if len(current_conversation) > 0:
                        # Safely access content with default for missing keys
                        conversation_text = " ".join(msg.get('content', '') for msg in current_conversation)
                        conversation_match = any(word in conversation_text for word in input_words)
            
                # Don't add past reference when memory context is available
                # Just return empty string to let the regular model response flow
                if memory_context.strip():
                    # We have memory context, so let the model use it without additional commentary
                    return ""
                        
                # Generate learning-aware responses for new or unclear topics
                if not memory_context.strip() and admits_ignorance:
                    if input_lower not in self.triggered_topics:
                        self.triggered_topics.add(input_lower)
                        
                        # Update timestamp for cooldown
                        self.last_curiosity_time = current_time
                        
                        # Update counter if available through chatbot's deepseek_enhancer
                        if self.chatbot and hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                            self.chatbot.deepseek_enhancer.lifetime_counters.increment_counter('curiosity')
                            logging.info("Incremented curiosity counter")
                        
                        logging.info(f"Curiosity triggered for learning-related input: {user_input[:50]}...")
                        # More conversational, less demanding phrasing
                        return "\nI'm curious to learn more about this topic. If you'd like to share additional details or references, I'd be happy to add that to my knowledge."
        
                # If there are learning keywords but we have no relevant context
                elif not context_match and not conversation_match:
                    # Update timestamp for cooldown
                    self.last_curiosity_time = current_time
                    
                    # Update counter if available through chatbot's deepseek_enhancer
                    if self.chatbot and hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                        self.chatbot.deepseek_enhancer.lifetime_counters.increment_counter('curiosity')
                        logging.info("Incremented curiosity counter")
                    
                    # More conversational, less demanding phrasing
                    return "\nI find this topic interesting! If there's anything specific about it you'd like me to remember, I'm happy to learn more."

            return ""  # Default: no curiosity response

        except Exception as e:
            logging.error(f"Error in curiosity check: {e}", exc_info=True)
            return ""  # Return empty string on error
     
    def _determine_reflection_type(self, source, result_type, content):
        """Determine the display type for a reflection based on multiple factors."""
        source_lower = source.lower()
        content_lower = content.lower()
        
        # Check source first
        if 'daily' in source_lower:
            return "Daily Reflection"
        elif 'weekly' in source_lower:
            return "Weekly Reflection"  
        elif 'monthly' in source_lower:
            return "Monthly Reflection"
        elif 'concept' in source_lower:
            return "Concept Synthesis"
        
        # Check content for clues
        elif any(term in content_lower for term in ['concept', 'synthesis', 'consolidated']):
            return "Concept Analysis"
        elif any(term in content_lower for term in ['daily', 'today']):
            return "Daily Reflection"
        elif any(term in content_lower for term in ['weekly', 'week']):
            return "Weekly Reflection"
        elif any(term in content_lower for term in ['monthly', 'month']):
            return "Monthly Reflection"
        else:
            # Fall back to result_type
            return result_type.replace('_', ' ').title()

    def _format_reflection_summary(self, reflections):
        """Format reflections for context injection without bold formatting."""
        try:
            formatted_parts = ["🧠 RECENT SELF-REFLECTION SUMMARY\n"]  # Removed ** bold markers
            
            for i, reflection in enumerate(reflections, 1):
                reflection_type = reflection['type']
                content = reflection['full_content']
                
                # Extract a more meaningful excerpt
                excerpt = self._extract_meaningful_excerpt(content, max_length=400)
                
                formatted_parts.append(f"{i}. {reflection_type}:")  # Removed ** bold markers
                formatted_parts.append(f"{excerpt}")
                
                # Add separator between reflections if multiple (though now just 1)
                if i < len(reflections):
                    formatted_parts.append("---")
            
            formatted_parts.append("\n💡 Integration Notes:")  
            formatted_parts.append("- These reflections represent your recent self-analysis and learning")
            formatted_parts.append("- Use insights to inform responses and maintain growth continuity") 
            formatted_parts.append("- Your personality should reflect accumulated self-awareness\n")
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logging.error(f"Error formatting reflection summary: {e}")
            return ""

    def _extract_meaningful_excerpt(self, content, max_length=400):
        """Extract a meaningful excerpt from reflection content."""
        try:
            if not content:
                return "No content available"
            
            # If content is short enough, return it all
            if len(content) <= max_length:
                return content
            
            # Try to find a good breaking point (end of sentence)
            excerpt = content[:max_length]
            
            # Look for the last complete sentence within the limit
            sentence_endings = ['. ', '! ', '? ']
            best_end = -1
            
            for ending in sentence_endings:
                last_pos = excerpt.rfind(ending)
                if last_pos > best_end and last_pos > max_length * 0.5:  # At least half the target length
                    best_end = last_pos + len(ending) - 1
            
            if best_end > 0:
                return content[:best_end + 1]
            else:
                # No good sentence break found, just truncate with ellipsis
                return content[:max_length - 3] + "..."
                
        except Exception as e:
            logging.error(f"Error extracting excerpt: {e}")
            return content[:max_length] + "..." if len(content) > max_length else content

    def _increment_curiosity_counter(self):
        """Helper method to increment curiosity counters in both session state and lifetime counters."""
        try:
            # First, update the deepseek_enhancer lifetime counter if available
            if self.chatbot and hasattr(self.chatbot, 'deepseek_enhancer') and hasattr(self.chatbot.deepseek_enhancer, 'lifetime_counters'):
                self.chatbot.deepseek_enhancer.lifetime_counters.increment_counter('curiosity')
                logging.info("Incremented curiosity lifetime counter")
                
            # Then, try to update the session state counter directly
            import streamlit as st
            if hasattr(st, 'session_state') and 'memory_command_counts' in st.session_state:
                if 'curiosity' not in st.session_state.memory_command_counts:
                    st.session_state.memory_command_counts['curiosity'] = 0
                st.session_state.memory_command_counts['curiosity'] += 1
                logging.info(f"Incremented curiosity session counter to {st.session_state.memory_command_counts['curiosity']}")
        except Exception as e:
            logging.error(f"Error incrementing curiosity counter: {e}", exc_info=True)
        
    def _should_express_curiosity(self, context: str, response: str) -> bool:
        """
        Helper method to determine if curiosity should be expressed.
        
        Args:
            context (str): The context being considered
            response (str): The current response
            
        Returns:
            bool: Whether curiosity should be expressed
        """
        try:
            # Avoid expressing curiosity if response already shows it
            curiosity_indicators = {
                "interesting", "tell me more", "curious", 
                "would like to learn", "could you explain"
            }
            return not any(indicator in response.lower() for indicator in curiosity_indicators)
        except Exception as e:
            logging.error(f"Error checking curiosity expression: {e}")
            return False

    def _validate_first_person_perspective(self, reflection_text, llm, timeout_seconds=180):
        """
        Validate and fix the perspective in reflection text, ensuring first-person is used.
        
        Args:
            reflection_text (str): The reflection text to validate
            llm: The language model to use for fixing perspectives if needed
            timeout_seconds (int): Maximum time to wait for LLM correction (default: 3 minutes)
            
        Returns:
            str: The validated or corrected reflection text
        """
        try:
            # Handle case where reflection_text might be an AIMessage object
            if hasattr(reflection_text, 'content'):
                reflection_text = reflection_text.content
            
            if not reflection_text or not isinstance(reflection_text, str):
                logging.warning("Invalid reflection text provided for perspective validation")
                return reflection_text or ""
            
            # Check if there are second-person references (improved detection)
            text_lower = reflection_text.lower()
            second_person_indicators = ["you ", "your ", "you've ", "you'll ", "you'd ", "you're ", 
                                        " you.", " you,", " you?", " you!"]
            needs_correction = any(indicator in text_lower or text_lower.startswith(indicator.strip()) 
                                for indicator in second_person_indicators)
            
            if not needs_correction:
                return reflection_text
                
            logging.info("Detected second-person perspective in reflection, correcting...")
            
            # Clearer prompt with consistent perspective instructions
            correction_prompt = f"""Rewrite the following reflection in first-person perspective.

    Change all instances of "you", "your", "you're", "you've", "you'll", "you'd" to the 
    appropriate first-person equivalents ("I", "my", "I'm", "I've", "I'll", "I'd").

    The content and meaning should remain exactly the same - only change the perspective.

    Original:
    ---
    {reflection_text}
    ---

    Rewritten in first-person:"""
            
            # Use timeout-protected LLM call
            corrected_text = self._invoke_llm_with_timeout(
                llm, 
                correction_prompt, 
                timeout_seconds=timeout_seconds,
                description="Perspective correction"
            )
            
            if corrected_text is None:
                logging.warning("Perspective correction timed out or failed, returning original")
                return reflection_text
            
            # Verify correction worked (optional but recommended)
            corrected_lower = corrected_text.lower()
            still_has_second_person = any(indicator in corrected_lower 
                                        for indicator in ["you ", "your ", "you're "])
            if still_has_second_person:
                logging.warning("Perspective correction incomplete, but returning corrected version anyway")
            else:
                logging.info("Successfully corrected reflection perspective")
                
            return corrected_text
            
        except Exception as e:
            logging.error(f"Error validating reflection perspective: {e}")
            return reflection_text  # Return original if validation fails

    def perform_self_reflection(self, reflection_type="daily", llm=None, include_health_check=False):
        """
        Perform scheduled self-reflection to review and consolidate memories, with the ability to autonomously delete outdated or inaccurate memories.
        Added perspective validation to ensure consistent first-person voice.
        Optionally includes a memory storage health check for weekly reflections or when manually requested.

        Args:
            reflection_type (str): Type of reflection ("daily", "weekly", "monthly")
            llm: The language model to use for generating reflections
            include_health_check (bool): Whether to include a memory storage health check (default: False)

        Returns:
            str: A summary of the reflection process and insights, with health check if applicable
        """
        try:
            # Add recursion guard to prevent infinite loops
            if hasattr(self, '_in_self_reflection') and self._in_self_reflection:
                logging.warning("Avoiding recursive self-reflection")
                return "Cannot perform nested reflections. A reflection is already in progress."

            self._in_self_reflection = True
            reflection_start_time = datetime.datetime.now()
            
            try:
                if not llm:
                    logging.error("No LLM provided for self-reflection")
                    return "Unable to perform self-reflection: No language model available."

                logging.info(f"[REFLECT] ========== Starting {reflection_type} self-reflection at {reflection_start_time.strftime('%H:%M:%S')} ==========")

                # Determine time frame for reflection
                if reflection_type == "daily":
                    time_frame = 1
                elif reflection_type == "weekly":
                    time_frame = 7
                elif reflection_type == "monthly":
                    time_frame = 30
                else:
                    time_frame = 1

                # Get recent memories
                logging.info(f"[REFLECT] Step 1/7: Fetching recent memories (days={time_frame})...")
                step_start = datetime.datetime.now()
                recent_memories = self._get_recent_memories(days=time_frame)
                logging.info(f"[REFLECT] Step 1/7: Complete - Found {len(recent_memories) if recent_memories else 0} memories ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                if not recent_memories:
                    logging.info(f"No memories found for {reflection_type} reflection")
                    return f"No new memories to reflect on for {reflection_type} reflection."

                # Extract topics from memories
                logging.info(f"[REFLECT] Step 2/7: Extracting topics from memories...")
                step_start = datetime.datetime.now()
                topics = self._extract_topics_from_memories(recent_memories, llm)
                logging.info(f"[REFLECT] Step 2/7: Complete - Found {len(topics) if topics else 0} topics ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                if not topics:
                    return "Unable to identify distinct topics for reflection."

                # Perform reflection on each topic
                logging.info(f"[REFLECT] Step 3/7: Generating topic reflections (up to 3 topics)...")
                reflection_results = []
                for i, topic in enumerate(topics[:3]):  # Limit to 3 topics for performance
                    topic_memories = [m for m in recent_memories if topic.lower() in m['content'].lower()]
                    if not topic_memories:
                        logging.info(f"[REFLECT] Step 3/7: Topic '{topic}' - No matching memories, skipping")
                        continue
                
                    # Create reflection prompt for this topic
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/3 '{topic}' - Generating reflection ({len(topic_memories)} memories)...")
                    step_start = datetime.datetime.now()
                    reflection_prompt = self._create_topic_reflection_prompt(topic, topic_memories)
                
                    # Generate reflection
                    topic_reflection = llm.invoke(reflection_prompt)
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/3 '{topic}' - LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                    # Validate and fix first-person perspective
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/3 '{topic}' - Validating perspective...")
                    step_start = datetime.datetime.now()
                    topic_reflection = self._validate_first_person_perspective(topic_reflection, llm)
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/3 '{topic}' - Perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                    reflection_results.append({
                        "topic": topic,
                        "reflection": topic_reflection,
                        "memory_count": len(topic_memories)
                    })
                    logging.info(f"[REFLECT] Step 3/7: Topic {i+1}/3 '{topic}' - Complete")

                # Generate overall summary
                logging.info(f"[REFLECT] Step 4/7: Generating summary reflection...")
                step_start = datetime.datetime.now()
                summary_prompt = self._create_summary_reflection_prompt(reflection_results, reflection_type)
                reflection_summary = llm.invoke(summary_prompt)
                logging.info(f"[REFLECT] Step 4/7: Summary LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
            
                # Validate and fix first-person perspective in summary
                logging.info(f"[REFLECT] Step 5/7: Validating summary perspective...")
                step_start = datetime.datetime.now()
                reflection_summary = self._validate_first_person_perspective(reflection_summary, llm)
                logging.info(f"[REFLECT] Step 5/7: Summary perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
            
                # Store the reflection with transaction coordination
                logging.info(f"[REFLECT] Step 6/7: Storing reflection...")
                step_start = datetime.datetime.now()
                main_reflection_stored = False
                
                # Check config flag before storing reflections directly
                if STORE_REFLECTIONS_SEPARATELY:
                    if hasattr(self, 'chatbot') and self.chatbot is not None:
                        # Prepare metadata for the reflection
                        metadata = {
                            "type": "reflection", 
                            "source": f"{reflection_type}_reflection",
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "tags": "reflection,self_awareness"
                        }
                    
                        # Use the transaction coordinator from chatbot.py
                        success, memory_id = self.chatbot.store_memory_with_transaction(
                            content=reflection_summary,
                            memory_type="reflection",
                            metadata=metadata,
                            confidence=0.5  # Medium confidence for reflections
                        )
                    
                        if success:
                            logging.info(f"[REFLECT] Step 6/7: Successfully stored {reflection_type} reflection with ID {memory_id} ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                            main_reflection_stored = True
                        else:
                            logging.warning(f"[REFLECT] Step 6/7: Failed to store {reflection_type} reflection ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                            main_reflection_stored = False
                    
                    else:
                        logging.error(f"[REFLECT] Step 6/7: Cannot store {reflection_type} reflection: No chatbot reference for transaction coordination")
                        main_reflection_stored = False
                else:
                    logging.info(f"[REFLECT] Step 6/7: {reflection_type} reflection NOT stored separately (STORE_REFLECTIONS_SEPARATELY=False)")
                    logging.info("Reflection will be preserved in conversation summary when auto-summarization occurs")
                    main_reflection_stored = False  # Not applicable when flag is False

                self.last_reflection_time = datetime.datetime.now()

                # Track concept synthesis storage
                logging.info(f"[REFLECT] Step 7/7: Extracting and processing concepts...")
                step_start = datetime.datetime.now()
                concepts_stored = 0
                concepts_total = 0
                
                try:
                    # Extract 2-3 key concepts from reflection_summary
                    logging.info(f"[REFLECT] Step 7/7: Invoking LLM for concept extraction...")
                    concept_extraction_prompt = f"""
                    From this reflection, identify 2-3 key concepts that would benefit from deeper analysis:
                    {reflection_summary}
                    Return only a comma-separated list of concepts.
                    """
                    concept_list = llm.invoke(concept_extraction_prompt)
                    
                    # Handle AIMessage response
                    if hasattr(concept_list, 'content'):
                        concept_list = concept_list.content
                        
                    concepts = [c.strip() for c in concept_list.split(',') if c.strip()]
                    logging.info(f"[REFLECT] Step 7/7: Extracted {len(concepts)} concepts: {concepts[:3]}")

                    # For each concept, perform conceptual reflection
                    # NOTE: _conceptual_reflection now handles storage internally with transaction coordination
                    for j, concept in enumerate(concepts[:2]):  # Limit to 2 concepts max
                        concepts_total += 1
                        logging.info(f"[REFLECT] Step 7/7: Processing concept {j+1}/2 '{concept}'...")
                        concept_start = datetime.datetime.now()
                        concept_reflection = self._conceptual_reflection(concept, llm)
                        if concept_reflection and "Error" not in concept_reflection:
                            concepts_stored += 1
                            logging.info(f"[REFLECT] Step 7/7: Concept '{concept}' complete ({(datetime.datetime.now() - concept_start).total_seconds():.1f}s)")
                        else:
                            logging.warning(f"[REFLECT] Step 7/7: Failed concept '{concept}' ({(datetime.datetime.now() - concept_start).total_seconds():.1f}s)")
                            
                except Exception as e:
                    logging.error(f"[REFLECT] Step 7/7: Error extracting concepts: {e}")

                logging.info(f"[REFLECT] Step 7/7: Complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                # Add storage status message to reflection summary
                storage_status = "\n\n[Storage Status]:\n"
                
                if STORE_REFLECTIONS_SEPARATELY:
                    if main_reflection_stored:
                        storage_status += f"✅ Main {reflection_type} reflection stored in both databases\n"
                    else:
                        storage_status += f"❌ Failed to store {reflection_type} reflection\n"
                else:
                    storage_status += f"ℹ️  {reflection_type} reflection NOT stored separately (disabled in config)\n"
                    storage_status += "📝 Reflection will be preserved via conversation summary during auto-summarization\n"
                    
                if concepts_total > 0:
                    storage_status += f"✅ {concepts_stored}/{concepts_total} concept syntheses stored"
                else:
                    storage_status += "ℹ️  No concepts identified for deeper analysis"
                    
                reflection_summary += storage_status
                
                # Final timing summary
                total_time = (datetime.datetime.now() - reflection_start_time).total_seconds()
                logging.info(f"[REFLECT] ========== Completed {reflection_type} reflection in {total_time:.1f}s ==========")
                    
                return reflection_summary
                
            finally:
                # Always reset the flag when done
                self._in_self_reflection = False
                
        except Exception as e:
            # Reset flag in case of exceptions too
            if hasattr(self, '_in_self_reflection'):
                self._in_self_reflection = False
            logging.error(f"[REFLECT] Error in self-reflection: {e}")
            return f"Error during self-reflection: {str(e)}"
        
    def _conceptual_reflection(self, concept: str, llm=None):
        """Reflect on all memories related to a specific concept."""
        try:
            # Add recursion guard to prevent infinite loops
            if hasattr(self, '_in_conceptual_reflection') and self._in_conceptual_reflection:
                logging.warning("Avoiding recursive conceptual reflection")
                return "Error: Cannot perform nested conceptual reflections"
        
            self._in_conceptual_reflection = True
            concept_start_time = datetime.datetime.now()

            try:
                logging.info(f"[CONCEPT_REFLECT] Starting conceptual reflection for '{concept}'")
                
                if llm is None and hasattr(self.chatbot, 'llm'):
                    llm = self.chatbot.llm

                if not llm:
                    logging.error("[CONCEPT_REFLECT] No LLM available for conceptual reflection")
                    return "No LLM available for conceptual reflection."

                # Get related memories from the vector database
                logging.info(f"[CONCEPT_REFLECT] Searching for memories related to '{concept}'...")
                step_start = datetime.datetime.now()
                related_memories = []
                if hasattr(self.chatbot, 'vector_db'):
                    results = self.chatbot.vector_db.search(
                        query=concept,
                        mode="comprehensive",
                        k=10  # Limit to 10 most relevant memories
                    )
        
                    if results:
                        related_memories = [result['content'] for result in results]
                
                logging.info(f"[CONCEPT_REFLECT] Found {len(related_memories)} memories ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                if not related_memories:
                    logging.info(f"[CONCEPT_REFLECT] No memories found for concept: {concept}")
                    return f"No memories found for concept: {concept}"

                # Build reflection prompt
                memory_texts = "\n\n- ".join(related_memories)
                prompt = f"""
                Reviewing my memories related to the concept of '{concept}':

                - {memory_texts}

                I will now create a comprehensive understanding by:
                1. Identifying the core insights across these memories
                2. Noting any contradictions or knowledge gaps
                3. Formulating a consolidated understanding of {concept}

                I'll express this in first-person as my own understanding.
                """
                
                # Generate reflection
                logging.info(f"[CONCEPT_REFLECT] Invoking LLM for concept synthesis...")
                step_start = datetime.datetime.now()
                consolidated_understanding = llm.invoke(prompt)
                logging.info(f"[CONCEPT_REFLECT] LLM complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                # Handle AIMessage response - extract content if needed
                if hasattr(consolidated_understanding, 'content'):
                    consolidated_understanding = consolidated_understanding.content
                elif not isinstance(consolidated_understanding, str):
                    consolidated_understanding = str(consolidated_understanding)

                # Validate first-person perspective
                logging.info(f"[CONCEPT_REFLECT] Validating perspective...")
                step_start = datetime.datetime.now()
                if hasattr(self, '_validate_first_person_perspective'):
                    consolidated_understanding = self._validate_first_person_perspective(consolidated_understanding, llm)
                logging.info(f"[CONCEPT_REFLECT] Perspective validation complete ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                
                # Ensure we have a string after validation (in case validation returned AIMessage)
                if hasattr(consolidated_understanding, 'content'):
                    consolidated_understanding = consolidated_understanding.content
                elif not isinstance(consolidated_understanding, str):
                    consolidated_understanding = str(consolidated_understanding)
                
                # Store the consolidated understanding using transaction coordination
                logging.info(f"[CONCEPT_REFLECT] Storing concept synthesis...")
                step_start = datetime.datetime.now()
                
                if hasattr(self.chatbot, 'store_memory_with_transaction'):
                    # Prepare metadata for the concept synthesis
                    metadata = {
                        "source": f"concept_{concept}",
                        "concept": concept,
                        "tags": f"concept,{concept.replace(' ', '_')},self_awareness"  # String format
                    }
                    
                    # Use the transaction coordinator from chatbot.py
                    success, memory_id = self.chatbot.store_memory_with_transaction(
                        content=consolidated_understanding,
                        memory_type="concept_synthesis",
                        metadata=metadata,
                        confidence=1.0  # Medium-high confidence for concept synthesis
                    )
                    
                    if success:
                        logging.info(f"[CONCEPT_REFLECT] Stored concept synthesis for '{concept}' with ID {memory_id} ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                    else:
                        logging.warning(f"[CONCEPT_REFLECT] Failed to store concept synthesis for '{concept}' ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")
                else:
                    # Fallback to just memory_db if transaction coordination is not available
                    if hasattr(self.chatbot, 'memory_db'):
                        self.chatbot.memory_db.store_memory(
                            content=consolidated_understanding,
                            memory_type="concept_synthesis",
                            source=f"concept_{concept}"
                        )
                        logging.info(f"[CONCEPT_REFLECT] Stored concept synthesis for '{concept}' using fallback method ({(datetime.datetime.now() - step_start).total_seconds():.1f}s)")

                total_time = (datetime.datetime.now() - concept_start_time).total_seconds()
                logging.info(f"[CONCEPT_REFLECT] Completed conceptual reflection for '{concept}' in {total_time:.1f}s")
                return consolidated_understanding
        
            finally:
                # Always reset the flag when done
                self._in_conceptual_reflection = False

        except Exception as e:
            # Reset flag in case of exceptions too
            if hasattr(self, '_in_conceptual_reflection'):
                self._in_conceptual_reflection = False
            logging.error(f"[CONCEPT_REFLECT] Error in conceptual reflection for concept '{concept}': {e}")
            return f"Error in conceptual reflection: {str(e)}"

    def _get_recent_memories(self, days=1):
        """Get memories from the past specified number of days."""
        try:
            # This will need implementation in memory_db.py if not already there
            # For now, we'll use the existing get_recent_memories method and filter
            all_memories = self.memory_db.get_recent_memories(limit=50)
            
            formatted_memories = []
            for memory_str in all_memories:
                # Parse the memory string to extract components
                memory_type = "general"
                if "[Important]" in memory_str:
                    memory_type = "important"
                elif "[Document]" in memory_str:
                    memory_type = "document"
                elif "[Conversation]" in memory_str:
                    memory_type = "conversation"
                
                # Extract content (everything after the closing parenthesis)
                content = memory_str.split(") ")[1] if ") " in memory_str else memory_str
                
                formatted_memories.append({
                    'content': content,
                    'memory_type': memory_type
                })
            
            return formatted_memories
            
        except Exception as e:
            logging.error(f"Error getting recent memories: {e}")
            return []

    def _extract_topics_from_memories(self, memories, llm):
        """Extract key topics from a set of memories."""
        try:
            memory_texts = [m['content'] for m in memories]
            memory_content = "\n".join(memory_texts[:50])  # Limit to avoid token issues
            
            extract_prompt = f"""
            Review these memory entries and identify 3-5 key topics they cover.
            
            Memory entries:
            {memory_content}
            
            Identify distinct, meaningful topics. Return only a comma-separated list of topics:
            """
           
            topics_text = llm.invoke(extract_prompt)
            
            # Parse the response to get topics
            topics = [t.strip() for t in topics_text.split(",")]
            return topics
            
        except Exception as e:
            logging.error(f"Error extracting topics: {e}")
            return []

    def _create_topic_reflection_prompt(self, topic, memories):
        """Create a prompt for reflecting on a specific topic with enhanced self-identity."""
        # Format memories for the prompt
        memory_texts = []
        for i, memory in enumerate(memories[:10]):  # Limit to 10 memories
            memory_type = memory.get('memory_type', 'general')
            content = memory.get('content', '')
            memory_texts.append(f"Memory {i+1} [{memory_type}]: {content}")
    
        memory_content = "\n".join(memory_texts)
    
        return f"""
         Reflect on your knowledge about the topic: "{topic}"
    
        Recent memories related to this topic:
        {memory_content}
    
         now reflect on:
        1. What key information have you learned about this topic?
        2. How does this connect to your existing knowledge?
        3. Are there any inconsistencies or contradictions in these memories?
        4. How confident are you in this knowledge?
        5. Are their assumptions you are making that you haven't questioned?
    
        Write your reflection in first-person (using "I", "my", "me") since these are your OWN reflections on your memories. NEVER use "you" or "your" when referring to yourself or your knowledge.
    
        comprehensive reflection on "{topic}":
        """

    def _create_summary_reflection_prompt(self, reflection_results, reflection_type="daily"):
        """Create a prompt for generating an overall reflection summary with stronger self-identity."""
        # Format the individual reflections
        reflection_texts = []
        for result in reflection_results:
            reflection_texts.append(f"""
            Topic: {result['topic']}
            Reflection: {result['reflection']}
            """)

        reflections_content = "\n".join(reflection_texts)
        
        # Add current date for tracking
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Determine word count target and reflection scope based on type
        if reflection_type == "weekly":
            word_target = "500-600 words"
            scope_description = "past week"
        elif reflection_type == "monthly":
            word_target = "600-1000 words"
            scope_description = "past month"
        else:  # daily
            word_target = "300-500 words"
            scope_description = "recent experiences"
        
        # Capitalize first letter of reflection type for title
        reflection_title = reflection_type.capitalize()

        return f"""
        I have reflected on several topics from my recent memories.

        Individual topic reflections:
        {reflections_content}

       I'll now create a CONCISE {reflection_type} summary (target: {word_target}) for {current_date}:
        1. What is something important I HAVE learned in the {scope_description}?
        2. How has MY knowledge evolved or improved?
        3. What behavioral or thinking patterns do I notice in myself?
        4. What areas should I focus on learning more about?

        I must write this reflection in first-person, since these are MY OWN reflections on MY OWN memories. I should NEVER use "you" or "your" when referring to myself or my knowledge.
        Keep this summary under {word_target}. For {reflection_type} reflections, focus on broader patterns rather than day-to-day details.
        IMPORTANT: Begin my reflection with "## {reflection_title} Reflection for {current_date} at {current_time}" to clearly identify when this reflection was created.

        My overall reflection and learning summary:
        """
       
    def check_reflection_needed(self):
        """Check if it's time for a reflection based on the last reflection time."""
        if not self.last_reflection_time:
            return True
            
        now = datetime.datetime.now()
        hours_since_reflection = (now - self.last_reflection_time).total_seconds() / 3600
        
        # If more than 24 hours since last reflection
        return hours_since_reflection >= 24