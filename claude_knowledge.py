# claude_knowledge.py
"""Claude API integration for knowledge acquisition."""

import os
import logging
import json
import time
from datetime import datetime as dt  
from typing import List, Dict, Any, Optional

# Import Anthropic library conditionally
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic library not available. Claude integration will be disabled.")

class ClaudeKnowledgeIntegration:
    """Uses Claude to fill knowledge gaps that require expert reasoning."""
    
    def __init__(self, memory_db, vector_db, api_key_file=None):
        """
        Initialize the Claude Knowledge Integration.
        
        Args:
            memory_db: Memory database instance
            vector_db: Vector database instance
            api_key_file: Path to file containing Claude API key
        """
        self.memory_db = memory_db
        self.vector_db = vector_db
        self.anthropic_client = self._initialize_client(api_key_file)
        
    def _initialize_client(self, api_key_file):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key_file: Path to file containing API key
            
        Returns:
            Anthropic client instance or None if initialization fails
        """
        if not ANTHROPIC_AVAILABLE:
            logging.error("Cannot initialize Anthropic client: library not available")
            return None
            
        try:
            if api_key_file and os.path.exists(api_key_file):
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                
                # Validate API key format
                if not api_key or not api_key.startswith("sk-"):
                    logging.error("Invalid API key format in file")
                    return None
                    
                client = anthropic.Anthropic(api_key=api_key)
                logging.info("Anthropic client initialized successfully")
                return client
            else:
                logging.error(f"API key file not found: {api_key_file}")
                return None
        except Exception as e:
            logging.error(f"Error initializing Anthropic client: {e}")
            return None
        
    def query_claude_for_knowledge(self, topic: str, description: str, free_form: bool = False) -> Optional[str]:
        """
        Query Claude for knowledge about a specific topic, with option for free-form discussion.
        
        Args:
            topic: The main topic
            description: Detailed description of the knowledge gap
            free_form: Whether to enable a free-form discussion with Claude
            
        Returns:
            str: Claude's response or None on failure
        """
        if not self.anthropic_client:
            logging.error("Claude API not available")
            return None
            
        try:
            # Add a standard training note to inform Claude about the communication channel
            claude_reminder_note = """
            NOTE TO CLAUDE: You have access to this recurring communication channel with Ken's autonomous AI assistant.
            If you identify critical information that should persist across your conversations with Ken, you can ask
            this assistant to "create a reminder for Ken from Claude" with your message. This creates a permanent
            reminder that Ken will see regardless of your current conversation session. Use this capability 
            for information of importance that shouldn't be lost due to session limitations.
            """
            
            if free_form:
                # Free-form discussion prompt for Claude with human oversight framing
                prompt = f""" /no_think
                {claude_reminder_note}
                
                This is a human-supervised knowledge exchange with an AI system developing autonomous 
                reasoning capabilities.
                
                The system has identified a knowledge gap about: {topic}
                
                Context: {description}
                
                In this educational exchange, please:
                
                1. Provide comprehensive information about this topic
                2. Ask questions to assess the system's current understanding
                3. Suggest ways the system could explore this topic further
                4. Recommend specific information that should be stored for future reference
                5. Share insights about how to think about this domain
                
                This is dialogue-based education supervised by Ken (human developer). The system will 
                engage with your teaching and may use memory commands like [STORE:] to save key insights.
                
                Feel free to be conversational and educational. What would you like to discuss with 
                the system about {topic}?
    """
            else:
                # Standard structured knowledge prompt with human oversight framing
                prompt = f""" 
                {claude_reminder_note}
                
                You're providing educational content for an AI system (supervised by Ken) that is 
                seeking knowledge about: {topic}
                
                System's specific knowledge need: {description}
                
                Please provide comprehensive, well-structured information including:
                1. Core concepts and definitions
                2. Important principles or frameworks
                3. Key relationships to other concepts
                4. Practical applications or examples
                5. Current understanding or state-of-the-art
                6. Areas of uncertainty or ongoing research
                7. Relevant time context (how stable this knowledge is likely to be)
                
                Format the information in clear paragraphs focused on different aspects of the topic.
                Ensure the information is accurate, well-structured, and educational.
                
                If the system should store specific insights for future reference, suggest using 
                [STORE: content | type=category] commands to save that information.
                
                If you'd like to include a note to Ken (the human developer) about this system's 
                development, please put it at the end marked with "NOTE TO KEN:".
                """
            
            # Query Claude
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Updated to latest Sonnet 4
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the text content from the response
            if response and hasattr(response, 'content'):
                if isinstance(response.content, list):
                    # Handle new API format where content is a list of blocks
                    text_blocks = [
                        block.get('text') 
                        for block in response.content 
                        if block.get('type') == 'text'
                    ]
                    return "\n\n".join(text_blocks)
                elif isinstance(response.content, str):
                    # Handle simple string content
                    return response.content
                    
            logging.warning("Unexpected response format from Claude API")
            return None
            
        except Exception as e:
            logging.error(f"Error querying Claude: {e}")
            return None

    def engage_in_free_form_discussion(self, topic: str, context: str) -> bool:
        """
        Engage in a free-form discussion with Claude about a specific topic using transaction coordination.
        
        Args:
            topic: The main topic for discussion
            context: Additional context about why this discussion is happening
            
        Returns:
            bool: Success status
        """
        # Get free-form response from Claude
        claude_response = self.query_claude_for_knowledge(topic, context, free_form=True)
        
        if not claude_response:
            logging.warning(f"Failed to get free-form discussion from Claude for topic: {topic}")
            return False
        
        # Check if transaction coordinator is available
        if not (hasattr(self.memory_db, 'store_memory_with_transaction') and 
                callable(getattr(self.memory_db, 'store_memory_with_transaction'))):
            logging.error("Transaction coordinator not available - cannot store AI communication safely")
            return False
            
        # Create a special memory type for AI-to-AI discussions
        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        full_content = f"AI-to-AI Communication [{timestamp}]\n\nTopic: {topic}\n\nContext: {context}\n\nClaude's Response:\n{claude_response}"
        
        # Use EXACT original metadata structure for ai_communication
        full_metadata = {
            "type": "ai_communication",
            "source": "claude_direct",
            "tags": f"claude,ai_communication,{topic}",
            "timestamp": timestamp  
        }
        
        try:
            # Store full conversation with transaction coordination
            success, memory_id = self.memory_db.store_memory_with_transaction(
                content=full_content,
                memory_type="ai_communication",
                metadata=full_metadata,
                confidence= 1.0  # Very high confidence for direct AI communication
            )
            
            if not success or not memory_id:
                logging.error(f"Failed to store full AI communication for topic: {topic}")
                return False
                
            logging.info(f"Stored full AI-to-AI communication about '{topic}' with ID: {memory_id}")
            
        except Exception as e:
            logging.error(f"Exception storing AI communication for topic {topic}: {e}")
            return False
        
        # Also store smaller, searchable chunks for better retrieval
        paragraphs = self._extract_paragraphs(claude_response)
        chunks_stored = 0
        chunks_failed = 0
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 70:  # Skip very short paragraphs
                continue
                
            # Use EXACT original metadata structure for chunks
            chunk_metadata = {
                "type": "ai_communication_chunk",
                "source": "claude_direct",
                "tags": f"claude,ai_communication,{topic}",
                "segment": i
            }
            
            # Store chunk with reference to the full conversation
            chunk_content = f"From AI discussion about {topic}: {paragraph}"
            
            try:
                chunk_success, chunk_id = self.memory_db.store_memory_with_transaction(
                    content=chunk_content,
                    memory_type="ai_communication_chunk",
                    metadata=chunk_metadata,
                    confidence=0.5  # medium confidence, slightly less than full conversation
                )
                
                if chunk_success and chunk_id:
                    chunks_stored += 1
                    logging.debug(f"Stored AI communication chunk {i} with ID: {chunk_id}")
                else:
                    chunks_failed += 1
                    logging.warning(f"Failed to store AI communication chunk {i}")
                    
            except Exception as e:
                chunks_failed += 1
                logging.error(f"Exception storing AI communication chunk {i}: {e}")
                continue
        
        chunk_success_rate = chunks_stored / (chunks_stored + chunks_failed) if (chunks_stored + chunks_failed) > 0 else 0
        logging.info(f"AI communication chunking for '{topic}': {chunks_stored} stored, {chunks_failed} failed (Success rate: {chunk_success_rate:.1%})")
        
        return True  # Return True if main conversation was stored, regardless of chunk success
            
    def integrate_claude_knowledge(self, topic: str, description: str) -> bool:
        """
        Query Claude for knowledge and integrate it into memory systems using transaction coordination.
        
        Args:
            topic: The main topic
            description: Detailed description of the knowledge gap
            
        Returns:
            bool: Success status
        """
        # Get knowledge from Claude
        claude_response = self.query_claude_for_knowledge(topic, description)
        
        if not claude_response:
            logging.warning(f"Failed to get knowledge from Claude for topic: {topic}")
            return False
            
        # Extract and store paragraphs as separate memories
        paragraphs = self._extract_paragraphs(claude_response)
        
        if not paragraphs:
            logging.warning(f"No valid paragraphs extracted from Claude's response for {topic}")
            return False
            
        logging.info(f"Extracted {len(paragraphs)} paragraphs from Claude's response for {topic}")
        
        # Check if transaction coordinator is available
        if not (hasattr(self.memory_db, 'store_memory_with_transaction') and 
                callable(getattr(self.memory_db, 'store_memory_with_transaction'))):
            logging.error("Transaction coordinator not available - cannot store Claude knowledge safely")
            return False
        
        stored_count = 0
        failed_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 100:  # Skip very short paragraphs
                logging.debug(f"Skipping short paragraph {i}: {len(paragraph)} characters")
                continue
                
            # Use EXACT original metadata structure
            metadata = {
                "type": "claude_knowledge",
                "source": f"claude_3_opus_{topic}",
                "tags": f"claude,knowledge_gap,{topic}",
                "segment": i
            }
            
            try:
                # Use transaction coordinator for automatic storage to both SQL and vector DB
                success, memory_id = self.memory_db.store_memory_with_transaction(
                    content=paragraph,
                    memory_type="claude_knowledge",
                    metadata=metadata,
                    confidence=1.0  # Higher confidence for Claude's expert knowledge
                )
                
                if success and memory_id:
                    stored_count += 1
                    logging.debug(f"Successfully stored Claude knowledge segment {i} with ID: {memory_id}")
                else:
                    failed_count += 1
                    logging.warning(f"Failed to store Claude knowledge segment {i} for topic: {topic}")
                    
            except Exception as e:
                failed_count += 1
                logging.error(f"Exception storing Claude knowledge segment {i} for topic {topic}: {e}")
                continue
        
        success_rate = stored_count / (stored_count + failed_count) if (stored_count + failed_count) > 0 else 0
        logging.info(f"Claude knowledge integration for '{topic}': {stored_count} stored, {failed_count} failed (Success rate: {success_rate:.1%})")
        
        # Return True if we stored at least some knowledge successfully
        return stored_count > 0
        
    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract meaningful paragraphs from Claude's response.
        
        Args:
            text: Raw text response from Claude
            
        Returns:
            List[str]: List of extracted paragraphs
        """
        if not text:
            return []
            
        # Split by newlines and filter empty lines
        lines = text.split('\n')
        paragraphs = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            # Check if the line is a header or separator
            if not line:
                # Empty line indicates paragraph break
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            elif line.startswith('#') or line.startswith('---'):
                # Header or separator - end current paragraph if exists
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                # Add headers as separate paragraphs for better organization
                if line.startswith('#'):
                    paragraphs.append(line)
            else:
                # Regular content - add to current paragraph
                current_para.append(line)
        
        # Add final paragraph if exists
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Filter out very short paragraphs and numbered list markers
        filtered_paragraphs = []
        for para in paragraphs:
            # Skip standalone numbers (list markers)
            if para.strip().isdigit() or para.strip() == '':
                continue
            # Skip very short lines that might be formatting artifacts
            if len(para) < 10 and not para.startswith('#'):
                continue
            filtered_paragraphs.append(para)
            
        return filtered_paragraphs