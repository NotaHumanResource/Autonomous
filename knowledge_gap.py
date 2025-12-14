# knowledge_gap.py - Updated version with semantic similarity checking
"""Knowledge gap tracking and management for autonomous learning with duplicate prevention."""

import sqlite3
import logging
import datetime
import uuid
from typing import List, Tuple, Dict, Any, Optional

# Qdrant imports for semantic similarity
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_ollama import OllamaEmbeddings


class KnowledgeGapQueue:
    """Manages and prioritizes identified knowledge gaps with semantic duplicate prevention."""
    
    # Class-level constants for semantic similarity
    SIMILARITY_THRESHOLD = 0.80  # 80% similarity = duplicate
    EMBEDDING_MODEL = "qwen3-embedding:4b"
    
    def __init__(self, db_path, vector_db_url: str = "http://localhost:6333", 
                 gaps_collection_name: str = None):
        """
        Initialize with the database path and optional Qdrant configuration.
        
        Args:
            db_path: Path to SQLite database
            vector_db_url: URL for Qdrant server (default: localhost:6333)
            gaps_collection_name: Name of Qdrant collection for gap embeddings
        """
        self.db_path = db_path
        self.vector_db_url = vector_db_url
        
        # Import collection name from config, with fallback
        try:
            from config import QDRANT_GAPS_COLLECTION_NAME, OLLAMA_BASE_URL
            self.gaps_collection_name = gaps_collection_name or QDRANT_GAPS_COLLECTION_NAME
            self.ollama_base_url = OLLAMA_BASE_URL
        except ImportError:
            # Fallback defaults if config not available
            self.gaps_collection_name = gaps_collection_name or "knowledge_gaps_embeddings"
            self.ollama_base_url = "http://localhost:11434"
            logging.warning("Could not import from config, using defaults")
        
        # Initialize SQL database
        self._initialize_db()
        
        # Initialize Qdrant client and embeddings (lazy loading)
        self._qdrant_client = None
        self._embeddings = None
        self._collection_initialized = False
        
    @property
    def qdrant_client(self):
        """Lazy initialization of Qdrant client."""
        if self._qdrant_client is None:
            try:
                self._qdrant_client = QdrantClient(url=self.vector_db_url, timeout=30.0)
                logging.info(f"Connected to Qdrant at {self.vector_db_url}")
            except Exception as e:
                logging.error(f"Failed to connect to Qdrant: {e}")
                raise
        return self._qdrant_client
    
    @property
    def embeddings(self):
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=self.EMBEDDING_MODEL,
                    base_url=self.ollama_base_url
                )
                logging.info(f"Initialized {self.EMBEDDING_MODEL} embeddings")
            except Exception as e:
                logging.error(f"Failed to initialize embeddings: {e}")
                raise
        return self._embeddings
        
    def _initialize_db(self):
        """Create the knowledge gaps table if it doesn't exist and ensure all columns are present."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the base table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_gaps (
                        id INTEGER PRIMARY KEY,
                        topic TEXT NOT NULL,
                        description TEXT,
                        priority FLOAT DEFAULT 0.5,
                        status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        fulfilled_at DATETIME
                    )
                ''')
                
                # Check existing columns
                cursor.execute("PRAGMA table_info(knowledge_gaps)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Add missing columns if they don't exist
                required_columns = {
                    'items_acquired': 'INTEGER DEFAULT 0',
                    'last_attempt_at': 'DATETIME',
                    'vector_id': 'TEXT'  # NEW: Store Qdrant point ID for cleanup
                }
                
                for column_name, column_def in required_columns.items():
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f'ALTER TABLE knowledge_gaps ADD COLUMN {column_name} {column_def}')
                            logging.info(f"Added missing column '{column_name}' to knowledge_gaps table")
                        except Exception as e:
                            logging.warning(f"Could not add column '{column_name}': {e}")
                
                # Add indexes for faster querying
                try:
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status_priority
                        ON knowledge_gaps(status, priority)
                    ''')
                except Exception as e:
                    logging.warning(f"Could not create status_priority index: {e}")
                
                try:
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_topic
                        ON knowledge_gaps(topic)
                    ''')
                except Exception as e:
                    logging.warning(f"Could not create topic index: {e}")
                
                conn.commit()
                logging.info("Knowledge gaps table initialized successfully with all required columns")
                
        except Exception as e:
            logging.error(f"Error initializing knowledge gaps table: {e}")
            raise

    def _ensure_gaps_collection_exists(self) -> bool:
        """
        Ensure the Qdrant collection for knowledge gap embeddings exists.
        Creates it if necessary with proper vector configuration.
        
        Returns:
            bool: True if collection exists or was created successfully
        """
        if self._collection_initialized:
            return True
            
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.gaps_collection_name for c in collections)
            
            if collection_exists:
                logging.info(f"Knowledge gaps collection '{self.gaps_collection_name}' already exists")
                self._collection_initialized = True
                return True
            
            # Get embedding dimension from a sample embedding
            sample_embedding = self.embeddings.embed_documents(["test dimension check"])
            vector_dimension = len(sample_embedding[0])
            
            # Create the collection
            self.qdrant_client.create_collection(
                collection_name=self.gaps_collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_dimension,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            logging.info(f"Created knowledge gaps collection '{self.gaps_collection_name}' "
                        f"with dimension {vector_dimension}")
            self._collection_initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error ensuring gaps collection exists: {e}")
            return False

    def _generate_embedding(self, topic: str, description: str) -> Optional[List[float]]:
        """
        Generate an embedding for a knowledge gap.
        
        Args:
            topic: The gap topic
            description: The gap description
            
        Returns:
            List[float]: The embedding vector, or None on failure
        """
        try:
            # Combine topic and description for richer semantic representation
            combined_text = f"{topic}: {description}"
            
            # Generate embedding
            embedding = self.embeddings.embed_documents([combined_text])
            
            if embedding and len(embedding) > 0:
                return embedding[0]
            else:
                logging.warning("Empty embedding returned")
                return None
                
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None

    def check_semantic_similarity(self, topic: str, description: str, 
                                   threshold: float = None) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a semantically similar knowledge gap already exists using vector similarity.
        
        Args:
            topic: The topic of the new gap
            description: The description of the new gap
            threshold: Similarity threshold (default: class SIMILARITY_THRESHOLD)
            
        Returns:
            Tuple[bool, Optional[Dict]]: 
                - (True, similar_gap_info) if similar gap exists
                - (False, None) if no similar gap found
        """
        threshold = threshold or self.SIMILARITY_THRESHOLD
        
        try:
            # Ensure collection exists
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection exists, skipping semantic check")
                return False, None
            
            # Check if collection has any points
            collection_info = self.qdrant_client.get_collection(self.gaps_collection_name)
            if collection_info.points_count == 0:
                logging.debug("Gaps collection is empty, no semantic duplicates possible")
                return False, None
            
            # Generate embedding for the new gap
            embedding = self._generate_embedding(topic, description)
            if embedding is None:
                logging.warning("Could not generate embedding, skipping semantic check")
                return False, None
            
            # Search for similar gaps
            search_results = self.qdrant_client.search(
                collection_name=self.gaps_collection_name,
                query_vector=embedding,
                limit=3,  # Check top 3 similar gaps
                score_threshold=threshold,  # Only return results above threshold
                with_payload=True
            )
            
            if search_results:
                # Found semantically similar gap(s)
                top_match = search_results[0]
                similar_gap_info = {
                    'score': top_match.score,
                    'topic': top_match.payload.get('topic', 'Unknown'),
                    'description': top_match.payload.get('description', ''),
                    'gap_id': top_match.payload.get('gap_id'),
                    'vector_id': str(top_match.id)
                }
                
                logging.info(f"🔍 Semantic duplicate detected! "
                           f"New: '{topic}' similar to existing: '{similar_gap_info['topic']}' "
                           f"(similarity: {top_match.score:.2%})")
                
                return True, similar_gap_info
            
            # No similar gaps found
            logging.debug(f"No semantic duplicates found for '{topic}'")
            return False, None
            
        except Exception as e:
            logging.error(f"Error checking semantic similarity: {e}")
            # On error, allow the gap to be created (fail open)
            return False, None

    def _store_gap_embedding(self, gap_id: int, topic: str, description: str) -> Optional[str]:
        """
        Store the embedding for a knowledge gap in Qdrant.
        
        Args:
            gap_id: The SQL ID of the gap
            topic: The gap topic
            description: The gap description
            
        Returns:
            str: The Qdrant point ID, or None on failure
        """
        try:
            # Ensure collection exists
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection exists")
                return None
            
            # Generate embedding
            embedding = self._generate_embedding(topic, description)
            if embedding is None:
                return None
            
            # Create unique point ID
            vector_id = str(uuid.uuid4())
            
            # Store in Qdrant with metadata
            self.qdrant_client.upsert(
                collection_name=self.gaps_collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload={
                            'gap_id': gap_id,
                            'topic': topic,
                            'description': description[:500],  # Truncate for payload size
                            'status': 'pending',
                            'created_at': datetime.datetime.now().isoformat()
                        }
                    )
                ]
            )
            
            logging.info(f"Stored embedding for gap {gap_id} with vector_id {vector_id}")
            return vector_id
            
        except Exception as e:
            logging.error(f"Error storing gap embedding: {e}")
            return None

    def _remove_gap_embedding(self, vector_id: str) -> bool:
        """
        Remove a gap embedding from Qdrant.
        
        Args:
            vector_id: The Qdrant point ID to remove
            
        Returns:
            bool: True if successful
        """
        try:
            if not vector_id:
                return True  # Nothing to remove
                
            self.qdrant_client.delete(
                collection_name=self.gaps_collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[vector_id]
                )
            )
            
            logging.info(f"Removed gap embedding with vector_id {vector_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing gap embedding: {e}")
            return False

    def add_gap(self, topic: str, description: str, priority: float = 0.5,
                skip_semantic_check: bool = False) -> int:
        """
        Add a new knowledge gap to the queue with semantic duplicate prevention.
        
        Args:
            topic: The knowledge topic (main subject)
            description: Detailed description of what needs to be learned
            priority: Importance value from 0.0-1.0
            skip_semantic_check: If True, skip the semantic similarity check
            
        Returns:
            int: ID of the newly created gap, -1 if duplicate, -2 on error
        """
        try:
            # STEP 1: Check for semantic duplicates (unless skipped)
            if not skip_semantic_check:
                is_duplicate, similar_info = self.check_semantic_similarity(topic, description)
                if is_duplicate:
                    logging.info(f"⚠️ Rejected duplicate gap '{topic}' - "
                               f"similar to existing gap '{similar_info['topic']}' "
                               f"(similarity: {similar_info['score']:.2%})")
                    
                    # Optionally boost priority of existing similar gap
                    if similar_info.get('gap_id'):
                        self._boost_gap_priority(similar_info['gap_id'], priority)
                    
                    return -1  # Indicate duplicate rejection
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # STEP 2: Check for exact topic match in SQL (fast check)
                cursor.execute('''
                    SELECT id, priority FROM knowledge_gaps
                    WHERE topic = ? AND status = 'pending'
                ''', (topic,))
                
                existing = cursor.fetchone()
                if existing:
                    # Update priority and description if already exists
                    gap_id, existing_priority = existing
                    new_priority = max(existing_priority, priority)
                    
                    cursor.execute('''
                        UPDATE knowledge_gaps 
                        SET priority = ?, description = ?, last_attempt_at = NULL
                        WHERE id = ?
                    ''', (new_priority, description, gap_id))
                    
                    logging.info(f"Updated existing knowledge gap '{topic}' (ID: {gap_id}) "
                               f"with priority {new_priority}")
                    conn.commit()
                    return gap_id
                
                # STEP 3: Insert new gap into SQL
                cursor.execute('''
                    INSERT INTO knowledge_gaps (topic, description, priority)
                    VALUES (?, ?, ?)
                ''', (topic, description, priority))
                gap_id = cursor.lastrowid
                
                # STEP 4: Store embedding in Qdrant
                vector_id = self._store_gap_embedding(gap_id, topic, description)
                
                # Update SQL record with vector_id for later cleanup
                if vector_id:
                    cursor.execute('''
                        UPDATE knowledge_gaps SET vector_id = ? WHERE id = ?
                    ''', (vector_id, gap_id))
                
                conn.commit()
                logging.info(f"✅ Added new knowledge gap '{topic}' with ID {gap_id} "
                           f"and priority {priority}")
                
                return gap_id
                
        except Exception as e:
            logging.error(f"Error adding knowledge gap '{topic}': {e}")
            return -2

    def _boost_gap_priority(self, gap_id: int, boost_priority: float):
        """
        Boost the priority of an existing gap when a duplicate is detected.
        
        Args:
            gap_id: ID of the gap to boost
            boost_priority: The priority of the rejected duplicate to consider
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current priority
                cursor.execute('SELECT priority FROM knowledge_gaps WHERE id = ?', (gap_id,))
                result = cursor.fetchone()
                
                if result:
                    current_priority = result[0]
                    # Increase priority slightly when duplicates are detected
                    # This indicates the topic is coming up repeatedly
                    new_priority = min(1.0, max(current_priority, boost_priority) + 0.05)
                    
                    cursor.execute('''
                        UPDATE knowledge_gaps SET priority = ? WHERE id = ?
                    ''', (new_priority, gap_id))
                    conn.commit()
                    
                    logging.debug(f"Boosted gap {gap_id} priority from {current_priority:.2f} "
                                f"to {new_priority:.2f} due to duplicate detection")
                                
        except Exception as e:
            logging.error(f"Error boosting gap priority: {e}")
            
    def get_next_gap(self) -> Optional[Tuple[int, str, str]]:
        """
        Get the highest priority unfulfilled knowledge gap that hasn't been recently attempted.
        
        Returns:
            Tuple[int, str, str]: (gap_id, topic, description) or None if no gaps
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get gaps that haven't been attempted recently (avoid thrashing)
                cursor.execute('''
                    SELECT id, topic, description FROM knowledge_gaps
                    WHERE status = 'pending' 
                    AND (last_attempt_at IS NULL 
                         OR datetime(last_attempt_at) < datetime('now', '-1 hour'))
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                
                if result:
                    gap_id = result[0]
                    # Mark as attempted to avoid immediate retry
                    cursor.execute('''
                        UPDATE knowledge_gaps 
                        SET last_attempt_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (gap_id,))
                    conn.commit()
                    
                    logging.info(f"Selected knowledge gap for filling: ID {gap_id}, topic '{result[1]}'")
                
                return result
        except Exception as e:
            logging.error(f"Error getting next knowledge gap: {e}")
            return None
            
    def mark_fulfilled(self, gap_id: int, items_acquired: int = 0) -> bool:
        """
        Mark a knowledge gap as fulfilled and remove its embedding from the vector store.
        
        Args:
            gap_id: ID of the gap to mark fulfilled
            items_acquired: Number of knowledge items acquired for this gap
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get vector_id before updating status
                cursor.execute('SELECT vector_id FROM knowledge_gaps WHERE id = ?', (gap_id,))
                result = cursor.fetchone()
                vector_id = result[0] if result else None
                
                # Update status in SQL
                cursor.execute('''
                    UPDATE knowledge_gaps
                    SET status = 'fulfilled', 
                        fulfilled_at = CURRENT_TIMESTAMP,
                        items_acquired = ?
                    WHERE id = ?
                ''', (items_acquired, gap_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    # Remove embedding from Qdrant to allow similar future gaps
                    if vector_id:
                        self._remove_gap_embedding(vector_id)
                    
                    logging.info(f"✅ Marked knowledge gap {gap_id} as fulfilled "
                               f"with {items_acquired} items acquired")
                    return True
                else:
                    logging.warning(f"⚠️ Knowledge gap {gap_id} not found for marking fulfilled")
                    return False
                    
        except Exception as e:
            logging.error(f"❌ Error marking knowledge gap {gap_id} as fulfilled: {e}")
            return False

    def mark_failed(self, gap_id: int, reason: str = "") -> bool:
        """
        Mark a knowledge gap as failed (couldn't be filled).
        Note: Does NOT remove from vector store - keeps blocking similar gaps.
        
        Args:
            gap_id: ID of the gap to mark failed
            reason: Optional reason for failure
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE knowledge_gaps
                    SET status = 'failed',
                        last_attempt_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (gap_id,))
                
                conn.commit()
                if cursor.rowcount > 0:
                    logging.info(f"Marked knowledge gap {gap_id} as failed: {reason}")
                    return True
                else:
                    logging.warning(f"Knowledge gap {gap_id} not found for marking failed")
                    return False
        except Exception as e:
            logging.error(f"Error marking knowledge gap {gap_id} as failed: {e}")
            return False
    
    def get_gaps_by_status(self, status: str = 'pending', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get knowledge gaps by status.
        
        Args:
            status: Status to filter by ('pending', 'fulfilled', 'failed')
            limit: Maximum number of gaps to return
            
        Returns:
            List[Dict]: List of knowledge gaps with their details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, topic, description, priority, status, 
                           created_at, fulfilled_at, items_acquired, last_attempt_at, vector_id
                    FROM knowledge_gaps
                    WHERE status = ?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                ''', (status, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Error getting knowledge gaps by status '{status}': {e}")
            return []

    def get_gap_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge gaps."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count by status
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM knowledge_gaps
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                stats['by_status'] = status_counts
                
                # Total gaps
                stats['total'] = sum(status_counts.values())
                
                # Success rate
                fulfilled = status_counts.get('fulfilled', 0)
                total_attempted = fulfilled + status_counts.get('failed', 0)
                if total_attempted > 0:
                    stats['success_rate'] = fulfilled / total_attempted
                else:
                    stats['success_rate'] = 0
                
                # Recent activity (last 7 days)
                cursor.execute('''
                    SELECT COUNT(*) FROM knowledge_gaps
                    WHERE created_at > datetime('now', '-7 days')
                ''')
                stats['recent_additions'] = cursor.fetchone()[0]
                
                # Items acquired
                cursor.execute('''
                    SELECT SUM(items_acquired) FROM knowledge_gaps
                    WHERE status = 'fulfilled'
                ''')
                result = cursor.fetchone()[0]
                stats['total_items_acquired'] = result if result else 0
                
                # Vector store stats
                try:
                    if self._collection_initialized or self._ensure_gaps_collection_exists():
                        collection_info = self.qdrant_client.get_collection(self.gaps_collection_name)
                        stats['vector_store_count'] = collection_info.points_count
                    else:
                        stats['vector_store_count'] = 'N/A'
                except:
                    stats['vector_store_count'] = 'N/A'
                
                return stats
        except Exception as e:
            logging.error(f"Error getting gap statistics: {e}")
            return {}

    def cleanup_old_gaps(self, days_old: int = 30) -> int:
        """
        Clean up old fulfilled gaps to prevent database bloat.
        Also removes any orphaned embeddings from the vector store.
        
        Args:
            days_old: Remove fulfilled gaps older than this many days
            
        Returns:
            int: Number of gaps removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First, get vector_ids of gaps to be deleted
                cursor.execute('''
                    SELECT vector_id FROM knowledge_gaps
                    WHERE status = 'fulfilled' 
                    AND fulfilled_at < datetime('now', '-{} days')
                    AND vector_id IS NOT NULL
                '''.format(days_old))
                
                vector_ids = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Delete from SQL
                cursor.execute('''
                    DELETE FROM knowledge_gaps
                    WHERE status = 'fulfilled' 
                    AND fulfilled_at < datetime('now', '-{} days')
                '''.format(days_old))
                
                removed = cursor.rowcount
                conn.commit()
                
                # Clean up any remaining vector embeddings
                for vector_id in vector_ids:
                    self._remove_gap_embedding(vector_id)
                
                if removed > 0:
                    logging.info(f"Cleaned up {removed} old fulfilled knowledge gaps "
                               f"and {len(vector_ids)} embeddings")
                
                return removed
        except Exception as e:
            logging.error(f"Error cleaning up old gaps: {e}")
            return 0

    def sync_vector_store(self) -> Dict[str, int]:
        """
        Synchronize the vector store with the SQL database.
        Removes orphaned embeddings and adds missing ones for pending gaps.
        
        Returns:
            Dict with counts of actions taken
        """
        try:
            results = {'removed': 0, 'added': 0, 'errors': 0}
            
            if not self._ensure_gaps_collection_exists():
                logging.warning("Could not ensure gaps collection for sync")
                return results
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all pending gaps
                cursor.execute('''
                    SELECT id, topic, description, vector_id 
                    FROM knowledge_gaps 
                    WHERE status = 'pending'
                ''')
                pending_gaps = cursor.fetchall()
                
                for gap_id, topic, description, vector_id in pending_gaps:
                    if not vector_id:
                        # Missing embedding - create it
                        new_vector_id = self._store_gap_embedding(gap_id, topic, description)
                        if new_vector_id:
                            cursor.execute('''
                                UPDATE knowledge_gaps SET vector_id = ? WHERE id = ?
                            ''', (new_vector_id, gap_id))
                            results['added'] += 1
                        else:
                            results['errors'] += 1
                
                conn.commit()
                
            logging.info(f"Vector store sync complete: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error syncing vector store: {e}")
            return {'removed': 0, 'added': 0, 'errors': 1}