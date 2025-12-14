"""Vector database operations using Qdrant with enhanced search capabilities and verification."""

import time
import threading
import socket
import os
import logging
import uuid
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from utils import RateLimiter  # Import from the shared utilities module
from qdrant_client.http.exceptions import UnexpectedResponse
from config import (
    QDRANT_LOCAL_PATH, 
    QDRANT_COLLECTION_NAME, 
    QDRANT_USE_LOCAL,
    QDRANT_URL,
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL
)

class VectorDB:
    """Handles vector storage and similarity search using Qdrant."""
    
   
    def __init__(self):
        """Initialize the Vector DB with Qdrant."""
        logging.info("Initializing VectorDB with Qdrant")
        try:
            
            EMBEDDING_MODEL = "qwen3-embedding:4b"
            
            logging.info(f"Using {EMBEDDING_MODEL} for embeddings and {OLLAMA_MODEL} for chat")
            
            # Initialize Ollama embeddings with a dedicated embedding model
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            
            # Initialize threshold values for different search modes
            # Calibrated for qwen3-embedding:4b based on diagnostic showing noise floor ~0.43-0.52
            self.similarity_threshold = 0.55      # Default: above random noise floor
            self.comprehensive_threshold = 0.50   # Comprehensive: catches weak matches but excludes noise
            self.selective_threshold = 0.62       # Selective: moderately relevant and above
            self.verification_threshold = 0.55    # Verification: above noise floor
            # Other default settings
            self.default_k = 10       # Default number of results to return
            self.max_k = 100         # Maximum number of results to ever return
            self.testing = False     # Flag for test mode
            
            # Initialize the Qdrant client before using it
            if QDRANT_USE_LOCAL:
                logging.info(f"Using local Qdrant at: {QDRANT_LOCAL_PATH}")
                # Only create the directory if we're actually using local storage
                os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
                self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
            else:
                logging.info(f"Using remote Qdrant server at: {QDRANT_URL}")
                self.client = QdrantClient(url=QDRANT_URL)
            
            # Now initialize the store
            self.vector_store = self._initialize_store()
        except Exception as e:
            logging.error(f"VectorDB initialization error: {e}")
            raise

    def diagnose_embedding_issue(self):
        """
        Run a diagnostic check on embeddings and collection configuration.
    
        Returns:
            dict: Diagnostic information about embeddings and collection
        """
        try:
            # Get a sample embedding
            sample_text = "Test embedding for diagnostics"
            embedding = self.embeddings.embed_documents([sample_text])
        
            # Get collection info
            collection_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
        
            return {
                "sample_embedding_dimension": len(embedding[0]),
                "collection_dimension": collection_info.config.params.vectors.size,
                "dimension_match": len(embedding[0]) == collection_info.config.params.vectors.size,
                "model_name": "qwen3-embedding:4b",
                "collection_name": QDRANT_COLLECTION_NAME
            }
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _initialize_qdrant_client(self):
        """Initialize the Qdrant client with better concurrency handling and increased timeout."""
        try:
            if self.testing:
                # Use a temporary directory for testing to avoid conflicts
                temp_dir = tempfile.mkdtemp()
                logging.info(f"Using temporary Qdrant storage for testing at: {temp_dir}")
                return QdrantClient(path=temp_dir)
        
            if QDRANT_USE_LOCAL:
                # Ensure directory exists
                os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
                logging.info(f"Using local Qdrant storage at: {QDRANT_LOCAL_PATH}")
            
                # Try to use a Qdrant server first, especially on Windows
                try:
                    # Increase timeout significantly to prevent timeout issues
                    client = QdrantClient(url="http://localhost:6333", timeout=120.0)
                    # Test connection
                    client.get_collections()
                    logging.info("Connected to local Qdrant server")
                    return client
                except Exception as server_error:
                    logging.warning(f"No local Qdrant server available: {server_error}")
                    logging.info("Falling back to file-based Qdrant storage")
                
                    # For scheduled tasks, we should wait and retry rather than failing immediately
                    # Check if this is running in a scheduler thread
                    current_thread = threading.current_thread()
                    if "Scheduler" in current_thread.name:
                        logging.info(f"Thread {current_thread.name} is waiting for Qdrant access...")
                    
                        # Try multiple times with increasing delays
                        for attempt in range(1, 6):
                            try:
                                time.sleep(attempt * 5)  # 5, 10, 15, 20, 25 seconds
                                return QdrantClient(path=QDRANT_LOCAL_PATH)
                            except Exception as e:
                                if "already accessed" in str(e) and attempt < 5:
                                    logging.warning(f"Qdrant still busy on attempt {attempt}, retrying...")
                                    continue
                                else:
                                    raise
                
                    # Use file-based storage for interactive sessions
                    # Add parameters to make it more resilient
                    return QdrantClient(path=QDRANT_LOCAL_PATH, timeout=60.0)
            else:
                logging.info(f"Using remote Qdrant server at: {QDRANT_URL}")
                # Increase timeout for remote connections
                return QdrantClient(url=QDRANT_URL, timeout=60.0)
        except Exception as e:
            logging.error(f"Error initializing Qdrant client: {e}")
            raise

    def _initialize_store(self):
        """Initialize or load the Qdrant vector store with improved error handling."""
        try:
            # First get sample embedding to determine actual dimension
            retry_count = 3
            sample_embedding = None
        
            # Retry embedding generation with exponential backoff
            for attempt in range(retry_count):
                try:
                    sample_embedding = self.embeddings.embed_documents(["INITIALIZATION"])
                    break
                except Exception as e:
                    if attempt < retry_count - 1:
                        backoff = (2 ** attempt) * 2  # 2, 4, 8 seconds
                        logging.warning(f"Embedding generation failed (attempt {attempt+1}/{retry_count}): {e}")
                        logging.info(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                    else:
                        logging.error(f"Failed to generate embeddings after {retry_count} attempts: {e}")
                        raise
        
            if not sample_embedding:
                raise ValueError("Failed to generate sample embedding")
            
            actual_dimension = len(sample_embedding[0])

            # Update the instance attribute to match actual dimension
            self.embedding_dimension = actual_dimension
            logging.info(f"Detected embedding dimension: {actual_dimension}")

            # Collection management with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Check if collection exists
                    try:
                        collection_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
                        existing_dimension = collection_info.config.params.vectors.size
                
                        # If dimensions don't match, recreate collection with warning
                        if existing_dimension != actual_dimension:
                            logging.warning(f"Embedding dimension mismatch: expected {existing_dimension}, got {actual_dimension}")
                            logging.warning(f"Recreating collection {QDRANT_COLLECTION_NAME} with correct dimensions")
                    
                            # Delete existing collection
                            self.client.delete_collection(QDRANT_COLLECTION_NAME)
                    
                            # Create new collection with correct dimensions
                            self.client.create_collection(
                                collection_name=QDRANT_COLLECTION_NAME,
                                vectors_config=qdrant_models.VectorParams(
                                    size=actual_dimension,
                                    distance=qdrant_models.Distance.COSINE
                                )
                            )
                        else:
                            logging.info(f"Found existing Qdrant collection: {QDRANT_COLLECTION_NAME} with correct dimension {existing_dimension}")
                    
                        # Successfully validated/created collection, break the retry loop
                        break
                    
                    except (UnexpectedResponse, Exception) as e:
                        # Collection doesn't exist, create it
                        logging.info(f"Creating new Qdrant collection: {QDRANT_COLLECTION_NAME} with dimension {actual_dimension}")
                        self.client.create_collection(
                            collection_name=QDRANT_COLLECTION_NAME,
                            vectors_config=qdrant_models.VectorParams(
                                size=actual_dimension,  # Use detected dimension
                                distance=qdrant_models.Distance.COSINE
                            )
                        )
                
                        # Add initialization point to ensure collection is properly set up
                        point_id = str(uuid.uuid4())
                        self.client.upsert(
                            collection_name=QDRANT_COLLECTION_NAME,
                            points=[
                                qdrant_models.PointStruct(
                                    id=point_id,
                                    payload={"text": "INITIALIZATION", "source": "system"},
                                    vector=sample_embedding[0]
                                )
                            ]
                        )
                        logging.info("Added initialization point to Qdrant collection")
                    
                        # Successfully created collection, break the retry loop
                        break
                    
                except Exception as collection_error:
                    if attempt < max_retries - 1:
                        backoff = (2 ** attempt) * 3  # 3, 6, 12 seconds
                        logging.warning(f"Collection operation failed (attempt {attempt+1}/{max_retries}): {collection_error}")
                        logging.info(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                    else:
                        logging.error(f"Failed to manage collection after {max_retries} attempts: {collection_error}")
                        raise

            # Use Qdrant class
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=QDRANT_COLLECTION_NAME,
                embedding=self.embeddings  # Changed from 'embeddings' to 'embedding'
            )
            
            # Return the initialized vector store
            return vector_store

        except Exception as e:
            logging.error(f"Error initializing Qdrant store: {e}", exc_info=True)
            raise

    def batch_add_texts(self, texts: List[str], metadatas: List[dict] = None, 
                       batch_size: int = 10) -> bool:
        """
        Add multiple texts in batches to avoid overwhelming the database.
    
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries
            batch_size: Number of texts to add in each batch
        
        Returns:
            bool: True if all texts were successfully added
        """
        if not texts:
            return True
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        rate_limiter = RateLimiter(operations_per_second=2)  # Limit to 2 batches per second
    
        success_count = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
        
            # Generate IDs for this batch
            batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_texts))]
        
            try:
                # Add batch to vector store
                self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                success_count += len(batch_texts)
            
                # Respect rate limits
                rate_limiter.wait_if_needed()
            
                # Log progress for long operations
                if len(texts) > batch_size * 2:
                    logging.info(f"Batch processing progress: {min(i+batch_size, len(texts))}/{len(texts)}")
                
            except Exception as e:
                logging.error(f"Error in batch {i//batch_size + 1}: {e}")
            
        # Return True only if all texts were added successfully
        return success_count == len(texts)

    def add_text(self, text: str, metadata: dict = None, memory_id: str = None,
                retry_count: int = 2, memory_db_rollback: callable = None,
                duplicate_threshold: float = 0.98) -> tuple[bool, str]:
        """
        Add text to the vector store with robust error handling and transaction coordination.
        
        Args:
            text: The text content to store
            metadata: Optional metadata dictionary
            memory_id: Optional unique identifier for tracking
            retry_count: Number of retry attempts on failure
            memory_db_rollback: Optional callback function for rolling back SQL on failure
            duplicate_threshold: Similarity threshold for duplicate detection (default 0.98)
                                Higher values = stricter matching (fewer false duplicates)
                                Use 0.995+ for conversation summaries to allow similar but different content
        
        Returns:
            tuple[bool, str]: (success, reason)
                - (True, "stored") - Successfully stored
                - (False, "duplicate") - Rejected due to duplicate (not an error)
                - (False, "error") - Failed due to actual error
        """
        
        if not text or not text.strip():
            logging.warning("Attempted to add empty text")
            return False, "error"

        # Initialize counter attributes if they don't exist yet
        self.verify_count = getattr(self, 'verify_count', 0) + 1
        cleaned_text = text.strip()
        
        # Log the threshold being used for debugging
        logging.debug(f"DUPLICATE_CHECK: Using threshold {duplicate_threshold} for duplicate detection")
        
        # ================================================================
        # DUPLICATE CHECK - Skip for conversation_summary type
        # ================================================================
        # Conversation summaries should ALWAYS be stored because:
        # 1. Each represents a unique temporal snapshot of a conversation
        # 2. They naturally share structural similarities ("I had a conversation with Ken...")
        # 3. The similarity boost for search interferes with duplicate detection
        # 4. Even "similar" summaries capture different conversation content
        # 5. Duplicate detection was causing legitimate summaries to be rejected
        # ================================================================
        
        # Determine memory type from metadata
        memory_type = metadata.get('type', '') if metadata else ''
        
        # Types that should skip duplicate checking entirely
        skip_duplicate_types = ['conversation_summary']
        skip_duplicate_check = memory_type in skip_duplicate_types
        
        if skip_duplicate_check:
            # Log that we're skipping duplicate check for this type
            logging.info(f"DUPLICATE_CHECK: Skipping for type '{memory_type}' - always store temporal snapshots")
        else:
            # Run duplicate check for all other memory types
            try:
                results = self.search(
                    query=cleaned_text,
                    k=5,
                    mode="selective",
                    skip_boost=True  # Get raw scores for accurate duplicate detection
                )
                
                # Check for exact matches or very close duplicates
                for result in results:
                    result_content = result.get('content', '')
                    similarity = result.get('similarity_score', 0)
                    
                    # Exact match - always reject regardless of threshold
                    if result_content == cleaned_text:
                        logging.info(f"Exact duplicate found in vector DB, skipping: {cleaned_text[:50]}...")
                        return False, "duplicate"
                    
                    # Use configurable threshold for near-duplicate detection
                    if similarity > duplicate_threshold and len(result_content) > 10:
                        logging.info(f"Near-duplicate found in vector DB (similarity: {similarity:.3f} > threshold: {duplicate_threshold}), skipping: {cleaned_text[:50]}...")
                        return False, "duplicate"
                        
            except Exception as search_error:
                logging.warning(f"Error checking for duplicates in vector DB: {search_error}")
                # Continue with storage if duplicate check fails - don't block storage
        

        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {"source": str(metadata)}

        # Make sure we don't overwrite LangChain's content field
        metadata_copy = metadata.copy()
        
        # Remove any fields that might conflict with content storage
        metadata_copy.pop('page_content', None)  
        metadata_copy.pop('text', None)
        metadata_copy.pop('content', None)
        
        # Add your tracking fields
        if memory_id is not None:
            metadata_copy["memory_id"] = memory_id
            metadata_copy["tracking_id"] = memory_id  # Add both for compatibility

        # Make sure necessary fields exist
        if "source" not in metadata:
            metadata["source"] = "unknown"

        if "type" not in metadata:
            metadata["type"] = "general"

        # Generate a unique ID for this text
        text_id = str(uuid.uuid4())

        # Add retry loop with exponential backoff
        for attempt in range(retry_count):
            try:
                # Add to vector store using Langchain's Qdrant wrapper
                self.vector_store.add_texts(
                    texts=[cleaned_text],
                    metadatas=[metadata_copy],  # Use the cleaned metadata
                    ids=[text_id]
                )
        
                logging.info(f"[Attempt {attempt+1}] Text added to Qdrant with ID {text_id}")
        
                # Simple verification on first success (no need to retry verification)
                if self.verify_count % 20 == 0:
                    try:
                        verification_results = self.search(
                            query=cleaned_text,
                            k=2,
                            mode="default"
                        )
                
                        if verification_results and len(verification_results) > 0:
                            if verification_results[0]['similarity_score'] >= self.verification_threshold:
                                logging.info(f"Verification successful (score: {verification_results[0]['similarity_score']:.2f})")
                            else:
                                logging.warning(f"Verification found match but below threshold: {verification_results[0]['similarity_score']:.2f}")
                        else:
                            logging.warning("Verification found no matches")
                    except Exception as verify_error:
                        logging.warning(f"Verification error (non-critical): {verify_error}")
                
                return True, "stored"
        
            except Exception as e:
                # Check if this is a retryable error
                retryable = "already accessed" in str(e) or "connection" in str(e).lower() or "timed out" in str(e).lower()
        
                if retryable and attempt < retry_count - 1:
                    # Calculate backoff time (1s, 2s, 4s...)
                    backoff = 1.0 * (2 ** attempt)
                    logging.warning(f"Qdrant error on attempt {attempt+1}: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logging.error(f"Error adding text to Qdrant after {attempt+1} attempts: {e}")
            
                    # Call the rollback function if provided
                    if memory_db_rollback is not None:
                        try:
                            memory_db_rollback()
                            logging.info("Successfully rolled back MemoryDB entry")
                        except Exception as rollback_error:
                            logging.error(f"Error rolling back MemoryDB: {rollback_error}")
                    
                    return False, "error"

        return False, "error"
    
    def delete_text(self, text: str) -> bool:
        """
        Delete a specific text entry from the Qdrant vector store.

        Args:
            text (str): The text to delete.

        Returns:
            bool: True if the text was successfully deleted, False otherwise.
        """
        # Guard against None or empty text
        if text is None or not text.strip():
            logging.warning("Attempted to delete None or empty text")
            return False
            
        try:
            # Search for the text with a more lenient similarity threshold
            results = self.search(query=text, k=5, mode="selective")
        
            # No results found
            if not results:
                logging.warning(f"Text '{text[:50]}...' not found in vector store.")
                return False

            # Get the documents and their IDs
            docs_and_scores = self.vector_store.similarity_search_with_score(
                text, k=5, score_threshold=self.similarity_threshold
            )
            
            if not docs_and_scores:
                logging.warning(f"Text '{text[:50]}...' not found in vector store.")
                return False
            
            # Get the IDs to delete
            deleted = False
            for doc, score in docs_and_scores:
                # Qdrant stores the document ID in the metadata
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    doc_id = doc.metadata['id']
                    # Delete the document using the Qdrant client directly
                    self.client.delete(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points_selector=qdrant_models.PointIdsList(
                            points=[doc_id]
                        )
                    )
                    deleted = True
                    logging.info(f"Deleted text with ID {doc_id}")
            
            return deleted
        
        except Exception as e:
            logging.error(f"Error deleting text from Qdrant: {e}")
            return False
        
    def delete_by_id(self, vector_id):
        """Delete a vector by its ID from the vector database.
        
        Args:
            vector_id (str): The ID of the vector to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Delete the vector from Qdrant
            self.client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=qdrant_models.PointIdsList(  # Fixed: use qdrant_models instead of models
                    points=[vector_id]
                )
            )
            logging.info(f"Vector with ID {vector_id} deleted successfully")
            return True
        except Exception as e:
            logging.error(f"Error deleting vector with ID {vector_id}: {e}", exc_info=True)
            return False
    
    def search(self, query: str = None, k: int = None, mode: str = "default", 
                metadata_filters: Dict[str, Any] = None, skip_boost: bool = False) -> List[Dict[str, Union[str, float, dict]]]:
        """
        Enhanced search with metadata filtering capabilities and robust retry logic.

        Args:
            query (str, optional): Text query for semantic search (can be None for pure metadata search)
            k (int, optional): Number of results to return
            mode (str): Search mode ("default", "comprehensive", or "selective")
            metadata_filters (Dict[str, Any], optional): Dictionary of metadata filters
                - 'type': Filter by memory type (e.g., 'self', 'general', etc.)
                - 'tags': Filter by tags (string or list of strings)
                - 'min_confidence': Minimum confidence value (float between 0-1)
                - 'max_age_days': Maximum age in days (int)
                - 'source': Filter by source (string)
            skip_boost (bool): If True, skip the conversation_summary score boost.
                            Use this for duplicate detection to get raw similarity scores.

        Returns:
            List[Dict]: Search results with metadata and similarity scores
        """
        # Validate inputs with consistent guards
        if query is None and (metadata_filters is None or not metadata_filters):
            logging.warning("Both query and metadata_filters are empty or None, cannot perform search")
            return []

        if query is not None and not isinstance(query, str):
            logging.warning(f"Invalid query type: {type(query)}, expected string")
            query = str(query)

        # Normalize parameters    
        k = max(1, k or self.default_k)
        mode = mode.lower() if mode else "default"

        # Validate mode
        if mode not in ["default", "comprehensive", "selective"]:
            logging.warning(f"Unknown search mode: {mode}, defaulting to 'default'")
            mode = "default"

        try:
            # Set threshold based on mode
            if mode == "comprehensive":
                threshold = self.comprehensive_threshold
                k = min(k, self.max_k)  # Respect max_k while honoring user request
            elif mode == "selective":
                threshold = self.selective_threshold
            else:  # default
                threshold = self.similarity_threshold

            # Log the search attempt with metadata filters
            filter_str = str(metadata_filters) if metadata_filters else "None"
            query_str = query[:50] + "..." if query and len(query) > 50 else query
            logging.info(f"Executing search: query='{query_str}', mode={mode}, k={k}, metadata_filters={filter_str}")

            # Convert metadata filters to Qdrant filter format if provided
            qdrant_filter = None
            if metadata_filters and isinstance(metadata_filters, dict):
                # ✅ NORMALIZE: Handle both flat and nested metadata formats
                # Case 1: Nested format like {'metadata': {'type': 'X', 'date': 'Y'}}
                if 'metadata' in metadata_filters and isinstance(metadata_filters['metadata'], dict):
                    # Extract the nested metadata dict
                    normalized_filters = metadata_filters['metadata']
                    logging.debug(f"Unpacked nested metadata format: {normalized_filters}")
                else:
                    # Case 2: Flat format - strip "metadata." prefix from keys if present
                    normalized_filters = {}
                    for key, value in metadata_filters.items():
                        clean_key = key.replace('metadata.', '') if key.startswith('metadata.') else key
                        normalized_filters[clean_key] = value
                    logging.debug(f"Normalized flat metadata format: {normalized_filters}")
                
                # ✅ CRITICAL FIX: Convert tags string to array if needed
                if 'tags' in normalized_filters and isinstance(normalized_filters['tags'], str):
                    # Split comma-separated string into array
                    tags_str = normalized_filters['tags']
                    normalized_filters['tags'] = [tag.strip() for tag in tags_str.split(',')]
                    logging.debug(f"Converted tags from string to array: {normalized_filters['tags']}")
                
                # Use normalized filters for processing
                metadata_filters = normalized_filters
                
                filter_conditions = []
        
                # Process type filter
                if 'type' in metadata_filters and metadata_filters['type']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value=metadata_filters['type'])
                        )
                    )
        
                # Process tags filter (can be a single tag or list of tags)
                if 'tags' in metadata_filters and metadata_filters['tags']:
                    tags = metadata_filters['tags']
                    if isinstance(tags, str):
                        # Single tag - must contain this tag
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.tags",
                                match=qdrant_models.MatchText(text=tags)
                            )
                        )
                    elif isinstance(tags, list):
                        # List of tags - must contain ANY of these tags (OR condition)
                        tag_conditions = [
                            qdrant_models.FieldCondition(
                                key="metadata.tags",
                                match=qdrant_models.MatchText(text=tag)
                            ) for tag in tags if tag  # Skip empty tags
                        ]
                        if tag_conditions:  # Only add if we have valid conditions
                            filter_conditions.append(
                                qdrant_models.Filter(
                                    should=tag_conditions
                                )
                            )
        
                # Process min_confidence filter with safe conversion
                if 'min_confidence' in metadata_filters:
                    try:
                        min_confidence = float(metadata_filters['min_confidence'])
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.confidence",
                                range=qdrant_models.Range(
                                    gte=min_confidence
                                )
                            )
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Invalid min_confidence value: {metadata_filters['min_confidence']}, error: {e}")
        
                # Process max_age_days filter with safe conversion
                if 'max_age_days' in metadata_filters:
                    try:
                        # Calculate the date that's max_age_days old
                        from datetime import datetime, timedelta
                        max_age = int(metadata_filters['max_age_days'])
                        cutoff_date = (datetime.now() - timedelta(days=max_age)).isoformat()
            
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.created_at",
                                range=qdrant_models.Range(
                                    gte=cutoff_date
                                )
                            )
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Invalid max_age_days value: {metadata_filters['max_age_days']}, error: {e}")
            
                # Process source filter
                if 'source' in metadata_filters and metadata_filters['source']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.source",
                            match=qdrant_models.MatchValue(value=metadata_filters['source'])
                        )
                    )
            
                # Process date filter (for conversation summaries, reminders, etc.)
                if 'date' in metadata_filters and metadata_filters['date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.date",
                            match=qdrant_models.MatchValue(value=metadata_filters['date'])
                        )
                    )

                # Also check for legacy field names for backward compatibility
                if 'summary_date' in metadata_filters and metadata_filters['summary_date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.summary_date",
                            match=qdrant_models.MatchValue(value=metadata_filters['summary_date'])
                        )
                    )
                    
                # Process due_date filter (for reminders)
                if 'due_date' in metadata_filters and metadata_filters['due_date']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.due_date",
                            match=qdrant_models.MatchValue(value=metadata_filters['due_date'])
                        )
                    )

                # Process memory_id filter (for tracking_id based searches)
                if 'memory_id' in metadata_filters and metadata_filters['memory_id']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.memory_id",
                            match=qdrant_models.MatchValue(value=metadata_filters['memory_id'])
                        )
                    )
        
                # Combine all conditions with AND only if we have valid conditions
                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(
                        must=filter_conditions
                    )
                else:
                    logging.info("No valid metadata filters were created")

            # Enhanced retry logic for search operations
            docs_and_scores = []
            max_retries = 3
        
            for attempt in range(max_retries):
                try:
                    if query:
                        # If we have both query and filters
                        docs_and_scores = self.vector_store.similarity_search_with_score(
                            query, k=min(k, self.max_k), 
                            score_threshold=threshold,
                            filter=qdrant_filter
                        )
                        # If successful, break out of retry loop
                        logging.info(f"Vector search successful on attempt {attempt+1}")
                        break
                    
                    elif qdrant_filter:
                        # If we only have metadata filters (no text query)
                        try:
                            docs = self.vector_store.similarity_search(
                                "", k=min(k, self.max_k),
                                filter=qdrant_filter
                            )
                            # For filter-only searches, assign perfect similarity
                            docs_and_scores = [(doc, 1.0) for doc in docs]
                            logging.info(f"Metadata-only search successful on attempt {attempt+1}")
                            break  # Exit retry loop on success
                        
                        except Exception as filter_error:
                            logging.error(f"Error in filter-only search: {filter_error}")
                            # Try with empty string query as fallback
                            try:
                                docs_and_scores = self.vector_store.similarity_search_with_score(
                                    "", k=min(k, self.max_k),
                                    filter=qdrant_filter
                                )
                                logging.info(f"Metadata search fallback successful on attempt {attempt+1}")
                                break  # Exit retry loop if fallback succeeds
                            except Exception as fallback_error:
                                # This will trigger the outer exception handler and retry
                                raise fallback_error
                
                except Exception as search_error:
                    # Check if this is the last attempt
                    if attempt < max_retries - 1:
                        # Log the error and retry
                        wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds exponential backoff
                        logging.warning(f"Vector search attempt {attempt+1}/{max_retries} failed: {str(search_error)}")
                        logging.info(f"Retrying search in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # All retries failed, log the error
                        logging.error(f"All {max_retries} vector search attempts failed: {str(search_error)}")
                        # Return empty results after all retries fail
                        return []

            # Add this validation check
            if docs_and_scores is None:
                logging.warning(f"Search returned None for query: {query}")
                return []

            # =========================================================================
            # FORMAT RESULTS WITH CONVERSATION SUMMARY BOOST
            # =========================================================================
            # This boost compensates for "embedding dilution" in long-form content.
            # Conversation summaries are typically 1000-5000+ characters, causing their
            # embeddings to average across many topics. Specific topic searches (like
            # "Tom Green") score lower against these averaged embeddings even when the
            # topic is discussed in the summary. This boost increases the likelihood
            # that relevant summaries appear in search results.
            #
            # TUNING GUIDE:
            #   - Increase CONVERSATION_SUMMARY_BOOST if summaries still rarely appear
            #   - Decrease if too many irrelevant summaries appear in results
            #   - Set skip_boost=True for duplicate detection to use raw scores
            # =========================================================================
            
            CONVERSATION_SUMMARY_BOOST = 0.10  # Reduced from 0.10 for Qwen3-embedding score distribution
            
            formatted_results = []
            boosted_count = 0  # Track how many summaries we boost for logging
            
            for doc, score in docs_and_scores:
                try:
                    # Use getattr with defaults for safety
                    page_content = getattr(doc, 'page_content', 'Unknown content') 
                    doc_metadata = getattr(doc, 'metadata', {}) or {}
            
                    # Ensure we have a valid similarity score
                    try:
                        score_value = float(score)
                    except (ValueError, TypeError):
                        score_value = 0.0
                    
                    # =========================================================
                    # CONVERSATION SUMMARY BOOST - Targeted fix for embedding dilution
                    # Only apply boost when skip_boost is False (normal searches)
                    # Skip boost for duplicate detection to get accurate raw scores
                    # =========================================================
                    original_score = score_value
                    memory_type = doc_metadata.get('type', '')
                    
                    if memory_type == 'conversation_summary' and not skip_boost:
                        # Apply boost to help conversation summaries surface in results
                        score_value = min(1.0, score_value + CONVERSATION_SUMMARY_BOOST)  # Cap at 1.0
                        boosted_count += 1
                        logging.info(f"📝 Boosted conversation_summary score: {original_score:.3f} -> {score_value:.3f}")
                    # =========================================================
            
                    formatted_results.append({
                        'content': page_content,
                        'similarity_score': score_value,
                        'metadata': doc_metadata,
                        'above_threshold': score_value >= threshold
                    })
                except Exception as format_error:
                    logging.error(f"Error formatting search result: {format_error}")

            # Sort by similarity (boosted scores will help summaries rank higher)
            formatted_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Log summary of boost activity
            if boosted_count > 0:
                logging.info(f"📝 Applied conversation_summary boost to {boosted_count} result(s)")

            logging.info(f"Search found {len(formatted_results)} results with mode={mode}, metadata_filters={filter_str}")
            return formatted_results

        except Exception as e:
            logging.error(f"Error in search with metadata filters: {e}", exc_info=True)
            return []

    def search_with_ids(self, query: str = None, k: int = None, mode: str = "default", 
                   metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Union[str, float, dict]]]:
        """
        Enhanced search that returns actual Qdrant point IDs for deletion operations.
        
        Args:
            query (str, optional): Text query for semantic search
            k (int, optional): Number of results to return
            mode (str): Search mode ("default", "comprehensive", or "selective")
            metadata_filters (Dict[str, Any], optional): Dictionary of metadata filters
        
        Returns:
            List[Dict]: Search results with metadata, similarity scores, and point IDs
        """
        # Validate inputs
        if query is None and (metadata_filters is None or not metadata_filters):
            logging.warning("Both query and metadata_filters are empty or None, cannot perform search")
            return []
        
        if query is not None and not isinstance(query, str):
            logging.warning(f"Invalid query type: {type(query)}, expected string")
            query = str(query)
        
        # Normalize parameters    
        k = max(1, k or self.default_k)
        mode = mode.lower() if mode else "default"
        
        # Set threshold based on mode
        if mode == "comprehensive":
            threshold = self.comprehensive_threshold
            k = min(k, self.max_k)
        elif mode == "selective":
            threshold = self.selective_threshold
        else:  # default
            threshold = self.similarity_threshold
        
        try:
            # Convert metadata filters to Qdrant filter format if provided
            qdrant_filter = None
            if metadata_filters and isinstance(metadata_filters, dict):
                filter_conditions = []
                
                # Process type filter
                if 'type' in metadata_filters and metadata_filters['type']:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value=metadata_filters['type'])
                        )
                    )
                
                # Process other filters... (same as in your existing search method)
                # [Include all the filter processing code from your existing search method]
                
                # Combine all conditions with AND only if we have valid conditions
                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(must=filter_conditions)
            
            # Use direct Qdrant client for search with IDs
            if query:
                # Generate embedding for the query
                query_embedding = self.embeddings.embed_query(query)
                
                # Search using Qdrant client directly
                search_result = self.client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_embedding,  # Changed from query_vector to query
                    query_filter=qdrant_filter,
                    limit=k,
                    score_threshold=threshold,
                    with_payload=True,
                    with_vectors=False
                ).points  # Note: query_points returns a QueryResponse object, need .points
                    
            else:
                # For filter-only searches, use scroll
                search_result, _ = self.client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Format results with point IDs
            formatted_results = []
            for point in search_result:
                try:
                    # Extract content from payload
                    payload = point.payload or {}
                    content = payload.get('page_content', payload.get('text', 'Unknown content'))
                    
                    # Get metadata (remove the 'page_content' key if it exists)
                    metadata = {k: v for k, v in payload.items() if k != 'page_content'}
                    
                    # Get similarity score
                    score = getattr(point, 'score', 1.0) if hasattr(point, 'score') else 1.0
                    
                    formatted_results.append({
                        'id': str(point.id),  # THIS IS THE KEY - the actual Qdrant point ID
                        'content': content,
                        'similarity_score': float(score),
                        'metadata': metadata,
                        'above_threshold': float(score) >= threshold
                    })
                    
                except Exception as format_error:
                    logging.error(f"Error formatting search result with ID: {format_error}")
            
            # Sort by similarity
            formatted_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            logging.info(f"Search with IDs found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in search_with_ids: {e}", exc_info=True)
            return []
      
    def search_with_pagination(self, query: str, page_size: int = 10, page: int = 1, threshold: float = None) -> Dict[str, Union[List, int]]:
        """
        Paginated search using Qdrant's native offset capability for efficiency.
        
        Args:
            query (str): Search query
            page_size (int): Results per page (default: 10)
            page (int): Page number (1-indexed)
            threshold (float): Optional similarity threshold override
        
        Returns:
            Dict: Paginated results with metadata
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                return {
                    'results': [],
                    'total_results': 0,
                    'current_page': page,
                    'total_pages': 0,
                    'page_size': page_size
                }
            
            # Ensure page is at least 1
            page = max(1, page)
            
            # Use provided threshold or default
            search_threshold = threshold if threshold is not None else self.similarity_threshold
            
            # Calculate offset for this page
            offset = (page - 1) * page_size
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Get total count first (to calculate total pages)
            # Use query_points - search() was removed in qdrant-client 1.12+
            initial_search = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_embedding,  # Changed from query_vector to query
                limit=self.max_k,  # Get max results to count total matches
                score_threshold=search_threshold
            ).points  # Returns QueryResponse, need .points for the list
            
            total_above_threshold = len([r for r in initial_search if r.score >= search_threshold])
            
            # Now get the specific page of results we need
            search_result = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_embedding,  # Changed from query_vector to query
                limit=page_size,  # Fixed: was 'k' (undefined)
                offset=offset,    # Added: for pagination
                score_threshold=search_threshold,  # Fixed: was 'threshold'
                with_payload=True,
                with_vectors=False
                # Removed: query_filter=qdrant_filter (was undefined)
            ).points  # Returns QueryResponse, need .points for the list
            
            # Format results
            formatted_results = []
            for point in search_result:
                try:
                    # Extract content from payload
                    payload = point.payload or {}
                    content = payload.get('page_content', payload.get('text', 'Unknown content'))
                    
                    # Get metadata (remove the 'page_content' key if it exists)
                    metadata = {k: v for k, v in payload.items() if k != 'page_content'}
                    
                    # Get similarity score
                    score = float(point.score)
                    
                    formatted_results.append({
                        'id': str(point.id),
                        'content': content,
                        'similarity_score': score,
                        'metadata': metadata,
                        'above_threshold': score >= search_threshold
                    })
                    
                except Exception as format_error:
                    logging.error(f"Error formatting search result: {format_error}")
            
            # Calculate total pages
            total_pages = (total_above_threshold + page_size - 1) // page_size if total_above_threshold > 0 else 0
            
            logging.info(f"Paginated search: page {page}/{total_pages}, returned {len(formatted_results)} results")
            
            return {
                'results': formatted_results,
                'total_results': total_above_threshold,
                'current_page': page,
                'total_pages': total_pages,
                'page_size': page_size
            }
        
        except Exception as e:
            logging.error(f"Error in paginated search: {e}", exc_info=True)
            return {
                'results': [],
                'total_results': 0,
                'current_page': page,
                'total_pages': 0,
                'page_size': page_size
            }
                
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Qdrant vector store.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Check if the collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections)
            
            if not collection_exists:
                return {
                    "status": "error",
                    "message": f"Collection {QDRANT_COLLECTION_NAME} does not exist",
                    "collection_count": len(collections)
                }
                
            # Get collection info
            collection_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
            
            # Get point count
            count_result = self.client.count(
                collection_name=QDRANT_COLLECTION_NAME,
                count_filter=None  # Count all points
            )
            
            return {
                "status": "healthy",
                "collection_name": QDRANT_COLLECTION_NAME,
                "vectors_count": count_result.count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "storage_type": "local" if QDRANT_USE_LOCAL else "remote"
            }
            
        except Exception as e:
            logging.error(f"Error checking Qdrant health: {e}")
            return {
                "status": "error",
                "message": str(e)
            }