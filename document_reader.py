# document_reader.py
"""Document reading and processing functionality."""

import os
import logging
import re
import json
import uuid
import datetime
from typing import List, Tuple, Dict, Any, Optional
import PyPDF2
import docx
from config import DOCS_PATH, SUPPORTED_EXTENSIONS, DEFAULT_CHUNK_SIZE

class DocumentReader:
    """Handles reading and processing different document types."""
    
    def __init__(self, docs_path: str = DOCS_PATH, chatbot=None):
        """
        Initialize the DocumentReader with a documents directory path and chatbot reference.
        
        Args:
            docs_path (str): Path to documents directory
            chatbot: Reference to the main chatbot instance for accessing LLM, memory functions, etc.
        """
        logging.info("Initializing DocumentReader")
        try:
            self.docs_path = os.path.abspath(docs_path)
            # Create LocalDocs directory if it doesn't exist
            os.makedirs(self.docs_path, exist_ok=True)
            self.supported_extensions = SUPPORTED_EXTENSIONS
            
            # Store reference to the chatbot instance
            self.chatbot = chatbot
            
            logging.info(f"Documents directory confirmed: {self.docs_path}")
        except Exception as e:
            logging.error(f"DocumentReader initialization error: {e}")
            raise

    # [Keep all your existing methods like find_actual_file, list_documents, read_file, etc.]
    
    def find_actual_file(self, partial_name: str) -> Optional[str]:
        """Find the actual filename from a partial or case-insensitive match."""
        try:
            # Guard against None or empty filename
            if not partial_name or not isinstance(partial_name, str):
                logging.warning(f"Invalid filename provided to find_actual_file: {partial_name}")
                return None
                
            available_files = os.listdir(self.docs_path)
            search_term = partial_name.lower().strip()
            
            # Try exact match first (case insensitive)
            for file in available_files:
                if file.lower() == search_term:
                    return file
                    
            # Try with common extensions if no extension provided
            if '.' not in search_term:
                for ext in self.supported_extensions:
                    test_name = search_term + ext
                    for file in available_files:
                        if file.lower() == test_name:
                            return file
            
            # Try matching just the name part (without extension)
            for file in available_files:
                name_only = os.path.splitext(file)[0].lower()
                if name_only == search_term:
                    return file
                    
            # Try more lenient matching
            for file in available_files:
                name_only = os.path.splitext(file)[0].lower()
                if search_term in name_only or name_only in search_term:
                    return file
            
            return None
            
        except Exception as e:
            logging.error(f"Error in file search: {e}")
            return None

    def list_documents(self) -> List[str]:
        """List all available documents in the LocalDocs directory."""
        try:
            return [file for file in os.listdir(self.docs_path) 
                   if file.lower().endswith(tuple(self.supported_extensions))]
        except Exception as e:
            logging.error(f"Error listing documents: {e}")
            return []

    def read_file(self, filename: str) -> Tuple[str, bool]:
        """Read and extract text from a file."""
        try:
            # Guard against None or empty filename
            if filename is None or not isinstance(filename, str) or not filename.strip():
                logging.error(f"Attempted to read a None or empty filename")
                return "", False
                
            actual_file = self.find_actual_file(filename)
            if not actual_file:
                logging.error(f"File not found: {filename}")
                return "", False
                
            file_path = os.path.join(self.docs_path, actual_file)
            file_ext = os.path.splitext(actual_file)[1].lower()
            
            if file_ext == '.pdf':
                content = self._read_pdf(file_path)
            elif file_ext == '.txt':
                content = self._read_txt(file_path)
            elif file_ext == '.docx':
                content = self._read_docx(file_path)
            else:
                logging.error(f"Unsupported file type: {file_ext}")
                return "", False
                
            if content:
                logging.info(f"Successfully read file: {actual_file}")
                return content, True
            return "", False
            
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")
            return "", False

    def _read_pdf(self, file_path: str) -> str:
        """Read PDF file and extract text."""
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error(f"Attempted to read PDF with None or empty file_path")
                return ""
                
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content.append(text)
                return "\n".join(content)
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {e}")
            return ""

    def _read_txt(self, file_path: str) -> str:
        """Read text file with encoding fallback support."""
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error(f"Attempted to read TXT with None or empty file_path")
                return ""
                
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logging.error(f"Error reading text file with latin-1 encoding: {e}")
                return ""
        except Exception as e:
            logging.error(f"Error reading text file: {e}")
            return ""

    def _read_docx(self, file_path: str) -> str:
        """Read DOCX file and extract text."""
        try:
            # Guard against None or empty file_path
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logging.error(f"Attempted to read DOCX with None or empty file_path")
                return ""
                
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        except Exception as e:
            logging.error(f"Error reading DOCX file {file_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Break text into manageable chunks for processing."""
        try:
            # Guard against None or empty text
            if text is None:
                logging.error(f"Error chunking text: 'NoneType' object has no attribute 'split'")
                return []
                
            if not isinstance(text, str) or not text.strip():
                logging.warning(f"Attempted to chunk empty text")
                return []
                
            sentences = text.split('.')
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + '.'
                sentence_size = len(sentence)
                
                if current_size + sentence_size > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
        except Exception as e:
            logging.error(f"Error chunking text: {e}")
            return [text] if text else []

    def process_uploaded_document(self, filename: str) -> str:
        """
        Process an uploaded document by generating and storing only a document summary.
        This method skips storing individual chunks/lines and focuses only on creating a
        comprehensive document summary.

        Args:
            filename (str): Name of the uploaded file in LocalDocs.

        Returns:
            str: Status message for the user, including document summary information.
        """
        try:
            # Guard against None or empty filename
            if filename is None or not isinstance(filename, str) or not filename.strip():
                return "Please provide a valid filename."

            # Ensure we have a chatbot reference
            if not self.chatbot:
                return "Error: Chatbot reference not available for document processing."

            # Find the actual file
            actual_filename = self.find_actual_file(filename)
            if not actual_filename:
                return f"Document not found: {filename}"

            # Generate a unique document processing ID for tracking this batch
            document_transaction_id = str(uuid.uuid4())
            logging.info(f"Starting document processing transaction {document_transaction_id} for {actual_filename}")

            # Check if the document has been processed before
            previously_processed = self.chatbot.memory_db.check_document_processed(actual_filename)
            if previously_processed:
                logging.info(f"Document {actual_filename} has been processed before, checking for existing summary")
                
                # Try to retrieve existing summary with metadata filters (consistent with deepseek.py)
                try:
                    vector_results = self.chatbot.vector_db.search(
                        query="",  # Empty query for metadata-only search
                        mode="selective",
                        metadata_filters={"metadata.type": "document_summary", "metadata.source": actual_filename}
                    )
                    
                    if vector_results and len(vector_results) > 0:
                        logging.info(f"Found existing summary for {actual_filename}")
                        return f"Document {actual_filename} has already been processed. You can retrieve its summary with:\n\n[RETRIEVE: | metadata.type=document_summary | metadata.source={actual_filename}]"
                except Exception as e:
                    logging.warning(f"Error checking for existing summary: {e}")

            # Read the document content
            content, success = self.read_file(actual_filename)
            if not success:
                return f"Failed to read {actual_filename}."

            if not content or not content.strip():
                return f"No content found in {actual_filename} to process."

            # Create tracking info for reporting
            extraction_info = {
                "document_name": actual_filename,
                "file_size_kb": os.path.getsize(os.path.join(self.docs_path, actual_filename)) / 1024,
                "previously_processed": previously_processed,
                "document_summary": "Not created",
                "summary_preview": ""
            }
                
            # Log the processing start with document details
            logging.info(f"Started document processing: {actual_filename}, Size: {extraction_info['file_size_kb']:.1f} KB")

            # Create a shorter version of content for logging
            content_preview = content[:1000] + ("..." if len(content) > 1000 else "")
            logging.info(f"Content preview for summary generation: {content_preview}")

            # Create a specific prompt for document summary generation
            summary_prompt = (
            f"As an autonomous AI system, you're analyzing document '{actual_filename}'.\n\n"
            f"TASK: Create a comprehensive summary of this document.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Create a summary (500 to 1000 words MAXIMUM)\n"
            f"2. Make the summary concise but informative, prioritizing the most important content\n"
            f"3. Begin with a clear statement of the document's purpose\n"
            f"4. Write in a clear, direct style optimized for retrieval\n"
            f"5. Do NOT include specific commands or implementation details in the summary\n\n"
            f"DOCUMENT CONTENT:\n{content}"
        )

            logging.info(f"Summary prompt created with length: {len(summary_prompt)}")

            # Generate the summary using chatbot's LLM
            try:
                logging.info("Invoking LLM to generate document summary")
                doc_summary = self.chatbot.llm.invoke(summary_prompt)
                
                # Log the summary generation result
                if doc_summary is None:
                    logging.error("LLM returned None for document summary generation")
                    extraction_info["document_summary"] = "Error: LLM returned None"
                    return f"Error: Failed to generate summary for {actual_filename}"
                else:
                    logging.info(f"Generated summary length: {len(doc_summary)}")
                    logging.info(f"Summary preview: {doc_summary[:200]}...")
                
                # Verify the summary is valid and meets quality standards
                if doc_summary and isinstance(doc_summary, str) and len(doc_summary.strip()) > 50:
                    # Format the summary with a clear prefix that identifies it as a document summary
                    formatted_summary = f"Document Summary - {actual_filename}: {doc_summary.strip()}"
                    
                    logging.info("Summary meets quality standards, proceeding with storage")
                    
                    # Simplified metadata - no nested structure
                    summary_metadata = {
                        "type": "document_summary",  # Standard type
                        "source": actual_filename,    # Document source/name
                        "tags": "summary,document",   # Helpful tags
                        "confidence": 0.5            # medium confidence for document summaries
                    }

                    # Store the document summary using chatbot's transaction coordination
                    success, summary_id = self.chatbot.store_memory_with_transaction(
                        content=formatted_summary,
                        memory_type="document_summary",  # Use consistent memory_type
                        metadata=summary_metadata,
                        confidence=0.85
                    )
                    
                    logging.info(f"Storing document summary with type=document_summary")
                    logging.info(f"Summary metadata: {summary_metadata}")
                    

                    if success:
                        # After storing the summary, verify it using a simple, direct query
                        verify_results = self.chatbot.vector_db.search(
                            query="",  # Empty query for metadata-only search
                            mode="selective",
                            metadata_filters={"type": "document_summary", "source": actual_filename},
                            k=1
                        )
                        if not verify_results:
                            logging.error(f"Document summary not found in vector database after storing: {actual_filename}")
                            # You could add cleanup code here if needed
                            
                            # Try another search approach as fallback verification
                            alt_results = self.chatbot.vector_db.search(
                                query=f"document summary {actual_filename}",
                                mode="comprehensive",
                                k=5
                            )
                            
                            if alt_results:
                                logging.info(f"Document summary found via text search but not metadata: {actual_filename}")
                                verification_status = "Found by text search but not metadata filter"
                            else:
                                logging.error(f"Document summary not found by any method: {actual_filename}")
                                verification_status = "Not found in vector database"
                        else:
                            logging.info(f"Document summary successfully verified in vector database: {actual_filename}")
                            verification_status = "Verified in vector database"
                            
                        # Include verification status in your report
                        verification_results = [f"✓ {verification_status}"]
                    
                    if success and summary_id:
                        logging.info(f"Successfully stored document summary with ID {summary_id}")
                        logging.info(f"DEBUG: Storing document summary with metadata: {json.dumps(summary_metadata)}")
                        extraction_info["document_summary"] = "Created and stored successfully"
                        extraction_info["summary_preview"] = doc_summary.strip()[:200] + "..."
                        
                        # TEMPORARY DISABLE - Testing context preservation
                        # # NEW CODE STARTS HERE ↓
                        # # Store the search command for sidebar display
                        search_command = f"[SEARCH: {actual_filename} | type=document_summary]"
                        # 
                        # # Add to session state for sidebar display
                        try:
                            import streamlit as st
                            if hasattr(st, 'session_state'):
                                st.session_state.recent_document_search = {
                                    'filename': actual_filename,
                                    'search_command': search_command,
                                    'processed_time': datetime.datetime.now().strftime("%H:%M:%S")
                                }
                                logging.info(f"Stored search command in session state: {search_command}")
                        except Exception as session_error:
                        #     logging.warning(f"Could not store search command in session state: {session_error}")
                        # # NEW CODE ENDS HERE ↑

                        # Verify that we can retrieve the summary
                            logging.info("Verifying summary storage with direct search")
                            verification_results = []
                        
                        try:
                            # Try to retrieve the summary with metadata filters (consistent with deepseek.py retrieval)
                            vector_results = self.chatbot.vector_db.search(
                                query="",  # Empty query for metadata-only search
                                mode="selective",
                                metadata_filters={"metadata.type": "document_summary", "metadata.source": actual_filename}
                            )
                            
                            if vector_results and len(vector_results) > 0:
                                verification_results.append("✓ Found in vector database by metadata filter")
                                logging.info(f"Vector DB verification success: Found summary by metadata")
                            else:
                                # Try alternate metadata format as fallback
                                vector_results = self.chatbot.vector_db.search(
                                    query="",
                                    mode="selective",
                                    metadata_filters={"type": "document_summary", "source": actual_filename}
                                )
                                
                                if vector_results and len(vector_results) > 0:
                                    verification_results.append("✓ Found in vector database by alternate metadata filter")
                                    logging.info(f"Vector DB verification success with alternate metadata format")
                                else:
                                    verification_results.append("✗ Not found in vector database by metadata filter")
                                    logging.warning(f"Vector DB verification failed: No results with metadata filter")
                                    
                                    # Try a text-based search as fallback
                                    text_results = self.chatbot.vector_db.search(
                                        query=f"document summary {actual_filename}",
                                        mode="comprehensive",
                                        k=5
                                    )
                                    
                                    if text_results and len(text_results) > 0:
                                        verification_results.append("✓ Found in vector database by text search")
                                        logging.info(f"Vector DB text search found {len(text_results)} results")
                                    else:
                                        verification_results.append("✗ Not found in vector database by text search")
                                        logging.warning(f"Vector DB text search failed to find summary")
                            
                            # Check memory DB for document_summary type
                            mem_results = self.chatbot.memory_db.get_memories_by_type("document_summary", limit=5)
                            if mem_results:
                                found_match = False
                                for mem in mem_results:
                                    if actual_filename in mem.get('content', '') or actual_filename in mem.get('source', ''):
                                        found_match = True
                                        break
                                        
                                if found_match:
                                    verification_results.append("✓ Found in memory database")
                                    logging.info(f"Memory DB verification success: Found summary")
                                else:
                                    verification_results.append("✗ Found summaries in memory DB but none match this document")
                                    logging.warning(f"Memory DB verification partial: Found summaries but none match")
                            else:
                                verification_results.append("✗ No document summaries found in memory database")
                                logging.warning(f"Memory DB verification failed: No summaries found")
                                
                            # Store verification results
                            extraction_info["verification"] = verification_results
                            logging.info(f"Verification results: {', '.join(verification_results)}")
                            
                        except Exception as verify_err:
                            logging.error(f"Error during summary verification: {verify_err}", exc_info=True)
                            extraction_info["verification"] = [f"Error during verification: {str(verify_err)}"]
                            
                        # Create a user-friendly response showing the retrieve command
                        formatted_report = (
                            f"# Document Summary: {actual_filename}\n\n"
                            f"**Status:** {extraction_info['document_summary']}\n\n"
                            f"**Summary:**\n{doc_summary.strip()}\n\n"
                            f"**To retrieve this summary, use this command:**\n"
                            f"```\n[SEARCH: {actual_filename} | type=document_summary]\n```\n\n"
                        )

                        if "verification" in extraction_info:
                            formatted_report += "**Verification:**\n"
                            for result in extraction_info["verification"]:
                                formatted_report += f"- {result}\n"
                                
                        return formatted_report
                    else:
                        logging.error(f"Failed to store document summary for {actual_filename}")
                        extraction_info["document_summary"] = "Failed to store"
                        return f"Error: Failed to store document summary for {actual_filename}"
                else:
                    logging.warning(f"Generated summary was too short or invalid for {actual_filename}")
                    if doc_summary:
                        logging.warning(f"Invalid summary content: {doc_summary[:200]}")
                    extraction_info["document_summary"] = "Generated summary was invalid or too short"
                    return f"Error: Generated summary was too short or invalid for {actual_filename}"
            except Exception as summary_error:
                logging.error(f"Error generating document summary: {summary_error}", exc_info=True)
                extraction_info["document_summary"] = f"Error: {str(summary_error)}"
                return f"Error generating document summary: {str(summary_error)}"
                
        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}", exc_info=True)
            return f"Error processing document: {str(e)}"
    
    def _format_document_extraction_report(self, extraction_info: Dict, filename: str) -> str:
        """Format an extraction report as rich HTML for better presentation in Streamlit."""
        try:
            # Format the report title with document name
            report = f"# Document Extraction Report: {filename}\n\n"
            
            # Add overview section
            report += "## Overview\n"
            report += f"- **Document:** {filename}\n"
            report += f"- **Total Chunks:** {extraction_info['total_chunks']}\n"
            report += f"- **Processed Chunks:** {extraction_info['processed_chunks']}\n"
            report += f"- **Stored Items:** {extraction_info['stored_items']}\n"
            report += f"- **Extraction Rate:** {extraction_info['extraction_rate']:.1f}%\n"
            report += f"- **Previously Processed:** {'Yes' if extraction_info['previously_processed'] else 'No'}\n"
            
            # Add document summary section if available
            if extraction_info['summary_preview']:
                report += "\n## Document Summary\n"
                report += f"Status: **{extraction_info['document_summary']}**\n\n"
                report += f"{extraction_info['summary_preview']}\n"
                
                # Add verification info if available
                if 'verification' in extraction_info:
                    report += "\n**Verification Tests:**\n"
                    for test in extraction_info['verification']:
                        report += f"- {test}\n"
            
            # Add extracted content section
            report += "\n## Extracted Content\n"
            
            # Important items first
            if extraction_info['important']:
                report += "\n### Important Items\n"
                for item in extraction_info['important']:
                    report += f"- **{item['content']}**"
                    report += f" (Confidence: {item['confidence']:.1f})\n"
            
            # General items
            if extraction_info['general']:
                report += "\n### General Items\n"
                for item in extraction_info['general']:
                    report += f"- {item['content']}"
                    report += f" (Confidence: {item['confidence']:.1f})\n"
            
            # Skipped chunks
            report += f"\n### Skipped Chunks: {extraction_info['skipped']}\n"
            
            # By confidence breakdown
            report += "\n## Extraction Stats\n"
            report += "**By Confidence Level:**\n"
            report += f"- High (0.8-1.0): {extraction_info['by_confidence']['high']}\n"
            report += f"- Medium (0.5-0.79): {extraction_info['by_confidence']['medium']}\n"
            report += f"- Low (0.1-0.49): {extraction_info['by_confidence']['low']}\n"
            
            return report
            
        except Exception as e:
            logging.error(f"Error formatting document extraction report: {e}")
            return f"Error creating extraction report: {str(e)}"

    def _store_extraction_report(self, extraction_info: Dict, filename: str) -> None:
        """Save document extraction report to a JSON file for future reference."""
        try:
            # Create report directory if it doesn't exist
            if self.chatbot and hasattr(self.chatbot, 'memory_db'):
                # Use memory_db path if available
                reports_dir = os.path.join(os.path.dirname(self.chatbot.memory_db.db_path), "extraction_reports")
            else:
                # Fall back to local path if chatbot or memory_db not available
                reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction_reports")
                
            os.makedirs(reports_dir, exist_ok=True)
        
            # Generate a safe filename
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(reports_dir, f"{safe_filename}_{timestamp}.json")
        
            # Add timestamp to the report
            report_data = {
                "document": filename,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "extraction_data": extraction_info
            }
        
            # Save the report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logging.info(f"Saved extraction report to {report_path}")
        
        except Exception as e:
            logging.error(f"Error saving extraction report: {e}")
            
    def test_document_summary_search(self, document_name: str) -> str:
        """
        Test the document summary search functionality.
        
        Args:
            document_name (str): Name of the document to search for summaries
            
        Returns:
            str: Test results
        """
        try:
            if not self.chatbot or not hasattr(self.chatbot, 'deepseek_enhancer'):
                return "Cannot test: Chatbot or DeepSeek enhancer not available"
                
            # Prepare the search query for document summaries
            query = f"document summary | source={document_name}"
            
            # Try to retrieve document summary
            retrieval_result, success = self.chatbot.deepseek_enhancer._handle_retrieve_command(query)
            
            if success and "NO DATA FOUND" not in retrieval_result:
                return f"✅ Successfully retrieved document summary for '{document_name}'.\n\nPreview: {retrieval_result[:200]}..."
            else:
                return f"❌ Failed to retrieve document summary for '{document_name}'.\nResult: {retrieval_result[:100]}..."
                
        except Exception as e:
            logging.error(f"Error testing document summary search: {e}")
            return f"Error testing document summary search: {str(e)}"