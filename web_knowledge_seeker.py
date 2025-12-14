# web_knowledge_seeker.py
"""Enhanced web knowledge seeker integration and AI-driven content selection."""
import datetime
import logging
import requests
import time
import re
import random  
import uuid
import json  # MISSING - needed for blacklist file operations
import os    # MISSING - needed for file existence checks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta 
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class WebKnowledgeSeeker:
    """Enhanced web knowledge seeker with multiple search engines and anti-blocking measures."""
    
    def __init__(self, memory_db, vector_db, chatbot=None):
        """Initialize the enhanced web knowledge seeker."""
        self.memory_db = memory_db
        self.vector_db = vector_db
        self.chatbot = chatbot
        self.session = requests.Session()
        
        # Enhanced configuration
        self.min_request_interval = 3  # Increased base delay
        self.max_request_interval = 7  # Maximum delay range
        self.last_request_time = 0
        self.min_content_length = 50  # Minimum content length filter
        
        # Domain blacklist management
        self.blacklist_file = 'failed_domains.json'
        self.blacklist_duration = 24 * 60 * 60  # 24 hours in seconds
        self.failed_domains = self._load_blacklist()
        
        # Searx instances (hardcoded reliable ones)
        self.searx_instances = []  # Empty - disabled
        self.current_searx_index = 0
        
        # UPDATED: Search engine priority order - removing failing engines
        self.search_engines = [
            'startpage',    # Primary - working well
            'wikipedia'     # Reliable backup
           
        ]
        
    logging.info("Enhanced Web Knowledge Seeker initialized with working engines only: StartPage, Wikipedia")
        

    def _get_headers_rotation(self):
        """Enhanced header rotation with realistic browser fingerprints."""
        headers_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            },
            {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Linux"'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Microsoft Edge";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            }
        ]
        
        # Add random referer sometimes to simulate natural browsing
        selected_headers = random.choice(headers_pool).copy()
        
        # 30% chance to add a referer
        if random.random() < 0.3:
            referers = [
                'https://www.google.com/',
                'https://www.bing.com/',
                'https://github.com/',
                'https://stackoverflow.com/'
            ]
            selected_headers['Referer'] = random.choice(referers)
            
        return selected_headers

    def _apply_enhanced_delays(self, last_request_time=None):
        """Apply enhanced delay strategy with randomization."""
        current_time = time.time()
        
        if last_request_time is None:
            last_request_time = self.last_request_time
            
        time_since_last = current_time - last_request_time
        
        # Calculate base delay with randomization
        base_delay = random.uniform(self.min_request_interval, self.max_request_interval)
        
        # Add jitter for more natural timing
        jitter = random.uniform(0.5, 2.5)
        total_delay = base_delay + jitter
        
        # Check if we need to wait
        if time_since_last < total_delay:
            sleep_time = total_delay - time_since_last
            logging.info(f"⏳ Applying enhanced delay: {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _load_blacklist(self):
        """Load domain blacklist from persistent file."""
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, 'r') as f:
                    blacklist_data = json.load(f)
                    
                # Clean expired entries and handle format compatibility
                current_time = datetime.now()
                cleaned_blacklist = {}
                
                for domain, entry in blacklist_data.items():
                    try:
                        # Handle both old (string) and new (dict) formats
                        if isinstance(entry, str):
                            # Old format - convert to new format or skip if expired
                            blacklist_time = datetime.fromisoformat(entry)
                            if current_time - blacklist_time < timedelta(seconds=self.blacklist_duration):
                                # Convert to new format with default values
                                expiry_time = blacklist_time + timedelta(seconds=self.blacklist_duration)
                                cleaned_blacklist[domain] = {
                                    'blacklisted_at': entry,
                                    'expires_at': expiry_time.isoformat(),
                                    'error_type': 'legacy',
                                    'duration_minutes': self.blacklist_duration // 60
                                }
                        else:
                            # New format - check if still valid
                            expires_at = datetime.fromisoformat(entry['expires_at'])
                            if current_time < expires_at:
                                cleaned_blacklist[domain] = entry
                                
                    except Exception as parse_error:
                        logging.warning(f"Skipping invalid blacklist entry for {domain}: {parse_error}")
                        continue
                
                # Save cleaned blacklist
                self._save_blacklist(cleaned_blacklist)
                
                logging.info(f"📋 Loaded {len(cleaned_blacklist)} blacklisted domains")
                return cleaned_blacklist
        except Exception as e:
            logging.error(f"Error loading blacklist: {e}")
            
        return {}
    
    def _blacklist_domain_graduated(self, domain, error_type='403'):
        """Implement graduated blacklisting based on error type."""
        try:
            parsed_domain = urlparse(domain).netloc if domain.startswith('http') else domain
            current_time = datetime.now()
            
            # Check if domain was previously blacklisted
            if parsed_domain in self.failed_domains:
                # Extend blacklist period for repeat offenders
                blacklist_duration = 4 * 60 * 60  # 4 hours for repeat failures
            else:
                # Different durations based on error type
                if error_type == '403':
                    blacklist_duration = 30 * 60  # 30 minutes for 403 (might be temporary)
                elif error_type == '429':
                    blacklist_duration = 2 * 60 * 60  # 2 hours for rate limiting
                elif error_type == 'timeout':
                    blacklist_duration = 15 * 60  # 15 minutes for timeouts
                else:
                    blacklist_duration = 60 * 60  # 1 hour for other errors
            
            # Store with expiration time instead of creation time
            expiry_time = current_time + timedelta(seconds=blacklist_duration)
            self.failed_domains[parsed_domain] = {
                'blacklisted_at': current_time.isoformat(),
                'expires_at': expiry_time.isoformat(),
                'error_type': error_type,
                'duration_minutes': blacklist_duration // 60
            }
            
            self._save_blacklist()
            
            logging.warning(f"🚫 Blacklisted {parsed_domain} for {blacklist_duration//60} minutes ({error_type} error)")
            
        except Exception as e:
            logging.error(f"Error in graduated blacklisting for {domain}: {e}")
    
    def _blacklist_domain(self, domain, error_type='403'):
        """Legacy method that calls the graduated blacklisting system."""
        return self._blacklist_domain_graduated(domain, error_type)

    def _is_domain_blacklisted(self, url):
        """Updated blacklist check supporting the new graduated system."""
        try:
            domain = urlparse(url).netloc
            
            if domain in self.failed_domains:
                blacklist_info = self.failed_domains[domain]
                
                # Handle both old format (string) and new format (dict)
                if isinstance(blacklist_info, str):
                    # Old format - use 24 hour default
                    blacklist_time = datetime.fromisoformat(blacklist_info)
                    if datetime.now() - blacklist_time < timedelta(hours=24):
                        return True
                else:
                    # New format with expiry time
                    expires_at = datetime.fromisoformat(blacklist_info['expires_at'])
                    if datetime.now() < expires_at:
                        return True
                
                # Remove expired blacklist entry
                del self.failed_domains[domain]
                self._save_blacklist()
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking blacklist for {url}: {e}")
            return False

    def _save_blacklist(self, blacklist_data=None):
        """Save domain blacklist to persistent file."""
        try:
            data_to_save = blacklist_data if blacklist_data is not None else self.failed_domains
            with open(self.blacklist_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving blacklist: {e}")

    # ========================================================================
    # NEW HELPER METHODS FOR CONTENT VALIDATION AND CLEANING
    # Added to fix web learning corruption issues
    # ========================================================================

    def _validate_content_type(self, response, url: str) -> tuple:
        """
        Validate that the response Content-Type is processable HTML/text.
        
        Args:
            response: The requests Response object
            url: The URL being fetched (for logging)
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Log the content type for debugging
            logging.debug(f"Content-Type for {url}: {content_type}")
            
            # Define valid content types for text extraction
            valid_types = [
                'text/html',
                'text/plain',
                'application/xhtml+xml',
                'application/xml',
                'text/xml'
            ]
            
            # Define explicitly invalid content types
            invalid_types = [
                'application/pdf',
                'application/octet-stream',
                'image/',
                'video/',
                'audio/',
                'application/zip',
                'application/gzip',
                'application/x-tar',
                'application/x-rar',
                'application/msword',
                'application/vnd.ms-',
                'application/vnd.openxmlformats',
                'font/',
                'application/javascript',
                'application/json'
            ]
            
            # Check for explicitly invalid types first
            for invalid_type in invalid_types:
                if invalid_type in content_type:
                    reason = f"Invalid content type: {content_type}"
                    logging.info(f"⚠️ Skipping {url}: {reason}")
                    return False, reason
            
            # Check for valid types
            for valid_type in valid_types:
                if valid_type in content_type:
                    return True, "Valid HTML/text content"
            
            # If no Content-Type header or unknown type
            if not content_type or content_type == '':
                return True, "No Content-Type header, will validate content"
            
            # Unknown content type - be cautious but try
            logging.warning(f"⚠️ Unknown Content-Type '{content_type}' for {url}, attempting to process")
            return True, f"Unknown Content-Type: {content_type}"
            
        except Exception as e:
            logging.error(f"Error validating Content-Type for {url}: {e}")
            return True, "Content-Type validation error, attempting to process"

    def _validate_content_is_html(self, content: str, url: str) -> tuple:
        """
        Validate that content appears to be HTML/text, not binary data.
        
        Args:
            content: The fetched content string
            url: The URL (for logging)
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            if not content or len(content) < 10:
                return False, "Content too short or empty"
            
            # Check first 500 characters for binary indicators
            sample = content[:500]
            
            # PDF signature check
            if sample.startswith('%PDF') or '%PDF-' in sample[:20]:
                logging.info(f"⚠️ Skipping {url}: PDF binary content detected")
                return False, "PDF binary content"
            
            # Binary/compressed data indicators
            non_printable_count = sum(1 for c in sample if ord(c) < 32 and c not in '\n\r\t')
            non_printable_ratio = non_printable_count / len(sample) if sample else 1
            
            if non_printable_ratio > 0.1:  # More than 10% non-printable
                logging.info(f"⚠️ Skipping {url}: Binary/compressed content detected ({non_printable_ratio:.1%} non-printable)")
                return False, f"Binary content ({non_printable_ratio:.1%} non-printable characters)"
            
            # Check for common binary file signatures
            binary_signatures = [
                '\x00',           # Null bytes
                '\x1f\x8b',       # Gzip
                'PK\x03\x04',     # ZIP/DOCX/XLSX
            ]
            
            for sig in binary_signatures:
                if sig in sample[:20]:
                    logging.info(f"⚠️ Skipping {url}: Binary file signature detected")
                    return False, "Binary file signature detected"
            
            return True, "Content appears to be valid HTML/text"
            
        except Exception as e:
            logging.error(f"Error validating content for {url}: {e}")
            return False, f"Content validation error: {e}"

    def _detect_login_wall(self, content: str, url: str) -> tuple:
        """
        Detect if page content is behind a login wall or paywall.
        
        Args:
            content: The page content
            url: The URL (for logging)
            
        Returns:
            Tuple[bool, str]: (is_login_wall, reason)
        """
        try:
            content_lower = content.lower()
            
            # Login wall indicators
            login_indicators = [
                'sign in to view',
                'log in to view',
                'login to view',
                'sign in to continue',
                'log in to continue',
                'create an account',
                'sign up to view',
                'register to view',
                'join to view',
                'members only',
                'subscribe to read',
                'subscribe to view',
                'subscription required',
                'premium content',
                'unlock this content',
                'please sign in',
                'please log in',
                'authentication required',
                'you must be logged in',
                'login required'
            ]
            
            # Check for login indicators
            found_indicators = []
            for indicator in login_indicators:
                if indicator in content_lower:
                    found_indicators.append(indicator)
            
            if len(found_indicators) >= 2:
                reason = f"Login wall detected: {', '.join(found_indicators[:3])}"
                logging.info(f"⚠️ Skipping {url}: {reason}")
                return True, reason
            
            # Short content with login indicator
            if len(content) < 2000 and found_indicators:
                reason = f"Short page with login indicator: {found_indicators[0]}"
                logging.info(f"⚠️ Skipping {url}: {reason}")
                return True, reason
            
            # LinkedIn special case
            domain = urlparse(url).netloc.lower()
            if 'linkedin.com' in domain:
                if 'sign in' in content_lower or 'log in' in content_lower:
                    reason = "LinkedIn login wall detected"
                    logging.info(f"⚠️ Skipping {url}: {reason}")
                    return True, reason
            
            return False, "No login wall detected"
            
        except Exception as e:
            logging.error(f"Error detecting login wall for {url}: {e}")
            return False, "Login wall detection error"

    def _strip_ai_thinking(self, ai_response: str) -> str:
        """
        Strip chain-of-thought reasoning and internal thinking from AI responses.
        
        DeepSeek and similar models often include their reasoning process.
        This removes those patterns to get clean extracted knowledge.
        
        Args:
            ai_response: The raw AI response
            
        Returns:
            str: Cleaned response with thinking patterns removed
        """
        try:
            if not ai_response:
                return ""
            
            cleaned = ai_response
            
            # Pattern 1: Remove explicit thinking blocks
            thinking_block_patterns = [
                r'<think>.*?</think>',
                r'<thinking>.*?</thinking>',
                r'<reasoning>.*?</reasoning>',
                r'<internal>.*?</internal>',
                r'\[thinking\].*?\[/thinking\]',
                r'\[internal\].*?\[/internal\]'
            ]
            
            for pattern in thinking_block_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Pattern 2: Remove common chain-of-thought openers
            cot_openers = [
                r"^(?:Okay|Ok|Alright|Let me|Let's|I need to|I should|I'll|I will|First,? I)[\s,]",
                r"^(?:The user wants|The query asks|This question|To answer this)",
                r"^(?:Let me (?:think|analyze|consider|examine|look at|break down))",
                r"^(?:Looking at|Analyzing|Examining|Considering|Breaking down)",
                r"^(?:Step \d+:|First,|Second,|Third,|Finally,|Next,)"
            ]
            
            lines = cleaned.split('\n')
            filtered_lines = []
            skip_until_content = True
            
            for line in lines:
                line_stripped = line.strip()
                
                if skip_until_content and not line_stripped:
                    continue
                
                is_thinking_line = False
                for pattern in cot_openers:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_thinking_line = True
                        break
                
                # Meta-commentary patterns
                meta_patterns = [
                    r"^(?:The (?:content|text|article|page|source) (?:mentions|discusses|talks about|provides|contains))",
                    r"^(?:From (?:the|this) (?:content|text|article|source|page))",
                    r"^(?:Based on (?:the|this|my) (?:analysis|reading|review))",
                    r"^(?:After (?:analyzing|reviewing|reading|examining))",
                    r"^(?:I (?:found|noticed|observed|see|can see) that)"
                ]
                
                for pattern in meta_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_thinking_line = True
                        break
                
                if is_thinking_line:
                    continue
                else:
                    skip_until_content = False
                    filtered_lines.append(line)
            
            cleaned = '\n'.join(filtered_lines)
            
            # Pattern 3: Remove task-reference phrases
            task_references = [
                r"(?:The (?:relevant|extracted|key|important) (?:information|content|points|data) (?:is|are|includes?):?)",
                r"(?:Here (?:is|are) the (?:extracted|relevant|key) (?:information|points|content):?)",
                r"(?:(?:Key|Main|Important|Relevant) (?:points?|information|findings?|takeaways?):?)"
            ]
            
            for pattern in task_references:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Clean up whitespace
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            cleaned = re.sub(r' {2,}', ' ', cleaned)
            cleaned = cleaned.strip()
            
            # If we stripped too much, return partial cleanup
            if len(cleaned) < len(ai_response) * 0.2:
                logging.warning("AI thinking stripping removed too much content, using partial cleanup")
                partial_clean = ai_response
                for pattern in thinking_block_patterns:
                    partial_clean = re.sub(pattern, '', partial_clean, flags=re.DOTALL | re.IGNORECASE)
                return partial_clean.strip()
            
            return cleaned
            
        except Exception as e:
            logging.error(f"Error stripping AI thinking: {e}")
            return ai_response

    def _handle_response_encoding(self, response, url: str) -> Optional[str]:
        """
        Properly handle response encoding to avoid garbled text.
        
        Args:
            response: The requests Response object
            url: The URL (for logging)
            
        Returns:
            Optional[str]: Decoded text content or None if decoding fails
        """
        try:
            # First, try the apparent encoding from the response
            if response.encoding:
                try:
                    text = response.text
                    if text.count('�') < len(text) * 0.01:
                        return text
                except Exception:
                    pass
            
            # Check for charset in Content-Type header
            content_type = response.headers.get('Content-Type', '')
            charset_match = re.search(r'charset=([^\s;]+)', content_type, re.IGNORECASE)
            if charset_match:
                detected_encoding = charset_match.group(1).strip('"\'')
                try:
                    text = response.content.decode(detected_encoding)
                    if text.count('�') < len(text) * 0.01:
                        return text
                except Exception:
                    pass
            
            # Try common encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
            
            for encoding in encodings_to_try:
                try:
                    text = response.content.decode(encoding)
                    if text.count('�') < len(text) * 0.01:
                        logging.debug(f"Successfully decoded {url} with {encoding}")
                        return text
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # Try chardet if available
            try:
                import chardet
                detected = chardet.detect(response.content)
                if detected and detected.get('encoding'):
                    text = response.content.decode(detected['encoding'])
                    logging.debug(f"Decoded {url} with chardet-detected {detected['encoding']}")
                    return text
            except ImportError:
                pass
            except Exception:
                pass
            
            # Last resort: decode with errors='replace'
            text = response.content.decode('utf-8', errors='replace')
            replacement_ratio = text.count('�') / len(text) if text else 1
            
            if replacement_ratio > 0.05:
                logging.warning(f"⚠️ High garbled character ratio ({replacement_ratio:.1%}) for {url}")
                return None
            
            return text
            
        except Exception as e:
            logging.error(f"Error handling encoding for {url}: {e}")
            return None

    # ========================================================================
    # END NEW HELPER METHODS
    # ========================================================================

    def _search_startpage(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search using StartPage with enhanced scraping and error handling.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of search results with 'url' and 'title' keys
        """
        try:
            logging.info(f"🔍 Starting StartPage search for: '{query}'")
            
            # Check if StartPage is blacklisted
            startpage_url = "https://www.startpage.com"
            if self._is_domain_blacklisted(startpage_url):
                logging.info("⚠️ StartPage is blacklisted, skipping")
                return []
            
            # Apply enhanced delays
            self._apply_enhanced_delays()
            
            # Prepare search URL
            search_url = "https://www.startpage.com/sp/search"
            params = {
                'query': query,
                'cat': 'web',
                'pl': '',
                'language': 'english',
                'rcount': max_results
            }
            
            # Get enhanced headers
            headers = self._get_headers_rotation()
            
            # Make request with retries
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(search_url, params=params, headers=headers, timeout=15)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        logging.warning(f"🚫 Rate limited by StartPage (attempt {attempt + 1})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status_code in [403, 503]:
                        logging.warning(f"🚫 Blocked by StartPage: HTTP {response.status_code}")
                        self._blacklist_domain(startpage_url)
                        return []
                    else:
                        logging.warning(f"⚠️ StartPage returned HTTP {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ StartPage timeout (attempt {attempt + 1})")
                    if attempt == 2:  # Last attempt
                        return []
                        
                except requests.exceptions.ConnectionError:
                    logging.warning(f"🔌 StartPage connection error")
                    self._blacklist_domain(startpage_url)
                    return []
            
            if not response or response.status_code != 200:
                logging.warning("❌ StartPage search failed after retries")
                return []
            
            # Parse results
            results = self._parse_startpage_results(response.text, max_results)
            
            if results:
                logging.info(f"✅ StartPage search successful: {len(results)} results")
            else:
                logging.info("⚠️ No results found in StartPage response")
                
            return results
            
        except Exception as e:
            logging.error(f"❌ Error in StartPage search: {e}", exc_info=True)
            return []

    def _parse_startpage_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Parse StartPage HTML results."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # StartPage result selectors (may need adjustment based on their current HTML structure)
            result_selectors = [
                '.w-gl__result',
                '.result',
                '.web-result',
                '[data-testid="result"]'
            ]
            
            result_elements = []
            for selector in result_selectors:
                result_elements = soup.select(selector)
                if result_elements:
                    break
            
            if not result_elements:
                # Fallback: look for any links that might be results
                result_elements = soup.find_all('a', href=True)
                
            for element in result_elements[:max_results * 2]:  # Get extra in case some are filtered out
                try:
                    # Extract URL
                    url = None
                    if element.name == 'a':
                        url = element.get('href', '')
                    else:
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            url = link_elem.get('href', '')
                    
                    if not url or not url.startswith('http'):
                        continue
                    
                    # Skip if domain is blacklisted
                    if self._is_domain_blacklisted(url):
                        continue
                    
                    # Extract title
                    title = ""
                    if element.name == 'a':
                        title = element.get_text(strip=True)
                    else:
                        # Try to find title in various places
                        title_elem = element.find(['h2', 'h3', 'h4', '.title', '.result-title'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                        else:
                            # Fallback to any text content
                            title = element.get_text(strip=True)[:100]
                    
                    if title and len(title) >= self.min_content_length:
                        results.append({
                            'url': url,
                            'title': title
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as parse_error:
                    logging.debug(f"Error parsing StartPage result element: {parse_error}")
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Error parsing StartPage results: {e}")
            return []
        
        
    def _is_valid_result(self, url: str, title: str) -> bool:
        """Validate if URL and title represent a valid search result."""
        if not url or not url.startswith('http'):
            return False
        
        if not title or len(title.strip()) < 5:
            return False
        
        # Skip Bing internal URLs
        if any(domain in url.lower() for domain in ['bing.com', 'microsoft.com/bing']):
            return False
        
        # Skip blacklisted domains
        if self._is_domain_blacklisted(url):
            return False
        
        return True
    
    def _parse_brave_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Parse Brave search results."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # Brave search result selectors
            result_selectors = [
                'div[data-type="web"] a[href^="http"]',  # Main web results
                '.snippet-content a[href^="http"]',      # Snippet links
                '.result a[href^="http"]',               # Generic result links
                'a[href^="http"]'                        # Fallback: any external links
            ]
            
            for selector in result_selectors:
                links = soup.select(selector)
                logging.info(f"🦁 Brave selector '{selector}': {len(links)} links found")
                
                for link in links:
                    try:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        # Filter out Brave internal links and ensure quality
                        if (href.startswith('http') and 
                            'brave.com' not in href and
                            'search.brave.com' not in href and
                            len(text) >= 15 and
                            not self._is_domain_blacklisted(href)):
                            
                            results.append({
                                'url': href,
                                'title': text[:100] + '...' if len(text) > 100 else text
                            })
                            
                            if len(results) >= max_results:
                                return results
                                
                    except Exception:
                        continue
            
            logging.info(f"🦁 Brave extracted {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"Error parsing Brave results: {e}")
            return []
        
    
    def _search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search Wikipedia using the MediaWiki API.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of search results with 'url' and 'title' keys
        """
        try:
            logging.info(f"📚 Starting Wikipedia search for: '{query}'")
            
            # Apply enhanced delays (though Wikipedia is usually reliable)
            self._apply_enhanced_delays()
            
            # MediaWiki API endpoint (the one that actually works)
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'srprop': 'snippet|titlesnippet'
            }
            
            headers = {
                'User-Agent': 'WebKnowledgeSeeker/1.0 (Educational AI Research)',
                'Accept': 'application/json'
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if 'query' in data and 'search' in data['query']:
                    for page in data['query']['search'][:max_results]:
                        title = page.get('title', '')
                        snippet = page.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                        
                        if title:
                            # Construct Wikipedia URL
                            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                            
                            # Create descriptive title
                            display_title = title
                            if snippet:
                                display_title = f"{title} - {snippet[:100]}..."
                            
                            results.append({
                                'url': url,
                                'title': display_title
                            })
                    
                    if results:
                        logging.info(f"✅ Wikipedia search successful: {len(results)} results")
                        return results
                    else:
                        logging.info("⚠️ No Wikipedia results found")
                else:
                    logging.warning("⚠️ Wikipedia API returned unexpected response structure")
            else:
                logging.warning(f"⚠️ Wikipedia API returned HTTP {response.status_code}")
            
            return []
            
        except Exception as e:
            logging.error(f"❌ Error in Wikipedia search: {e}", exc_info=True)
            return []

    
    def _parse_wikipedia_results(self, json_data: dict, max_results: int) -> List[Dict[str, str]]:
        """Parse Wikipedia API results."""
        try:
            results = []
            pages = json_data.get('pages', [])
            
            for page in pages[:max_results]:
                title = page.get('title', '')
                description = page.get('description', '')
                key = page.get('key', '')
                
                if title and key:
                    # Construct Wikipedia URL
                    url = f"https://en.wikipedia.org/wiki/{key}"
                    
                    # Create descriptive title
                    display_title = title
                    if description:
                        display_title = f"{title} - {description}"
                    
                    results.append({
                        'url': url,
                        'title': display_title
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error parsing Wikipedia results: {e}")
            return []
        
    def search_for_knowledge(self, topic: str, description: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced knowledge search with Claude discussion as intelligent fallback."""
        try:
            logging.info(f"🔍 Starting knowledge search for: '{topic}'")
            
            search_queries = self._generate_search_queries(topic, description)
            acquired_knowledge = []
            processed_urls = set()
            successful_engine = None
            
            # Try web search engines
            web_engines = ['startpage', 'wikipedia']
            
            for engine in web_engines:
                logging.info(f"🌐 Attempting search with engine: {engine}")
                
                engine_results = []
                for query in search_queries[:3]:
                    search_results = self._execute_search_by_engine(engine, query, max_results)
                    if search_results:
                        engine_results.extend(search_results)
                
                if engine_results:
                    successful_engine = engine
                    logging.info(f"🎯 Processing {len(engine_results)} results from {engine}")
                    
                    # ADD THIS MISSING SECTION: Process each search result
                    for result in engine_results:
                        url = result.get('url', '')
                        title = result.get('title', '')
                        
                        # Skip if we've already processed this URL
                        if url in processed_urls:
                            continue
                        processed_urls.add(url)
                        
                        # Skip blacklisted domains
                        if self._is_domain_blacklisted(url):
                            logging.info(f"⚠️ Skipping blacklisted domain: {url}")
                            continue
                        
                        logging.info(f"📄 Processing result: {title[:50]}... from {url}")
                        
                        # Fetch and process the webpage content
                        content = self._fetch_webpage_content_enhanced(url)
                        if not content:
                            logging.warning(f"Failed to fetch content from {url}")
                            continue
                        
                        # Use AI to extract relevant knowledge
                        extracted_knowledge = self._extract_knowledge_with_ai_improved(
                            content, topic, description, url, title
                        )
                        
                        if extracted_knowledge:
                            acquired_knowledge.extend(extracted_knowledge)
                            logging.info(f"✅ Extracted {len(extracted_knowledge)} knowledge items from {url}")
                        else:
                            logging.info(f"⚠️ No relevant knowledge extracted from {url}")
                    
                    # If we got useful knowledge, we can stop trying other engines
                    if acquired_knowledge:
                        break
            
            # If no knowledge acquired from web, use Claude discussion
            if not acquired_knowledge:
                logging.info(f"🤖 Web search unsuccessful, initiating Claude knowledge discussion")
                
                claude_results = self._search_with_claude_discussion(topic, max_results)
                
                if claude_results:
                    # Create knowledge items from Claude discussion
                    for result in claude_results:
                        if result.get('status') == 'discussion_initiated':
                            knowledge_item = {
                                'content': f"Claude knowledge discussion initiated for topic: {topic}. Description: {description}. The AI-to-AI discussion has been stored in memory and can be retrieved using search commands.",
                                'topic': topic,
                                'description': description,
                                'source': 'claude_discussion',
                                'title': f'Claude Knowledge Discussion: {topic}',
                                'search_query': topic,
                                'transaction_id': str(uuid.uuid4()),
                                'items_stored': 1,
                                'relevance_score': 0.9,
                                'extracted_at': datetime.now().isoformat(),
                                'extraction_method': 'claude_discussion'
                            }
                            acquired_knowledge.append(knowledge_item)
                            
                    successful_engine = 'claude_discussion'
                    logging.info(f"✅ Claude discussion completed for topic '{topic}'")
            
            # Log final results
            if acquired_knowledge:
                logging.info(f"🎯 Knowledge acquisition complete: {len(acquired_knowledge)} items using {successful_engine}")
            else:
                logging.warning(f"❌ No knowledge acquired for topic '{topic}' - all methods failed")
            
            return acquired_knowledge
            
        except Exception as e:
            logging.error(f"❌ Error in knowledge search: {e}", exc_info=True)
            return []

    def _execute_search_by_engine(self, engine: str, query: str, max_results: int) -> List[Dict[str, str]]:
        """Execute search using specified engine."""
        try:
            if engine == 'searx':
                logging.info("⚠️ Searx engine disabled - skipping")
                return []
            elif engine == 'startpage':
                return self._search_startpage(query, max_results)
            elif engine == 'wikipedia':
                return self._search_wikipedia(query, max_results)
            elif engine == 'claude_discussion':
                return self._search_with_claude_discussion(query, max_results)
            else:
                logging.error(f"Unknown search engine: {engine}")
                return []
                
        except Exception as e:
            logging.error(f"Error executing search with {engine}: {e}", exc_info=True)
            return []
    
    def _search_with_claude_discussion(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Use [DISCUSS_WITH_CLAUDE:] command to get knowledge instead of fallback search.
        
        Args:
            query (str): Search query/topic
            max_results (int): Not used but kept for interface consistency
            
        Returns:
            List[Dict[str, str]]: Results indicating Claude discussion was initiated
        """
        try:
            logging.info(f"🤖 Initiating Claude discussion for knowledge gap: '{query}'")
            
            # Check if we have access to Claude knowledge integration
            if not self.chatbot or not hasattr(self.chatbot, 'claude_knowledge'):
                logging.warning("Claude knowledge integration not available")
                return []
            
            # Use your existing Claude knowledge system
            claude_integration = self.chatbot.claude_knowledge
            
            # Start a focused knowledge discussion with Claude
            topic = f"web search knowledge gap: {query}"
            description = f"The autonomous AI system couldn't find current information about '{query}' through web search engines. Please provide expert knowledge and guidance on this topic."
            
            # Use the free-form discussion method for interactive learning
            discussion_success = claude_integration.engage_in_free_form_discussion(topic, description)
            
            if discussion_success:
                logging.info(f"✅ Successfully initiated Claude discussion about '{query}'")
                
                # Return a result that indicates the discussion was started
                return [{
                    'url': 'claude_discussion',
                    'title': f'Claude Discussion: {query}',
                    'content': f'Initiated direct AI-to-AI knowledge discussion with Claude about: {query}. The conversation and insights have been stored in memory for future reference.',
                    'discussion_topic': topic,
                    'status': 'discussion_initiated'
                }]
            else:
                logging.warning(f"Failed to initiate Claude discussion about '{query}'")
                return []
                
        except Exception as e:
            logging.error(f"Error initiating Claude discussion for '{query}': {e}")
            return []

    def _fetch_webpage_content_enhanced(self, url: str) -> Optional[str]:
        """
        Enhanced webpage content fetching with comprehensive error handling and anti-blocking.
        """
        try:
            # Check if domain is blacklisted
            if self._is_domain_blacklisted(url):
                logging.info(f"⚠️ Skipping blacklisted domain: {url}")
                return None
            
            # Apply enhanced delays
            self._apply_enhanced_delays()
            
            # Try multiple fetching strategies
            fetch_strategies = [
                ('enhanced_session', self._fetch_with_enhanced_session),
                ('retry_with_backoff', self._fetch_with_retries_enhanced),
                ('basic_fallback', self._fetch_basic_fallback)
            ]
            
            for strategy_name, strategy_func in fetch_strategies:
                try:
                    logging.info(f"🌐 Trying {strategy_name} for {url}")
                    response_text = strategy_func(url)
                    
                    if response_text:
                        logging.info(f"✅ {strategy_name} succeeded for {url}")
                        # Process and clean the content
                        cleaned_content = self._process_webpage_content(response_text, url)
                        return cleaned_content
                    else:
                        logging.info(f"⚠️ {strategy_name} returned no content for {url}")
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ {strategy_name} timed out for {url}")
                    
                except requests.exceptions.ConnectionError:
                    logging.warning(f"🔌 {strategy_name} connection error for {url}")
                    
                except requests.exceptions.HTTPError as e:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code in [403, 429, 503]:
                        error_type = str(status_code)
                        logging.warning(f"🚫 {strategy_name} blocked for {url}: HTTP {status_code}")
                        self._blacklist_domain_graduated(url, error_type)
                        return None
                    else:
                        logging.warning(f"⚠️ {strategy_name} HTTP error for {url}: {e}")
                        
                except Exception as strategy_error:
                    logging.warning(f"❌ {strategy_name} failed for {url}: {strategy_error}")
                    continue
            
            # If all strategies failed
            logging.warning(f"❌ All fetch strategies failed for {url}")
            self._blacklist_domain_graduated(url, 'fetch_failed')
            return None
            
        except Exception as e:
            logging.error(f"❌ Error in enhanced webpage fetch for {url}: {e}", exc_info=True)
            return None

    def _fetch_with_enhanced_session(self, url: str) -> Optional[str]:
        """
        Fetch using enhanced session with realistic browser behavior.
        UPDATED: Added Content-Type validation and encoding handling.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Optional[str]: The page content as text, or None if fetch fails
        """
        try:
            # Update session headers to look more realistic
            self.session.headers.update(self._get_headers_rotation())
            
            # Add random delay to simulate human behavior
            time.sleep(random.uniform(0.5, 2.0))
            
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            # NEW: Validate Content-Type before processing
            is_valid_type, type_reason = self._validate_content_type(response, url)
            if not is_valid_type:
                logging.info(f"⚠️ Skipping {url}: {type_reason}")
                return None
            
            # NEW: Handle encoding properly
            text = self._handle_response_encoding(response, url)
            if not text:
                logging.warning(f"⚠️ Failed to decode content from {url}")
                return None
            
            # NEW: Validate content appears to be HTML/text
            is_valid_content, content_reason = self._validate_content_is_html(text, url)
            if not is_valid_content:
                logging.info(f"⚠️ Skipping {url}: {content_reason}")
                return None
            
            return text
            
        except Exception as e:
            logging.debug(f"Enhanced session fetch failed for {url}: {e}")
            return None

    def _fetch_with_retries_enhanced(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Enhanced retry mechanism with exponential backoff and header rotation.
        UPDATED: Added Content-Type validation and encoding handling.
        """
        for attempt in range(max_retries):
            try:
                headers = self._get_headers_rotation()
                
                # Exponential backoff with jitter
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                
                response = requests.get(url, headers=headers, timeout=15 + (attempt * 5))
                response.raise_for_status()
                
                # NEW: Validate Content-Type
                is_valid_type, type_reason = self._validate_content_type(response, url)
                if not is_valid_type:
                    logging.info(f"Skipping {url}: {type_reason}")
                    return None
                
                # NEW: Handle encoding properly
                text = self._handle_response_encoding(response, url)
                if not text:
                    logging.warning(f"Failed to decode content from {url}")
                    continue  # Try again with next attempt
                
                # NEW: Validate content
                is_valid_content, content_reason = self._validate_content_is_html(text, url)
                if not is_valid_content:
                    logging.info(f"Skipping {url}: {content_reason}")
                    return None
                
                return text
                
            except Exception as e:
                logging.debug(f"Retry attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return None

    def _fetch_basic_fallback(self, url: str) -> Optional[str]:
        """
        Basic fallback fetch method.
        UPDATED: Added Content-Type validation and encoding handling.
        """
        try:
            basic_headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; WebKnowledgeSeeker/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            time.sleep(random.uniform(2, 4))
            response = requests.get(url, headers=basic_headers, timeout=25)
            response.raise_for_status()
            
            # NEW: Validate Content-Type
            is_valid_type, type_reason = self._validate_content_type(response, url)
            if not is_valid_type:
                logging.info(f"Skipping {url}: {type_reason}")
                return None
            
            # NEW: Handle encoding properly
            text = self._handle_response_encoding(response, url)
            if not text:
                logging.warning(f"Failed to decode content from {url}")
                return None
            
            # NEW: Validate content
            is_valid_content, content_reason = self._validate_content_is_html(text, url)
            if not is_valid_content:
                logging.info(f"Skipping {url}: {content_reason}")
                return None
            
            return text
            
        except Exception as e:
            logging.debug(f"Basic fallback fetch failed for {url}: {e}")
            return None

    def _process_webpage_content(self, html_content: str, url: str) -> Optional[str]:
        """
        Process and clean webpage content with enhanced filtering.
        UPDATED: Added login wall detection and improved content validation.
        """
        try:
            # NEW: Check for login walls first
            is_login_wall, login_reason = self._detect_login_wall(html_content, url)
            if is_login_wall:
                logging.info(f"Skipping {url}: {login_reason}")
                return None
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe']
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove unwanted classes and IDs - EXPANDED LIST
            unwanted_selectors = [
                '.advertisement', '.ad', '.ads', '.banner', '.popup',
                '.cookie-notice', '.newsletter', '.social-share',
                '#comments', '.comments', '.sidebar', '.related-posts',
                '.login-form', '.signup-form', '.auth-wall',
                '.paywall', '.subscription-wall',
                '.social-login', '.oauth-buttons'
            ]
            
            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Try to find main content
            main_content = None
            content_selectors = [
                'main', 'article', '.content', '.main-content', '#content',
                '.post-content', '.entry-content', '.article-body', '.story-body'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                main_content = soup
            
            # Extract and clean text
            text = main_content.get_text(separator=' ', strip=True)
            
            # Clean up whitespace and formatting
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = ' '.join(lines)
            
            # Remove common unwanted phrases - EXPANDED LIST
            unwanted_phrases = [
                'Cookie Policy', 'Privacy Policy', 'Terms of Service', 'Terms and Conditions',
                'Subscribe to our newsletter', 'Sign up for updates', 'Follow us on',
                'Share this article', 'Like us on Facebook', 'Tweet', 'LinkedIn',
                'Advertisement', 'Sponsored Content', 'Continue Reading',
                'Sign in to view', 'Log in to continue', 'Create an account',
                'Already have an account', 'Forgot password', 'Remember me',
                'Sign up for free', 'Join now', 'Get started',
                'Share on Twitter', 'Share on Facebook', 'Share on LinkedIn',
                'Click to share', 'Pin it', 'Email this',
                'Skip to content', 'Skip to main content', 'Back to top',
                'Previous article', 'Next article', 'Related articles',
                'Read more', 'See more', 'Load more', 'Show more'
            ]
            
            for phrase in unwanted_phrases:
                text = re.sub(rf'\b{re.escape(phrase)}\b', '', text, flags=re.IGNORECASE)
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Validate content length
            if len(text) < self.min_content_length:
                logging.warning(f"Content too short ({len(text)} chars) for {url}")
                return None
            
            # NEW: Additional validation - check for meaningful content
            words = text.split()
            if words:
                avg_word_length = sum(len(w) for w in words) / len(words)
                if avg_word_length < 3.5 and len(words) < 200:
                    logging.warning(f"Content appears to be mostly navigation ({avg_word_length:.1f} avg word len) for {url}")
                    return None
            
            # Limit content length to prevent processing huge pages
            max_length = 15000  # 15KB limit
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            logging.info(f"Processed {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logging.error(f"Error processing webpage content from {url}: {e}")
            return None
    
    def _generate_search_queries(self, topic: str, description: str) -> List[str]:
        """Enhanced search query generation with better strategies."""
        queries = []
        
        # Base query with topic
        queries.append(topic)
        
        # Enhanced query generation based on description
        if description:
            # Extract meaningful keywords (longer than 3 chars, alphabetic)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
            key_phrases = [word for word in words if word not in [
                'what', 'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'said',
                'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'
            ]]
            
            if key_phrases:
                # Combine topic with most relevant phrases
                for phrase in key_phrases[:2]:  # Use top 2 phrases
                    queries.append(f"{topic} {phrase}")
                
                # Add specific query types
                queries.append(f"what is {topic}")
                queries.append(f"{topic} guide tutorial")
                queries.append(f"{topic} explained")
                
                # If it seems technical, add technical queries
                if any(tech_term in description.lower() for tech_term in [
                    'api', 'code', 'programming', 'software', 'algorithm', 'method', 'function'
                ]):
                    queries.append(f"{topic} best practices")
                    queries.append(f"{topic} examples documentation")
        
        # Add some general educational queries
        queries.extend([
            f"{topic} overview",
            f"learn {topic}",
            f"{topic} fundamentals"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            query_clean = query.lower().strip()
            if query_clean not in seen and len(query_clean) > 2:
                seen.add(query_clean)
                unique_queries.append(query)
        
        logging.info(f"🔍 Generated {len(unique_queries)} search queries for topic '{topic}'")
        return unique_queries[:5]  # Limit to 5 queries max

    def _extract_knowledge_with_ai_improved(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> List[Dict[str, Any]]:
        """
        Improved AI knowledge extraction with multiple fallback levels.
        """
        try:
            if not self.chatbot or not hasattr(self.chatbot, 'llm'):
                logging.warning("AI chatbot not available, using content-based extraction")
                return self._create_content_based_knowledge_item(content, topic, source_url, title)
            
            # Try multiple extraction approaches with decreasing strictness
            extraction_attempts = [
                ('strict', self._create_strict_extraction_prompt),
                ('moderate', self._create_moderate_extraction_prompt),
                ('flexible', self._create_flexible_extraction_prompt)
            ]
            
            for approach_name, prompt_creator in extraction_attempts:
                try:
                    logging.info(f"🤖 Trying {approach_name} AI extraction for {source_url}")
                    
                    prompt = prompt_creator(content, topic, description, source_url, title)
                    ai_response = self.chatbot.llm.invoke(prompt)
                    
                    if ai_response and "NO_RELEVANT_CONTENT_FOUND" not in ai_response:
                        knowledge_items = self._parse_ai_extracted_knowledge_enhanced(
                            ai_response, topic, source_url, title, description
                        )
                        
                        if knowledge_items:
                            logging.info(f"🤖 {approach_name} extraction successful: {len(knowledge_items)} items")
                            return knowledge_items
                        else:
                            logging.info(f"🤖 {approach_name} extraction returned no parseable items")
                    else:
                        logging.info(f"🤖 {approach_name} extraction found no relevant content")
                        
                except Exception as ai_error:
                    logging.warning(f"🤖 {approach_name} extraction failed: {ai_error}")
                    continue
            
            # Final fallback: content-based extraction
            logging.info("🤖 All AI extraction methods failed, using content-based fallback")
            return self._create_content_based_knowledge_item(content, topic, source_url, title)
            
        except Exception as e:
            logging.error(f"Error in improved AI knowledge extraction: {e}")
            return self._create_content_based_knowledge_item(content, topic, source_url, title)

    def _create_flexible_extraction_prompt(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> str:
        """Create a more flexible extraction prompt that accepts broader relevance."""
        return f"""
        Extract useful information from this content that relates to: "{topic}"

        Content from: {title} ({source_url})
        Target: {topic}
        Context: {description}

        Content: {content[:3000]}...

        Instructions:
        - Look for ANY information related to "{topic}" or similar concepts
        - Include background information, related topics, or contextual details
        - Each piece of information should be factual and complete
        - If you find relevant information, format as numbered points
        - If truly no relevant information exists, respond with: "NO_RELEVANT_CONTENT_FOUND"

        Extract information:
        """

    def _create_strict_extraction_prompt(self, content: str, topic: str, description: str, 
                                    source_url: str, title: str) -> str:
        """Create a strict extraction prompt that only accepts highly relevant content."""
        return f"""
        I need to extract ONLY information that DIRECTLY addresses this specific knowledge gap.

        KNOWLEDGE GAP:
        Topic: "{topic}"
        Description: "{description}"

        SOURCE MATERIAL:
        Title: {title}
        URL: {source_url}
        Content: {content[:2500]}...

        STRICT REQUIREMENTS:
        1. Extract ONLY information that directly explains or teaches about "{topic}"
        2. Information must be factual, specific, and actionable
        3. Each extracted point must be directly relevant to: {description}
        4. Ignore general background, marketing content, or tangentially related topics
        5. If no directly relevant information exists, respond with: "NO_RELEVANT_CONTENT_FOUND"

        FORMAT: Number each direct insight (1., 2., 3.)
        
        Extract strictly relevant information about "{topic}":
        """

    def _create_moderate_extraction_prompt(self, content: str, topic: str, description: str, 
                                        source_url: str, title: str) -> str:
        """Create a moderately flexible extraction prompt."""
        return f"""
        Extract information that relates to or helps understand: "{topic}"

        SOURCE: {title} ({source_url})
        TOPIC: {topic}
        CONTEXT: {description}

        Content: {content[:3500]}...

        INSTRUCTIONS:
        - Extract information that directly relates to "{topic}"
        - Include related concepts that provide context or understanding
        - Focus on factual information, examples, or explanations
        - Each point should be informative and complete
        - If no relevant information found, respond with: "NO_RELEVANT_CONTENT_FOUND"

        Format as numbered points. Extract information:
        """

    def _create_content_based_knowledge_item(self, content: str, topic: str, source_url: str, title: str) -> List[Dict[str, Any]]:
        """Create knowledge items based on content analysis without AI."""
        try:
            # Extract sentences containing the topic
            sentences = re.split(r'[.!?]+', content)
            relevant_sentences = []
            
            topic_lower = topic.lower()
            topic_words = set(topic_lower.split())
            
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if (len(sentence_clean) > 30 and  # Minimum length
                    any(word in sentence_clean.lower() for word in topic_words)):
                    relevant_sentences.append(sentence_clean)
            
            # Take best relevant sentences
            if relevant_sentences:
                # Combine up to 3 best sentences
                combined_content = '. '.join(relevant_sentences[:3]) + '.'
                
                return [self._create_knowledge_item_enhanced(
                    combined_content, topic, source_url, title, 
                    f"Content-based extraction from {title}"
                )]
            
            # Ultimate fallback: take first substantial paragraph mentioning topic
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if (len(paragraph) > 100 and 
                    topic_lower in paragraph.lower()):
                    return [self._create_knowledge_item_enhanced(
                        paragraph[:500] + '...', topic, source_url, title,
                        f"Paragraph extraction from {title}"
                    )]
            
            return []
            
        except Exception as e:
            logging.error(f"Error in content-based knowledge extraction: {e}")
            return []

    def _parse_ai_extracted_knowledge_enhanced(self, ai_response: str, topic: str, 
                                             source_url: str, title: str, description: str) -> List[Dict[str, Any]]:
        """
        Enhanced parsing of AI-extracted knowledge with better validation.
        UPDATED: Added chain-of-thought stripping to remove AI reasoning.
        """
        try:
            knowledge_items = []
            
            # NEW: Strip AI thinking/reasoning before processing
            response_cleaned = self._strip_ai_thinking(ai_response)
            
            if not response_cleaned:
                logging.warning("AI response was empty after stripping thinking content")
                return []
            
            # Log if significant content was stripped
            original_len = len(ai_response)
            cleaned_len = len(response_cleaned)
            if original_len > 0 and (original_len - cleaned_len) / original_len > 0.3:
                logging.info(f"Stripped {((original_len - cleaned_len) / original_len * 100):.1f}% AI thinking content")
            
            # Split into numbered items more intelligently
            # Look for patterns like "1.", "2.", etc.
            numbered_pattern = r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=(?:\n\s*\d+\.)|$)'
            matches = re.findall(numbered_pattern, response_cleaned, re.MULTILINE | re.DOTALL)
            
            if matches:
                # Process numbered items
                for number, content in matches:
                    content = content.strip()
                    
                    # NEW: Additional cleaning for each extracted item
                    content = self._strip_ai_thinking(content)
                    
                    if len(content) >= self.min_content_length:
                        # Validate content relevance
                        if self._is_content_relevant(content, topic, description):
                            knowledge_items.append(self._create_knowledge_item_enhanced(
                                content, topic, source_url, title, description
                            ))
                        else:
                            logging.debug(f"Filtered out irrelevant content: {content[:50]}...")
            else:
                # No numbered items found, try to split by sentences or paragraphs
                paragraphs = [p.strip() for p in response_cleaned.split('\n') if p.strip()]
                
                for paragraph in paragraphs:
                    # NEW: Clean each paragraph
                    paragraph = self._strip_ai_thinking(paragraph)
                    
                    if (len(paragraph) >= self.min_content_length and 
                        self._is_content_relevant(paragraph, topic, description)):
                        knowledge_items.append(self._create_knowledge_item_enhanced(
                            paragraph, topic, source_url, title, description
                        ))
            
            # If still no items, treat whole response as one item
            if not knowledge_items and len(response_cleaned) >= self.min_content_length:
                if self._is_content_relevant(response_cleaned, topic, description):
                    knowledge_items.append(self._create_knowledge_item_enhanced(
                        response_cleaned, topic, source_url, title, description
                    ))
            
            logging.info(f"Parsed {len(knowledge_items)} knowledge items from AI response")
            return knowledge_items
            
        except Exception as e:
            logging.error(f"Error in enhanced AI knowledge parsing: {e}")
            return []

    def _is_content_relevant(self, content: str, topic: str, description: str) -> bool:
        """Check if extracted content is relevant to the topic and description."""
        try:
            content_lower = content.lower()
            topic_lower = topic.lower()
            
            # Must contain the topic
            if topic_lower not in content_lower:
                return False
            
            # Check for description keywords
            if description:
                desc_words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
                if desc_words:
                    # At least one description keyword should be present
                    if not any(word in content_lower for word in desc_words[:3]):
                        return False
            
            # Filter out common irrelevant patterns
            irrelevant_patterns = [
                'cookie policy', 'privacy policy', 'terms of service',
                'subscribe', 'newsletter', 'advertisement', 'sponsored',
                'follow us', 'like us', 'share this', 'related articles'
            ]
            
            for pattern in irrelevant_patterns:
                if pattern in content_lower:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking content relevance: {e}")
            return True  # Default to relevant if error

    def _validate_knowledge_items(self, knowledge_items: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Validate and filter knowledge items for quality and relevance."""
        try:
            validated_items = []
            
            for item in knowledge_items:
                content = item.get('content', '')
                
                # Basic validation
                if len(content) < self.min_content_length:
                    continue
                
                # Check for topic relevance
                if topic.lower() not in content.lower():
                    continue
                
                # Check for meaningful content (not just boilerplate)
                if self._is_meaningful_content(content):
                    validated_items.append(item)
                    
            logging.info(f"✅ Validated {len(validated_items)} out of {len(knowledge_items)} knowledge items")
            return validated_items
            
        except Exception as e:
            logging.error(f"Error validating knowledge items: {e}")
            return knowledge_items  # Return original if validation fails

    def _is_meaningful_content(self, content: str) -> bool:
        """Check if content contains meaningful information."""
        try:
            # Check for minimum complexity
            words = content.split()
            if len(words) < 8:  # Too short
                return False
            
            # Check for variety in vocabulary
            unique_words = set(word.lower() for word in words if word.isalpha())
            if len(unique_words) < len(words) * 0.3:  # Too repetitive
                return False
            
            # Check for sentence structure
            sentences = re.split(r'[.!?]+', content)
            if len([s for s in sentences if len(s.strip()) > 10]) < 1:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking meaningful content: {e}")
            return True

    def _create_knowledge_item_enhanced(self, content: str, topic: str, source_url: str, 
                                      title: str, description: str) -> Dict[str, Any]:
        """Create enhanced structured knowledge item with additional metadata."""
        # Clean content
        content_cleaned = re.sub(r'\s+', ' ', content.strip())
        
        # Calculate relevance score based on topic/description match
        relevance_score = self._calculate_relevance_score(content_cleaned, topic, description)
        
        return {
            'content': content_cleaned,
            'topic': topic,
            'description': description,
            'source': source_url,
            'title': title,
            'search_query': topic,
            'transaction_id': str(uuid.uuid4()),
            'items_stored': 1,
            'relevance_score': relevance_score,
            'content_length': len(content_cleaned),
            'word_count': len(content_cleaned.split()),
            'extracted_at': datetime.now().isoformat(),
            'extraction_method': 'ai_enhanced'
        }

    def _calculate_relevance_score(self, content: str, topic: str, description: str) -> float:
        """Calculate relevance score for knowledge item."""
        try:
            score = 0.0
            content_lower = content.lower()
            
            # Topic mentions (base score)
            topic_mentions = content_lower.count(topic.lower())
            score += min(topic_mentions * 0.2, 0.6)  # Max 0.6 for topic mentions
            
            # Description keyword matches
            if description:
                desc_words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
                matches = sum(1 for word in desc_words if word in content_lower)
                if desc_words:
                    score += (matches / len(desc_words)) * 0.3  # Max 0.3 for description match
            
            # Technical content indicators (higher relevance for learning)
            technical_indicators = [
                'example', 'method', 'function', 'implementation', 'approach',
                'technique', 'algorithm', 'process', 'steps', 'procedure'
            ]
            tech_score = sum(0.02 for indicator in technical_indicators if indicator in content_lower)
            score += min(tech_score, 0.1)  # Max 0.1 for technical content
            
            # Ensure minimum score for any extracted content
            score = max(score, 0.3)
            
            # Cap at maximum score
            score = min(score, 1.0)
            
            return round(score, 2)
            
        except Exception as e:
            logging.error(f"Error calculating relevance score: {e}")
            return 0.5  # Default moderate relevance

    def _create_knowledge_item(self, content: str, topic: str, source_url: str, title: str) -> Dict[str, Any]:
        """Create basic structured knowledge item (backwards compatibility)."""
        return {
            'content': content.strip(),
            'topic': topic,
            'source': source_url,
            'title': title,
            'search_query': topic,
            'transaction_id': str(uuid.uuid4()),
            'items_stored': 1,
            'relevance_score': 0.7,  # Default good relevance
            'extracted_at': datetime.now().isoformat()
        }

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search performance and blacklisted domains."""
        try:
            stats = {
                'blacklisted_domains': len(self.failed_domains),
                'blacklisted_domain_list': list(self.failed_domains.keys()),
                'searx_instances': len(self.searx_instances),
                'searx_instance_list': self.searx_instances,
                'current_searx_instance': self.searx_instances[self.current_searx_index] if self.searx_instances else None,
                'search_engines': self.search_engines,
                'min_content_length': self.min_content_length,
                'request_interval': f"{self.min_request_interval}-{self.max_request_interval} seconds"
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting search statistics: {e}")
            return {}

    def clear_blacklist(self):
        """Clear the domain blacklist (useful for testing or manual reset)."""
        try:
            self.failed_domains.clear()
            self._save_blacklist()
            logging.info("🧹 Domain blacklist cleared")
            
        except Exception as e:
            logging.error(f"Error clearing blacklist: {e}")

    
    def debug_bing_parsing(self, query: str = "test query"):
        """Debug method to test Bing parsing with detailed output."""
        try:
            logging.info(f"🧪 DEBUG: Testing Bing parsing for '{query}' with production parameters")
            
            # Use same parameters as production _search_bing method
            search_url = "https://www.bing.com/search"
            params = {
                'q': query,
                'first': 1,
                'FORM': 'PERE',  # Same as production
                'PC': 'U316',    # Same as production
                'ensearch': 1     # Same as production
            }
            
            # Use Bing-specific headers like production
            headers = self._get_bing_specific_headers()
            
            response = requests.get(search_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                debug_file = f'bing_debug_{query.replace(" ", "_")}_production.html'
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logging.info(f"🧪 Saved Bing HTML to '{debug_file}' using production parameters")
                
                # Test selectors with actual production response
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try broader selectors to see what's actually there
                broader_selectors = [
                    'li',  # All list items
                    'h2',  # All h2 tags
                    'h3',  # All h3 tags  
                    'a[href^="http"]',  # All external links
                    '[class*="result"]',  # Anything with "result" in class
                    '[class*="algo"]'    # Anything with "algo" in class
                ]
                
                for selector in broader_selectors:
                    elements = soup.select(selector)
                    logging.info(f"🧪 Broader selector '{selector}': {len(elements)} elements found")
                    
                    if elements and len(elements) < 20:  # Only show details for reasonable numbers
                        for i, elem in enumerate(elements[:3]):
                            logging.info(f"  Sample {i+1}: {str(elem)[:100]}...")
            else:
                logging.error(f"🧪 Bing returned HTTP {response.status_code}")
                
        except Exception as e:
            logging.error(f"🧪 Debug error: {e}")

    def _get_bing_specific_headers(self):
        """Get headers specifically optimized for Bing search."""
        base_headers = self._get_headers_rotation()
        
        # Bing-specific enhancements
        bing_headers = base_headers.copy()
        bing_headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',  # Explicitly handle compression
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        })
        
        # Remove problematic headers that might trigger detection
        headers_to_remove = ['Connection']
        for header in headers_to_remove:
            bing_headers.pop(header, None)
        
        return bing_headers
    
    
    def test_search_engines(self, test_query: str = "python programming") -> Dict[str, bool]:
        """Test all search engines with a simple query."""
        try:
            logging.info(f"🧪 Testing all search engines with query: '{test_query}'")
            
            results = {}
            
            for engine in self.search_engines:
                try:
                    logging.info(f"🧪 Testing {engine}...")
                    search_results = self._execute_search_by_engine(engine, test_query, 1)
                    
                    if search_results:
                        results[engine] = True
                        logging.info(f"✅ {engine} test passed")
                    else:
                        results[engine] = False
                        logging.warning(f"❌ {engine} test failed - no results")
                        
                except Exception as e:
                    results[engine] = False
                    logging.error(f"❌ {engine} test failed with error: {e}")
                
                # Small delay between tests
                time.sleep(2)
            
            working_engines = sum(1 for status in results.values() if status)
            logging.info(f"🧪 Search engine test complete: {working_engines}/{len(results)} engines working")
            
            return results
            
        except Exception as e:
            logging.error(f"Error testing search engines: {e}")
            return {}