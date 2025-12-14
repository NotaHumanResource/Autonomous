"""Session management for tracking user sessions and command usage."""

import uuid
import datetime
import logging
from typing import Optional

class SessionManager:
    """Manages user sessions and integrates with lifetime counters."""
    
    def __init__(self, lifetime_counters):
        """Initialize session manager with lifetime counters reference."""
        self.lifetime_counters = lifetime_counters
        self.current_session_id = None
        self.session_start_time = None
        
    def start_new_session(self) -> str:
        """Start a new session and return the session ID."""
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.datetime.now()
        logging.info(f"Started new session: {self.current_session_id}")
        return self.current_session_id
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.current_session_id
    
    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        if not self.current_session_id:
            return {}
        
        try:
            session_counters = self.lifetime_counters.get_session_counters(self.current_session_id)
            total_commands = sum(session_counters.values())
            
            return {
                'session_id': self.current_session_id,
                'start_time': self.session_start_time.isoformat() if self.session_start_time else None,
                'duration_minutes': (datetime.datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0,
                'total_commands': total_commands,
                'command_breakdown': session_counters
            }
        except Exception as e:
            logging.error(f"Error getting session summary: {e}")
            return {}
    
    def end_session(self):
        """End the current session."""
        if self.current_session_id:
            logging.info(f"Ending session: {self.current_session_id}")
            summary = self.get_session_summary()
            logging.info(f"Session summary: {summary}")
            self.current_session_id = None
            self.session_start_time = None