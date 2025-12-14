# autonomous_utils.py
"""Utility functions for the Autonomous Cognition system."""

import logging
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

def get_memory_stats(db_path: str) -> Dict[str, Any]:
    """Get comprehensive memory storage statistics.
    
    Args:
        db_path (str): Path to the SQLite database
        
    Returns:
        Dict[str, Any]: Dictionary of memory statistics
    """
    try:
        stats = {}
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Total memory count
            cursor.execute("SELECT COUNT(*) FROM memories")
            stats["total_memories"] = cursor.fetchone()[0]
            
            # Memories by type
            cursor.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)
            stats["memory_types"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Autonomous thought count
            cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'autonomous_thought'")
            stats["autonomous_thoughts"] = cursor.fetchone()[0]
            
            # Recent memory activity
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM memories
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY date
                ORDER BY date DESC
            """)
            stats["recent_activity"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average memory weight
            cursor.execute("SELECT AVG(weight) FROM memories")
            stats["avg_weight"] = cursor.fetchone()[0]
            
            # Memory age distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN julianday('now') - julianday(created_at) < 1 THEN 'today'
                        WHEN julianday('now') - julianday(created_at) < 7 THEN 'this_week'
                        WHEN julianday('now') - julianday(created_at) < 30 THEN 'this_month'
                        ELSE 'older'
                    END as age_bucket,
                    COUNT(*) as count
                FROM memories
                GROUP BY age_bucket
            """)
            stats["age_distribution"] = {row[0]: row[1] for row in cursor.fetchall()}
            
        return stats
    
    except Exception as e:
        logging.error(f"Error getting memory stats: {e}", exc_info=True)
        return {"error": str(e)}

def analyze_memory_health(db_path: str) -> Dict[str, Any]:
    """Analyze database health and identify potential issues.
    
    This function checks for:
    - Duplicate content in syncable memories (excluding reminders and autonomous_thoughts)
    - Memories with NULL or empty content
    - Old, low-weight, rarely accessed memories that could be archived
    
    Args:
        db_path (str): Path to the SQLite database
        
    Returns:
        Dict[str, Any]: Dictionary with health analysis including:
            - status: 'healthy', 'issues_found', or 'error'
            - issues: List of identified issues
            - recommendations: List of recommended actions
            - duplicate_details: Detailed info about duplicates (if found)
    """
    try:
        health = {
            "status": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check for duplicate contents (excluding SQLite-only types)
            # Note: autonomous_thoughts and reminders are intentionally excluded as they may legitimately repeat
            # autonomous_thoughts are process logs that may run multiple times
            # reminders are SQLite-only and not synced to vector DB
            cursor.execute("""
                SELECT 
                    content, 
                    COUNT(*) as count,
                    GROUP_CONCAT(id) as memory_ids,
                    GROUP_CONCAT(memory_type) as memory_types,
                    MIN(created_at) as first_created,
                    MAX(created_at) as last_created
                FROM memories
                WHERE memory_type NOT IN ('reminder', 'autonomous_thought')
                GROUP BY content
                HAVING count > 1
                LIMIT 10
            """)
            duplicates = cursor.fetchall()
            
            if duplicates:
                health["status"] = "issues_found"
                
                # Calculate total duplicate entries (not just groups)
                # Each group with count N has N-1 redundant entries
                total_duplicate_entries = sum(count - 1 for content, count, ids, types, first, last in duplicates)
                duplicate_groups = len(duplicates)
                
                health["issues"].append(
                    f"Found {duplicate_groups} duplicate content group(s) "
                    f"({total_duplicate_entries} redundant entries total)"
                )
                
                # Store detailed information about duplicates for review
                health["duplicate_details"] = []
                for content, count, ids, types, first_created, last_created in duplicates[:5]:
                    health["duplicate_details"].append({
                        "content_preview": content[:150] + "..." if len(content) > 150 else content,
                        "count": count,
                        "memory_ids": ids,
                        "memory_types": types,
                        "first_created": first_created,
                        "last_created": last_created
                    })
                
                health["recommendations"].append(
                    "Run 'Remove Duplicates' in System Maintenance to clean up redundant memories. "
                    "Review duplicate_details to see what content is duplicated."
                )
            
            # Check for memories with NULL content
            cursor.execute("SELECT COUNT(*) FROM memories WHERE content IS NULL OR content = ''")
            null_content = cursor.fetchone()[0]
            
            if null_content > 0:
                health["status"] = "issues_found"
                health["issues"].append(f"Found {null_content} memories with empty content")
                health["recommendations"].append("Clean up empty memory entries")
            
            # Check for very old memories with low weight
            # These are candidates for archival or pruning
            cursor.execute("""
                SELECT COUNT(*) FROM memories 
                WHERE julianday('now') - julianday(created_at) > 90 
                AND weight < 0.3
                AND access_count < 2
            """)
            stale_memories = cursor.fetchone()[0]
            
            if stale_memories > 50:  # Only flag if there are many
                health["status"] = "issues_found"
                health["issues"].append(f"Found {stale_memories} old, low-weight, rarely accessed memories")
                health["recommendations"].append("Consider archiving or pruning old, low-value memories")
            
        return health
    
    except Exception as e:
        logging.error(f"Error analyzing memory health: {e}", exc_info=True)
        return {
            "status": "error",
            "issues": [f"Error analyzing memory health: {str(e)}"],
            "recommendations": ["Check database connection and integrity"]
        }
    

def get_knowledge_domains(db_path: str, min_memories: int = 3) -> List[str]:
    """Get knowledge domains with sufficient representation.
    
    Args:
        db_path (str): Path to the SQLite database
        min_memories (int): Minimum number of memories to consider a valid domain
        
    Returns:
        List[str]: List of knowledge domains
    """
    try:
        domains = []
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get memory types with sufficient representation
            cursor.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                HAVING count >= ?
                ORDER BY count DESC
            """, (min_memories,))
            
            memory_types = cursor.fetchall()
            
            # Add valid memory types to domains
            for memory_type, count in memory_types:
                if memory_type and memory_type.lower() not in ['general', 'important', 'autonomous_thought']:
                    domains.append(memory_type)
            
            # Extract domains from tags
            cursor.execute("""
                SELECT tags, COUNT(*) as count
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
                GROUP BY tags
                HAVING count >= ?
            """, (min_memories,))
            
            tag_groups = cursor.fetchall()
            
            # Process tags
            for tags_str, count in tag_groups:
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(',')]
                    for tag in tags:
                        if tag and tag.lower() not in ['general', 'important', 'autonomous', 'cognition', 'reflection']:
                            domains.append(tag)
        
        # Remove duplicates and sort
        return sorted(list(set(domains)))
    
    except Exception as e:
        logging.error(f"Error getting knowledge domains: {e}", exc_info=True)
        return []
