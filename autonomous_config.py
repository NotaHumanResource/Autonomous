# autonomous_config.py
"""Configuration settings for the Autonomous Cognition system."""

# Activity settings (probabilities of selection)
COGNITIVE_ACTIVITIES = {
		  
    "optimize_memory_organization": {
        "weight": 0.9,
        "min_interval_hours": 36,  # Every 1.5 days at most
        "description": "Organizes and optimizes stored memories"
    },
    "consolidate_similar_memories": {
        "weight": 0.8,
        "min_interval_hours": 24,  # Once per day at most
        "description": "Finds and consolidates similar memories"
    },
	"categorize_user_information": {
        "weight": 0.7,
        "min_interval_hours": 24,  # Once per day at most
        "description": "Categorizes user information for better personalization"
    },
    "analyze_knowledge_gaps": {
        "weight": 0.8,
        "min_interval_hours": 72,  # Every three days at most
        "description": "Identifies gaps in user knowledge"
    },
    "fill_knowledge_gaps": {
        "weight": 0.9,  # Higher weight, run after analysis
        "min_interval_hours": 96,  # Every four days at most
        "description": "Fills identified knowledge gaps about the user"
    }
}

# General cognition settings
COGNITION_SETTINGS = {
    "inactivity_threshold_seconds": 1200,  # 20 minutes of user inactivity before cognition activates
    "cognitive_cycle_interval_seconds": 600,  # 10 minutes between cognitive activities
    "max_thought_history": 50,  # Maximum number of thoughts to keep in memory
    "initial_startup_delay_seconds": 120  # Delay before first cognition after startup
}

# Database storage settings
STORAGE_SETTINGS = {
    "default_confidence": 0.5,  # Default confidence for autonomous thoughts
    "memory_type": "memory_management",  # Memory type for storage
    "base_tags": "autonomous,memory_management",  # Base tags for all autonomous thoughts
}