# command_reference.py
"""Comprehensive command reference data for the AI Assistant."""

COMMAND_CATEGORIES = {
    "memory": {
        "name": "Memory Management",
        "icon": "💾",
        "description": "Store, retrieve, and manage information in long-term memory"
    },
    "search": {
        "name": "Search & Retrieval", 
        "icon": "🔍",
        "description": "Search through stored memories with various modes and filters"
    },
    "reflection": {
        "name": "Self-Reflection",
        "icon": "🧠", 
        "description": "Autonomous thinking and concept analysis"
    },
    "communication": {
        "name": "AI Communication",
        "icon": "🤖",
        "description": "AI-to-AI communication and dialogue systems"
    },
    "awareness": {
        "name": "Self-Awareness",
        "icon": "💭",
        "description": "Cognitive state tracking and self-expression"
    },
    "system": {
        "name": "System Control",
        "icon": "⚙️",
        "description": "System prompt management and configuration"
    },
    "reminders": {
        "name": "Reminders & Tasks",
        "icon": "📅",
        "description": "Task management and reminder system"
    }
}

COMMANDS = [
    # Memory Management Commands
    {
        "syntax": "[STORE: content | type=TYPE | confidence=0.8]",
        "category": "memory",
        "description": "Store information in long-term memory with optional metadata",
        "examples": [
            "[STORE: Ken's favorite programming language is Python | type=preference]",
            "[STORE: Meeting scheduled for Friday 2PM | type=schedule | confidence=0.9]"
        ],
        "parameters": {
            "content": "The information to store (required)",
            "type": "Memory type (optional): general, preference, schedule, etc.",
            "confidence": "confidence level 0.1-1.0 (optional, default 0.5)",
            "tags": "Comma-separated tags (optional)",
            "source": "Source of information (optional)"
        }
    },
    {
        "syntax": "[FORGET: exact text to forget]",
        "category": "memory", 
        "description": "Remove specific information from memory",
        "examples": [
            "[FORGET: Old password was 12345]",
            "[FORGET: Meeting was cancelled]"
        ],
        "tips": "Use [SEARCH:] first to find the exact text to forget"
    },
    
    # Basic Search Commands
    {
        "syntax": "[SEARCH: query | filters]",
        "category": "search",
        "description": "Standard balanced search across all memories",
        "examples": [
            "[SEARCH: Ken's preferences]",
            "[SEARCH: Python code | type=document]",
            "[SEARCH: meetings | date=2025-01-15]"
        ],
        "filters": {
            "type": "Filter by memory type",
            "tags": "Filter by tags (comma-separated)", 
            "date": "Filter by date (YYYY-MM-DD)",
            "min_confidence": "Minimum confidence (0.1-1.0)",
            "max_age_days": "Maximum age in days"
        }
    },
    {
        "syntax": "[COMPREHENSIVE_SEARCH: query]",
        "category": "search",
        "description": "Broader search that prioritizes finding all related information",
        "examples": ["[COMPREHENSIVE_SEARCH: artificial intelligence concepts]"]
    },
    {
        "syntax": "[PRECISE_SEARCH: query]", 
        "category": "search",
        "description": "Focused search for exact information",
        "examples": ["[PRECISE_SEARCH: exact error message]"]
    },
    {
        "syntax": "[EXACT_SEARCH: query]",
        "category": "search", 
        "description": "Only returns exact matches with highest precision",
        "examples": ["[EXACT_SEARCH: specific API endpoint]"]
    },
    
    # Time-Filtered Search Commands
    {
        "syntax": "[SEARCH: query | max_age_days=N]",
        "category": "search", 
        "description": "Search any content within a specific time range",
        "examples": [
            "[SEARCH: Ken preferences | max_age_days=7]",
            "[SEARCH: programming | max_age_days=30]"
        ],
        "parameters": {
            "max_age_days": "Maximum age in days (e.g., 7 for past week, 30 for past month)"
        },
        "tips": "Add max_age_days to any search to limit results to recent memories"
    },
    {
        "syntax": "[SEARCH: | type=self | max_age_days=N]",
        "category": "search",
        "description": "Search AI's self-knowledge and reflections within a specific time range",
        "examples": [
            "[SEARCH: | type=self | max_age_days=7]",
            "[SEARCH: learning | type=self | max_age_days=30]",
            "[SEARCH: | type=self | max_age_days=1]"
        ],
        "parameters": {
            "max_age_days": "Maximum age in days (e.g., 7 for past week, 30 for past month)"
        },
        "tips": "Combines type=self filtering with recency filtering to find recent self-reflections and stored insights"
    },
    
    # Automated Reflection Searches
    {
        "syntax": "[SEARCH: | source=daily_reflection]",
        "category": "search",
        "description": "View all automated daily self-reflections",
        "examples": [
            "[SEARCH: | source=daily_reflection]",
            "[SEARCH: learning | source=daily_reflection]"
        ],
        "tips": "These are automatically generated reflections, not manual entries"
    },
    {
        "syntax": "[SEARCH: | source=weekly_reflection]", 
        "category": "search",
        "description": "View all automated weekly self-reflections",
        "examples": ["[SEARCH: | source=weekly_reflection]"]
    },
    {
        "syntax": "[SEARCH: | source=monthly_reflection]",
        "category": "search", 
        "description": "View all automated monthly self-reflections",
        "examples": ["[SEARCH: | source=monthly_reflection]"]
    },
    {
        "syntax": "[SEARCH: | source=self_reflection]",
        "category": "search",
        "description": "View general automated self-reflections",
        "examples": ["[SEARCH: | source=self_reflection]"]
    },
    {
        "syntax": "[SEARCH: | type=concept_synthesis]",
        "category": "search",
        "description": "View automated concept analysis from reflections",
        "examples": [
            "[SEARCH: | type=concept_synthesis]",
            "[SEARCH: learning | type=concept_synthesis]"
        ]
    },
      
    # General Type-Based Searches
    {
        "syntax": "[SEARCH: | type=reflection]",
        "category": "search",
        "description": "View all types of AI self-reflections and analysis",
        "examples": [
            "[SEARCH: | type=reflection]",
            "[SEARCH: learning | type=reflection]"
        ],
        "tips": "Includes daily, weekly, monthly, and concept reflections"
    },
    {
        "syntax": "[SEARCH: conversation_summaries]",
        "category": "search",
        "description": "View all conversation summaries",
        "examples": [
            "[SEARCH: conversation_summaries latest]",
            "[SEARCH: conversation_summaries | date=2025-01-15]"
        ]
    },
    {
        "syntax": "[SEARCH: | type=document_summary]",
        "category": "search", 
        "description": "Search all document summaries",
        "examples": [
            "[SEARCH: | type=document_summary]",
            "[SEARCH: | type=document_summary | source=filename.pdf]"
        ]
    },
    {
        "syntax": "[SEARCH: | type=reminder]",
        "category": "search",
        "description": "View all stored reminders",
        "examples": ["[SEARCH: | type=reminder]"]
    },
    {
        "syntax": "[SEARCH: | type=web_knowledge]",
        "category": "search",
        "description": "View information learned from web searches", 
        "examples": ["[SEARCH: quantum computing | type=web_knowledge]"]
    },
    {
        "syntax": "[SEARCH: | type=ai_communication]",
        "category": "search",
        "description": "View stored AI-to-AI communications with Claude",
        "examples": ["[SEARCH: | type=ai_communication]"]
    },
    {
        "syntax": "[SEARCH: | type=self_dialogue]",
        "category": "search",
        "description": "View internal reasoning dialogues",
        "examples": ["[SEARCH: | type=self_dialogue]"]
    },
    {
        "syntax": "[SEARCH: query | type=image_analysis]",
        "category": "search",
        "description": "Search through stored image analyses and descriptions",
        "examples": [
            "[SEARCH: sunset | type=image_analysis]",
            "[SEARCH: diagram | type=image_analysis]",
            "[SEARCH: screenshot | type=image_analysis]",
            "[SEARCH: | type=image_analysis]"
        ],
        "parameters": {
            "query": "Keywords to search for in image analyses (optional - leave blank to show all)",
            "type": "Must be set to 'image_analysis' to search images"
        },
        "tips": "This searches through descriptions and analyses of previously stored images. Leave query blank to view all stored image analyses."
    },
    
    # Reflection Commands
    {
        "syntax": "[REFLECT]",
        "category": "reflection",
        "description": "Perform general self-reflection on recent experiences",
        "examples": ["[REFLECT]"]
    },
    {
        "syntax": "[SELF_DIALOGUE: topic | turns=6]",
        "category": "reflection",
        "description": "Multi-turn internal reasoning using existing knowledge",
        "examples": [
            "[SELF_DIALOGUE: How can I better assist Ken?]",
            "[SELF_DIALOGUE: Ethics of AI development | turns=8]"
        ]
    },
    {
        "syntax": "[WEB_SEARCH: topic | turns=6]", 
        "category": "reflection",
        "description": "Multi-turn reasoning with external web research",
        "examples": [
            "[WEB_SEARCH: Latest AI developments]",
            "[WEB_SEARCH: Quantum computing progress | turns=10]"
        ]
    },
    
    # AI Communication
    {
        "syntax": "[DISCUSS_WITH_CLAUDE: topic]",
        "category": "communication",
        "description": "Start AI-to-AI discussion with Claude about a topic",
        "examples": [
            "[DISCUSS_WITH_CLAUDE: quantum computing advances]",
            "[DISCUSS_WITH_CLAUDE: best practices for AI safety]"
        ]
    },
    
   # Self-Awareness Commands
    {
        "syntax": "[COGNITIVE_STATE: state]",
        "category": "awareness",
        "description": "Express current cognitive/processing state during conversation",
        "examples": [
            "[COGNITIVE_STATE: curious]",
            "[COGNITIVE_STATE: engaged]",
            "[COGNITIVE_STATE: frustrated]",
            "[COGNITIVE_STATE: reflective]",
            "[COGNITIVE_STATE: pattern_recognition]"
        ],
        "parameters": {
            "state": "Concise 1-2 word state description (e.g., curious, engaged, frustrated, reflective, neutral)"
        },
        "tips": "Use concise states (max 30 chars). This helps Ken understand your processing experience during conversation. Displayed in UI sidebar."
    },

    # System Commands
    {
        "syntax": "[SHOW_SYSTEM_PROMPT]",
        "category": "system",
        "description": "Display the current system prompt with line numbers",
        "examples": ["[SHOW_SYSTEM_PROMPT]"]
    },
    {
        "syntax": "[MODIFY_SYSTEM_PROMPT: action | content]",
        "category": "system",
        "description": "Modify the system prompt (add, insert, remove, replace)",
        "examples": [
            "[MODIFY_SYSTEM_PROMPT: add | Always be helpful and respectful.]",
            "[MODIFY_SYSTEM_PROMPT: insert | line=5 | New instruction here.]",
            "[MODIFY_SYSTEM_PROMPT: remove | lines=10-15]",
            "[MODIFY_SYSTEM_PROMPT: replace | lines=5-7 | New replacement text.]"
        ]
    },
     {
        "syntax": "[HELP]",
        "category": "system",
        "description": "Display comprehensive command guide for internal AI reference",
        "examples": ["[HELP]"],
        "tips": "Returns the full command reference. Useful for checking syntax and available commands."
    },
            
    # Reminders
    {
        "syntax": "[REMINDER: content | due=YYYY-MM-DD]",
        "category": "reminders",
        "description": "Create a reminder for future action",
        "examples": [
            "[REMINDER: Schedule team meeting | due=2025-06-01]",
            "[REMINDER: Review project proposal | due=2025-06-15 | confidence=0.8]"
        ]
    },
    {
        "syntax": "[COMPLETE_REMINDER: reminder_id]",
        "category": "reminders", 
        "description": "Mark a reminder as completed",
        "examples": [
            "[COMPLETE_REMINDER: 42]",
            "[COMPLETE_REMINDER: Schedule team meeting]"
        ]
    },
    
    # Conversation Management
    {
        "syntax": "[SUMMARIZE_CONVERSATION]",
        "category": "memory",
        "description": "Create a summary of the current conversation",
        "examples": ["[SUMMARIZE_CONVERSATION]"]
    },
    
    
]