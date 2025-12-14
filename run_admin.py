# run_admin.py
"""Run the DeepSeek Admin Dashboard."""

import subprocess
import os
import sys

def main():
    """Run the admin dashboard."""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to admin.py
        admin_path = os.path.join(script_dir, "admin.py")
        
        # Run the admin dashboard with Streamlit
        subprocess.run(["streamlit", "run", admin_path])
    
    except Exception as e:
        print(f"Error running admin dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()