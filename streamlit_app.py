import sys
import os
from pathlib import Path

# Streamlit works best when the main script is in the root or handles its own pathing.
# Adding 'src' to sys.path ensures that 'from engine import ...' in ui.py works correctly.
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Using a wrapper to run the existing ui.py logic
if __name__ == "__main__":
    # Import and run the UI
    import ui
