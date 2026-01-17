import sys
import os

# Add the UmeTrack directory to sys.path so that 'lib' can be imported as a top-level module
_umetrack_dir = os.path.dirname(os.path.abspath(__file__))
if _umetrack_dir not in sys.path:
    sys.path.insert(0, _umetrack_dir)
