import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import DearPyGui GUI (new version)
from gui.synthesizer_gui_dpg import main

# Old Pygame GUI is still available at gui.synthesizer_gui_vst if needed

if __name__ == "__main__":
    main()
