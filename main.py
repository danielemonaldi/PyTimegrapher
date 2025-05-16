"""
Timegrapher Application
Main entry point for the application.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from views.main_window import MainWindow

def main():
    """Main entry point for the application"""
    # Create PyQt application
    app = QApplication(sys.argv)
    
    # Set application-wide icon for taskbar
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/icons/icon.ico')
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()