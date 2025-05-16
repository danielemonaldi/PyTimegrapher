"""
Main Window View
Responsible for rendering the UI and capturing user input.
Follows Single Responsibility Principle by focusing only on UI presentation.
"""

import os
import time
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QLabel, QSlider, QRadioButton, QPushButton, QComboBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont
import pyqtgraph as pg

from views.view_model import IViewModel, ViewModel
from controllers.app_controller import IAppController, AppController

class MainWindow(QMainWindow):
    """
    Main application window
    Follows Single Responsibility Principle by focusing only on UI presentation
    """
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Initialize view model and controller
        self.view_model = ViewModel()
        self.controller = AppController(self.view_model)
        
        # Register for view model updates
        self.view_model.register_callback(self.update_ui_from_model)
        
        # Set up window properties
        self.setWindowTitle('PyTimegrapher')
        self.setGeometry(100, 100, 1400, 800)
        
        # Data for plotting
        self.data = {'timestamp': [], 'daily_rate': []}
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/icons/icon.ico')
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
        
        # Set up UI components
        self.setup_ui()
        
        # Timer to update the interface
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_timer_tick)
        self.update_timer.start(100)  # Update every 100ms
        
        # Start time tracking
        self.start_time = None
        self.elapsed_time = 0

    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel layout for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        # BPH Mode (Auto-detect or Manual)
        config_layout.addWidget(QLabel("BPH Mode:"), 0, 0)
        mode_layout = QHBoxLayout()
        
        self.auto_detect_radio = QRadioButton("Auto-detect")
        self.auto_detect_radio.setChecked(True)
        self.auto_detect_radio.toggled.connect(self.toggle_bph_mode)
        
        self.manual_bph_radio = QRadioButton("Preset Selection")
        self.manual_bph_radio.toggled.connect(self.toggle_bph_mode)
        
        mode_layout.addWidget(self.auto_detect_radio)
        mode_layout.addWidget(self.manual_bph_radio)
        config_layout.addLayout(mode_layout, 0, 1)
        
        # BPH Presets
        config_layout.addWidget(QLabel("BPH Presets:"), 1, 0)
        self.bph_combo = QComboBox()
        
        # Add all preset values
        bph_presets = [
            ("12,000 BPH", 12000),
            ("14,400 BPH", 14400),
            ("18,000 BPH", 18000),
            ("19,800 BPH", 19800),
            ("21,600 BPH", 21600),
            ("25,200 BPH", 25200),
            ("28,800 BPH", 28800),
            ("36,000 BPH", 36000),
            ("43,200 BPH", 43200),
            ("72,000 BPH", 72000)
        ]
        
        for label, value in bph_presets:
            self.bph_combo.addItem(label, value)
            
        # Set default BPH
        self.bph_combo.setCurrentIndex(self.bph_combo.findData(28800))
        self.bph_combo.currentIndexChanged.connect(self.set_bph_from_combo)
        self.bph_combo.setEnabled(False)  # Disabled by default (auto-detect mode)
        config_layout.addWidget(self.bph_combo, 1, 1)
        
        # Lift Angle
        config_layout.addWidget(QLabel("Lift Angle (°):"), 2, 0)
        self.lift_angle_slider = QSlider(Qt.Horizontal)
        self.lift_angle_slider.setRange(40, 60)
        self.lift_angle_slider.setValue(52)
        self.lift_angle_slider.setTickPosition(QSlider.TicksBelow)
        self.lift_angle_slider.setTickInterval(1)
        self.lift_angle_value = QLabel("52°")
        self.lift_angle_slider.valueChanged.connect(self.update_lift_angle)
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(self.lift_angle_slider)
        angle_layout.addWidget(self.lift_angle_value)
        config_layout.addLayout(angle_layout, 2, 1)
        
        # Apply configuration
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.clicked.connect(self.apply_configuration)
        config_layout.addWidget(apply_btn, 3, 1)
        
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)
        
        # Recording Controls
        control_group = QGroupBox("Recording Controls")
        control_layout = QHBoxLayout()
        
        # Create buttons with icon support
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.setEnabled(self.controller.is_audio_available())
        self.start_btn.clicked.connect(self.start_recording)
        # Set icon placeholder for start button
        start_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/buttons/start.png')
        if os.path.exists(start_icon_path):
            self.start_btn.setIcon(QIcon(start_icon_path))
        
        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        # Set icon placeholder for stop button
        stop_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/buttons/stop.png')
        if os.path.exists(stop_icon_path):
            self.stop_btn.setIcon(QIcon(stop_icon_path))
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_app)
        # Set icon placeholder for reset button
        reset_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/buttons/reset.png')
        if os.path.exists(reset_icon_path):
            self.reset_btn.setIcon(QIcon(reset_icon_path))
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.reset_btn)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # Results Group
        results_group = QGroupBox("Measurement Results")
        results_layout = QGridLayout()
        
        # Result labels
        results_layout.addWidget(QLabel("BPH:"), 0, 0)
        self.bph_label = QLabel("—")
        self.bph_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.bph_label, 0, 1)
        
        results_layout.addWidget(QLabel("Daily Rate:"), 1, 0)
        self.daily_rate_label = QLabel("—")
        self.daily_rate_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.daily_rate_label, 1, 1)
        
        results_layout.addWidget(QLabel("Beat Error:"), 2, 0)
        self.beat_error_label = QLabel("—")
        self.beat_error_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.beat_error_label, 2, 1)
        
        results_layout.addWidget(QLabel("Amplitude:"), 3, 0)
        self.amplitude_label = QLabel("—")
        self.amplitude_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.amplitude_label, 3, 1)
        
        results_layout.addWidget(QLabel("Elapsed Time:"), 4, 0)
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setFont(QFont("Arial", 12))
        results_layout.addWidget(self.elapsed_label, 4, 1)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready to start")
        self.status_label.setFont(QFont("Arial", 11))
        status_layout.addWidget(self.status_label)
        
        if not self.controller.is_audio_available():
            self.audio_error_label = QLabel("WARNING: Microphone not available. This application requires microphone access. Run it on your local computer.")
            self.audio_error_label.setStyleSheet("color: red; font-weight: bold;")
            status_layout.addWidget(self.audio_error_label)
        
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        instructions = """
        1. Place your watch close to your computer's microphone
        2. Ensure your environment is quiet (no background noise)
        3. Select the correct BPH mode and lift angle for your watch
        4. Press 'Start Recording' to begin analysis
        5. Wait at least 10-15 seconds for accurate measurements
        """
        instructions_label = QLabel(instructions)
        instructions_layout.addWidget(instructions_label)
        instructions_group.setLayout(instructions_layout)
        left_layout.addWidget(instructions_group)
        
        # Set left layout
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(450)
        
        # Right layout for the graph
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Graph for daily rate deviation
        graph_label = QLabel("Daily Rate Deviation Chart")
        graph_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(graph_label)
        
        # Configure pyqtgraph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # white background
        self.plot_widget.setLabel('left', 'Daily Rate', units='s/day')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Add horizontal line at zero (perfect accuracy)
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('g', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(zero_line)
        
        # Initial Y axis range
        self.plot_widget.setYRange(-30, 30)
        
        # Curve for daily rate data
        self.rate_curve = self.plot_widget.plot([], [], pen=pg.mkPen('b', width=2), symbol='o', symbolSize=5)
        
        right_layout.addWidget(self.plot_widget)
        right_panel.setLayout(right_layout)
        
        # Assemble panels in main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def update_lift_angle(self):
        """Update lift angle value display"""
        value = self.lift_angle_slider.value()
        self.lift_angle_value.setText(f"{value}°")
    
    def toggle_bph_mode(self):
        """Toggle between auto-detect and manual BPH mode"""
        if self.auto_detect_radio.isChecked():
            self.bph_combo.setEnabled(False)
        else:
            self.bph_combo.setEnabled(True)
            self.set_bph_from_combo()
    
    def set_bph_from_combo(self):
        """Set BPH from combo box selection"""
        if self.manual_bph_radio.isChecked():
            selected_bph = self.bph_combo.currentData()
            if selected_bph:
                # BPH will be applied when apply_configuration is called
                pass
    
    def apply_configuration(self):
        """Apply configuration settings"""
        config = {
            'auto_detect_bph': self.auto_detect_radio.isChecked(),
            'lift_angle': self.lift_angle_slider.value()
        }
        
        # Add expected BPH if in manual mode
        if not self.auto_detect_radio.isChecked():
            config['expected_bph'] = self.bph_combo.currentData()
        
        # Apply configuration via controller
        self.controller.apply_configuration(config)
        
        # Update status
        bph_mode = "Auto-detect" if self.auto_detect_radio.isChecked() else f"Fixed {self.bph_combo.currentData()} BPH"
        self.status_label.setText(f"Configuration applied:\n- Lift Angle = {self.lift_angle_slider.value()}°\n- BPH Mode = {bph_mode}")
    
    def start_recording(self):
        """Start recording audio and analyzing"""
        if not self.controller.is_audio_available():
            self.status_label.setText("Error: Microphone not available")
            return
            
        # Reset data
        self.data = {'timestamp': [], 'daily_rate': []}
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_time = time.time()
        
        # Start recording via controller
        if self.controller.start_recording():
            self.view_model.set_recording_status(True)
            self.status_label.setText("Recording and analyzing in progress...")
    
    def stop_recording(self):
        """Stop recording audio and analyzing"""
        self.controller.stop_recording()
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.view_model.set_recording_status(False)
        
        self.status_label.setText("Recording stopped. Results displayed.")
    
    def reset_app(self):
        """Reset the application"""
        # Stop recording if in progress
        if self.view_model.recording:
            self.stop_recording()
        
        # Reset data
        self.data = {'timestamp': [], 'daily_rate': []}
        
        # Reset graph
        self.rate_curve.setData([], [])
        
        # Reset controller and view model
        self.controller.reset()
        self.view_model.reset()
        
        # Reset elapsed time
        self.start_time = None
        self.elapsed_time = 0
        self.elapsed_label.setText("00:00")
        
        self.status_label.setText("Reset completed. Ready to start.")
    
    def update_timer_tick(self):
        """Handle timer tick to update UI"""
        # Update controller to process new data
        self.controller.update()
        
        # Update elapsed time if recording
        if self.view_model.recording and self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            self.view_model.update_elapsed_time(self.elapsed_time)
    
    def update_ui_from_model(self):
        """Update UI elements from view model"""
        # Get formatted results from view model
        formatted = self.view_model.get_formatted_results()
        
        # Update labels
        if 'bph' in formatted:
            self.bph_label.setText(formatted['bph'])
            
        if 'daily_rate' in formatted:
            self.daily_rate_label.setText(formatted['daily_rate'])
            
        if 'beat_error' in formatted:
            self.beat_error_label.setText(formatted['beat_error'])
            
        if 'amplitude' in formatted:
            self.amplitude_label.setText(formatted['amplitude'])
            
        if 'elapsed_time' in formatted:
            self.elapsed_label.setText(formatted['elapsed_time'])
        
        # Update graph if we have a daily rate
        if self.view_model.daily_rate is not None:
            # Add point to graph
            self.data['timestamp'].append(self.elapsed_time)
            self.data['daily_rate'].append(self.view_model.daily_rate)
            
            # Update graph
            self.rate_curve.setData(self.data['timestamp'], self.data['daily_rate'])

def create_main_window():
    """Create and return the main application window"""
    return MainWindow()