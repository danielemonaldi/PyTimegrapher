"""
Application Controller
Responsible for coordinating between models and views.
Follows Single Responsibility Principle by focusing only on application control flow.
"""

from models.audio_processor import IAudioProcessor, AudioProcessor
from models.watch_analyzer import IWatchAnalyzer, WatchAnalyzer

class IAppController:
    """Interface for AppController following Interface Segregation Principle"""
    def start_recording(self):
        """Start recording and analyzing"""
        pass
        
    def stop_recording(self):
        """Stop recording and analyzing"""
        pass
        
    def reset(self):
        """Reset all data and analysis"""
        pass
        
    def update(self):
        """Update application state"""
        pass
        
    def apply_configuration(self, config):
        """Apply configuration settings"""
        pass

class AppController(IAppController):
    """
    Application controller implementation
    Follows Dependency Inversion Principle by depending on abstractions
    """
    def __init__(self, view_model):
        """
        Initialize the application controller
        
        Args:
            view_model: The view model to update with data
        """
        self.view_model = view_model
        self.analyzer = WatchAnalyzer()
        self.audio_processor = None
        self.recording = False
        self.start_time = None
        
        # Initialize audio processor with callback to analyzer
        try:
            self.audio_available = True
            self.audio_processor = AudioProcessor(
                callback=self.analyzer.process_audio_chunk
            )
        except Exception:
            self.audio_available = False
            
    def is_audio_available(self):
        """Check if audio is available"""
        return self.audio_available
        
    def start_recording(self):
        """Start recording and analyzing"""
        if not self.audio_available or self.recording:
            return False
            
        # Reset data
        self.analyzer.reset()
        self.recording = True
        
        # Start audio processor
        if self.audio_processor:
            self.audio_processor.start()
            return True
        
        return False
        
    def stop_recording(self):
        """Stop recording and analyzing"""
        if self.audio_processor:
            self.audio_processor.stop()
        
        self.recording = False
        return True
        
    def reset(self):
        """Reset all data and analysis"""
        # Stop recording if in progress
        if self.recording:
            self.stop_recording()
        
        # Reset analyzer
        self.analyzer.reset()
        
        return True
        
    def update(self):
        """Update application state and pass data to view model"""
        # Only update if recording and new data is available
        if self.recording and self.analyzer.has_new_data():
            # Get latest results from analyzer
            results = self.analyzer.get_latest_results()
            
            # Update view model with results
            self.view_model.update_results(results)
            
            return True
            
        return False
        
    def apply_configuration(self, config):
        """
        Apply configuration settings
        
        Args:
            config: Dictionary containing configuration settings
                - auto_detect_bph: Whether to automatically detect BPH
                - expected_bph: Expected BPH value if not auto-detecting
                - lift_angle: Lift angle in degrees
        """
        # Update analyzer configuration
        if 'auto_detect_bph' in config:
            self.analyzer.auto_detect_bph = config['auto_detect_bph']
            
        if 'expected_bph' in config:
            self.analyzer.expected_bph = config['expected_bph']
            
        if 'lift_angle' in config:
            self.analyzer.lift_angle = config['lift_angle']
            
        return True