"""
View Model
Responsible for managing the state of the view and communicating with the controller.
Follows Single Responsibility Principle by focusing only on view state management.
"""

from typing import Dict, Any, Callable, List, Optional

class IViewModel:
    """Interface for ViewModel following Interface Segregation Principle"""
    def update_results(self, results: Dict[str, Any]) -> None:
        """Update results in view model"""
        pass
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for view updates"""
        pass
        
    def notify_views(self) -> None:
        """Notify views of state changes"""
        pass

class ViewModel(IViewModel):
    """
    View model implementation
    Implements Observer pattern for notifying views of changes
    """
    def __init__(self):
        """Initialize the view model"""
        # State data
        self.bph = None  # Beats per hour
        self.daily_rate = None  # Daily rate in seconds per day
        self.beat_error = None  # Beat error in milliseconds
        self.amplitude = None  # Amplitude in degrees
        self.elapsed_time = 0  # Elapsed time in seconds
        self.diagnostic_data = {}  # Diagnostic data
        self.recording = False  # Recording status
        
        # Callbacks for view update notifications
        self.callbacks = []
        
    def update_results(self, results: Dict[str, Any]) -> None:
        """
        Update results in view model
        
        Args:
            results: Dictionary containing watch analysis results
        """
        # Update state with new results
        if 'bph' in results and results['bph'] is not None:
            self.bph = results['bph']
            
        if 'daily_rate' in results and results['daily_rate'] is not None:
            self.daily_rate = results['daily_rate']
            
        if 'beat_error' in results and results['beat_error'] is not None:
            self.beat_error = results['beat_error']
            
        if 'amplitude' in results and results['amplitude'] is not None:
            self.amplitude = results['amplitude']
            
        if 'diagnostic' in results:
            self.diagnostic_data = results['diagnostic']
            
        # Notify views of state change
        self.notify_views()
        
    def update_elapsed_time(self, elapsed: float) -> None:
        """
        Update elapsed time
        
        Args:
            elapsed: Elapsed time in seconds
        """
        self.elapsed_time = elapsed
        self.notify_views()
        
    def set_recording_status(self, recording: bool) -> None:
        """
        Set recording status
        
        Args:
            recording: True if recording, False otherwise
        """
        self.recording = recording
        self.notify_views()
        
    def reset(self) -> None:
        """Reset view model state"""
        self.bph = None
        self.daily_rate = None
        self.beat_error = None
        self.amplitude = None
        self.elapsed_time = 0
        self.diagnostic_data = {}
        self.notify_views()
        
    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for view updates
        
        Args:
            callback: Function to call when view should update
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            
    def unregister_callback(self, callback: Callable) -> None:
        """
        Unregister callback for view updates
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def notify_views(self) -> None:
        """Notify views of state changes"""
        for callback in self.callbacks:
            callback()
            
    def get_formatted_results(self) -> Dict[str, str]:
        """
        Get formatted results for display
        
        Returns:
            Dictionary of formatted result strings
        """
        results = {}
        
        # Format BPH
        if self.bph is not None:
            results['bph'] = f"{int(self.bph)} BPH"
        else:
            results['bph'] = "—"
            
        # Format daily rate
        if self.daily_rate is not None:
            results['daily_rate'] = f"{self.daily_rate:.1f} s/day"
        else:
            results['daily_rate'] = "—"
            
        # Format beat error
        if self.beat_error is not None:
            results['beat_error'] = f"{self.beat_error:.1f} ms"
        else:
            results['beat_error'] = "—"
            
        # Format amplitude
        if self.amplitude is not None:
            results['amplitude'] = f"{self.amplitude:.1f}°"
        else:
            results['amplitude'] = "—"
            
        # Format elapsed time
        minutes = int(self.elapsed_time // 60)
        seconds = int(self.elapsed_time % 60)
        results['elapsed_time'] = f"{minutes:02d}:{seconds:02d}"
        
        return results
        
    def get_plot_data(self) -> Dict[str, List[float]]:
        """
        Get data for plotting
        
        Returns:
            Dictionary with timestamp and daily_rate lists
        """
        # In a real implementation, this would store and return
        # the time series data for plotting
        return {
            'timestamp': [],
            'daily_rate': []
        }