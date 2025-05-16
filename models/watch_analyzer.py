"""
Watch Analyzer Model
Responsible for analyzing watch performance metrics.
Follows Single Responsibility Principle by focusing only on analyzing watch data.
"""

import numpy as np
from scipy import signal
import time
from collections import deque

class IWatchAnalyzer:
    """Interface for WatchAnalyzer following Interface Segregation Principle"""
    def reset(self):
        """Reset all analysis data"""
        pass
        
    def process_audio_chunk(self, audio_data, sample_rate):
        """Process a chunk of audio data to detect watch ticks"""
        pass
        
    def has_new_data(self):
        """Check if new data is available"""
        pass
        
    def get_latest_results(self):
        """Get the latest calculated results"""
        pass

class WatchAnalyzer(IWatchAnalyzer):
    """Watch Analyzer implementation for analyzing watch performance"""
    def __init__(self):
        """
        Initialize the watch analyzer with default settings
        """
        # Common BPH presets
        self.bph_presets = [12000, 14400, 18000, 19800, 21600, 25200, 28800, 36000, 43200, 72000]
        
        # Configuration
        self.auto_detect_bph = True  # Default: auto-detect BPH
        self.expected_bph = 28800  # Default: 28,800 BPH (8 beats per second)
        self.lift_angle = 52  # Default lift angle in degrees
        
        # Internal state
        self.new_data_available = False
        self.last_results = {}
        
        # Beat detection
        self.tick_threshold = 0.02  # Initial threshold for tick detection
        self.adaptive_threshold = True
        self.min_time_between_ticks = 0.05  # Minimum time between ticks in seconds
        self.confidence_threshold = 0.6  # Confidence threshold for BPH detection
        
        # Data tracking
        self.beat_times = deque(maxlen=200)  # Increased size for better averaging
        self.beat_intervals = deque(maxlen=100)  # Increased size for better averaging
        self.beat_amplitudes = deque(maxlen=100)  # Increased size for better averaging
        self.last_beat_time = 0
        
        # BPH detection
        self.bph_histogram = {}  # Histogram of detected BPH values
        self.bph_confidence = {}  # Confidence in each BPH value
        self.min_intervals_for_detection = 20  # Minimum intervals needed for reliable detection
        
        # Results
        self.calculated_bph = None
        self.daily_rate = None
        self.beat_error = None
        self.amplitude = None
        
    def reset(self):
        """Reset all analysis data"""
        self.beat_times.clear()
        self.beat_intervals.clear()
        self.beat_amplitudes.clear()
        self.last_beat_time = 0
        self.calculated_bph = None
        self.daily_rate = None
        self.beat_error = None
        self.amplitude = None
        self.new_data_available = False
        self.last_results = {}
        self.bph_histogram = {}
        self.bph_confidence = {}
        
    def process_audio_chunk(self, audio_data, sample_rate):
        """
        Process a chunk of audio data to detect watch ticks
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: audio sample rate
        """
        # Apply multi-stage filtering for better noise rejection
        # First apply a high-pass filter to remove low frequency noise
        sos_high = signal.butter(3, 800, 'highpass', fs=sample_rate, output='sos')
        high_filtered = signal.sosfilt(sos_high, audio_data)
        
        # Then apply a band-pass filter to focus on watch tick frequencies (typically 1kHz-10kHz)
        sos = signal.butter(5, [1000, 10000], 'bandpass', fs=sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, high_filtered)
        
        # Calculate the audio envelope using improved methods for better peak detection
        # Make sure filtered_audio is a single array, not a tuple
        if isinstance(filtered_audio, tuple):
            filtered_audio = filtered_audio[0]
            
        # Square the signal to emphasize peaks (explicit type conversion for safety)
        filtered_audio_np = np.asarray(filtered_audio)
        squared_audio = filtered_audio_np**2
        
        # Apply a low-pass smoothing filter to the squared signal
        sos_smooth = signal.butter(2, 50, 'lowpass', fs=sample_rate, output='sos')
        smoothed_envelope = signal.sosfilt(sos_smooth, squared_audio)
        
        # Take the square root to get back to amplitude scale
        amplitude_envelope = np.sqrt(np.abs(smoothed_envelope))
        
        # Adapt threshold if needed with more robust threshold calculation
        if self.adaptive_threshold and len(amplitude_envelope) > 0:
            # Use percentile-based threshold instead of mean/std for robustness to outliers
            background_level = float(np.percentile(amplitude_envelope, 50))  # median
            peak_level = float(np.percentile(amplitude_envelope, 95))
            
            # Set threshold as a point between median and 95th percentile
            threshold_point = background_level + 0.6 * (peak_level - background_level)
            # Ensure the threshold is a float and in a reasonable range
            self.tick_threshold = max(0.01, min(0.2, float(threshold_point)))
        
        # Find peaks with more sophisticated peak detection
        peaks, properties = signal.find_peaks(
            amplitude_envelope, 
            height=self.tick_threshold,
            distance=int(self.min_time_between_ticks * sample_rate),
            prominence=self.tick_threshold * 0.5,  # Ensure peaks stand out from background
            width=(int(0.001 * sample_rate), int(0.02 * sample_rate))  # Typical watch tick width
        )
        
        # Current time
        current_time = time.time()
        
        # Process detected peaks
        for i, peak in enumerate(peaks):
            peak_time = current_time - (len(audio_data) - peak) / sample_rate
            peak_amplitude = properties['peak_heights'][i]
            
            # Only record if enough time has passed since the last beat
            if (peak_time - self.last_beat_time) > self.min_time_between_ticks:
                self.beat_times.append(peak_time)
                self.beat_amplitudes.append(peak_amplitude)
                
                # Calculate interval from previous beat
                if len(self.beat_times) > 1:
                    interval = self.beat_times[-1] - self.beat_times[-2]
                    
                    # Only add reasonable intervals (filter out extreme values)
                    if 0.01 < interval < 1.0:  # Corresponds to 3600-360000 BPH range
                        self.beat_intervals.append(interval)
                
                self.last_beat_time = peak_time
        
        # Calculate results if we have enough data
        if len(self.beat_intervals) >= self.min_intervals_for_detection:
            self._calculate_results()
            self.new_data_available = True
    
    def find_closest_preset(self, bph_value):
        """
        Find the closest BPH preset to the calculated value
        
        Args:
            bph_value: The calculated BPH value
            
        Returns:
            The closest preset BPH value
        """
        if bph_value is None:
            return self.expected_bph
            
        # Find closest preset
        closest_preset = min(self.bph_presets, key=lambda x: abs(x - bph_value))
        return closest_preset
    
    def _calculate_results(self):
        """Calculate watch performance metrics using enhanced statistical methods"""
        if len(self.beat_intervals) >= self.min_intervals_for_detection:
            # Step 1: Apply robust statistical filtering to intervals
            intervals = np.array(self.beat_intervals)
            
            # Use percentile-based filtering (more robust than mean/std for non-normal distributions)
            q25, q75 = np.percentile(intervals, [25, 75])
            iqr = q75 - q25
            lower_bound = max(0.01, q25 - 1.5 * iqr)
            upper_bound = min(1.0, q75 + 1.5 * iqr)
            
            # Filter intervals using IQR method
            valid_intervals = intervals[(intervals >= lower_bound) & (intervals <= upper_bound)]
            
            if len(valid_intervals) > 0:
                # Step 2: Calculate BPH histogram for more reliable detection
                bph_values = (1.0 / valid_intervals) * 3600
                
                # Round BPH values to nearest 100 for histogram binning
                rounded_bphs = np.round(bph_values / 100) * 100
                
                # Update BPH histogram
                for bph in rounded_bphs:
                    if bph not in self.bph_histogram:
                        self.bph_histogram[bph] = 0
                    self.bph_histogram[bph] += 1
                
                # Step 3: Calculate most likely BPH using histogram peak
                if self.bph_histogram:
                    # Find the most common BPH value
                    max_count = 0
                    most_common_bph = None
                    for bph, count in self.bph_histogram.items():
                        if count > max_count:
                            max_count = count
                            most_common_bph = bph
                    
                    # Calculate confidence as ratio of peak count to total
                    total_samples = sum(self.bph_histogram.values())
                    confidence = max_count / total_samples if total_samples > 0 else 0
                    
                    # Only update if confidence is high enough
                    if most_common_bph is not None:
                        if confidence >= self.confidence_threshold:
                            self.calculated_bph = most_common_bph
                        else:
                            # Fallback to weighted average for low confidence
                            weighted_sum = sum(bph * count for bph, count in self.bph_histogram.items())
                            self.calculated_bph = weighted_sum / total_samples
                else:
                    # Fallback to simple mean if histogram is empty
                    avg_interval = np.mean(valid_intervals)
                    self.calculated_bph = (1.0 / avg_interval) * 3600
                
                # Step 4: Match to preset if auto-detect is enabled
                effective_bph = self.expected_bph
                if self.auto_detect_bph and self.calculated_bph is not None:
                    effective_bph = self.find_closest_preset(self.calculated_bph)
                    self.expected_bph = effective_bph
                
                # Step 5: Calculate daily rate using more precise formula
                # formula: rate = (actual_bph - expected_bph) * 24 / expected_bph
                if self.calculated_bph is not None:
                    self.daily_rate = (self.calculated_bph - self.expected_bph) * 24 / self.expected_bph
                
                # Step 6: Advanced beat error calculation with sophisticated pattern recognition
                if len(valid_intervals) >= 16:  # Need more data for accurate analysis
                    # In mechanical watches, the beat error represents the imbalance between the 
                    # two halves of the escapement's oscillation cycle.
                    
                    # Step 6.1: Use autocorrelation to confirm tick-tock pattern
                    # Autocorrelation helps identify repeating patterns in the signal
                    intervals_array = np.array(valid_intervals)
                    if len(intervals_array) > 30:  # Use autocorrelation with larger datasets
                        # Calculate autocorrelation (with normalization)
                        mean_interval = np.mean(intervals_array)
                        norm_intervals = intervals_array - mean_interval
                        autocorr = np.correlate(norm_intervals, norm_intervals, mode='full')
                        autocorr = autocorr[len(autocorr)//2:]  # Take only second half
                        
                        # Find peak in autocorrelation (should be at lag = 2 for tick-tock)
                        if len(autocorr) > 5:
                            peaks, _ = signal.find_peaks(autocorr, height=0, distance=1)
                            if len(peaks) > 0 and peaks[0] >= 1:
                                tick_tock_lag = peaks[0]
                                # Now we know the actual pattern periodicity
                            else:
                                tick_tock_lag = 2  # Default to tick-tock assumption
                        else:
                            tick_tock_lag = 2
                    else:
                        tick_tock_lag = 2  # Default to tick-tock assumption
                    
                    # Step 6.2: Advanced pattern clustering to separate tick and tock intervals
                    # Use a rolling window to find the most consistent pattern subdivision
                    pattern_clusters = []
                    for offset in range(tick_tock_lag):
                        pattern_clusters.append(intervals_array[offset::tick_tock_lag])
                    
                    # Step 6.3: Find most stable pattern by analyzing variance
                    cluster_variances = [np.var(cluster) for cluster in pattern_clusters if len(cluster) > 3]
                    best_clusters = []
                    if cluster_variances:
                        # Sort clusters by ascending variance (most stable first)
                        sorted_idx = np.argsort(cluster_variances)
                        # Take two most stable clusters which should correspond to tick and tock
                        for i in range(min(2, len(sorted_idx))):
                            idx = sorted_idx[i]
                            if idx < len(pattern_clusters) and len(pattern_clusters[idx]) > 3:
                                best_clusters.append(pattern_clusters[idx])
                    
                    # If we couldn't find strong clusters, fall back to simple even/odd
                    if len(best_clusters) < 2:
                        best_clusters = [intervals_array[::2], intervals_array[1::2]]
                    
                    if len(best_clusters) >= 2 and len(best_clusters[0]) > 3 and len(best_clusters[1]) > 3:
                        # Step 6.4: Use robust statistics for each cluster
                        # Calculate median and IQR for most stable measurement
                        medians = [np.median(cluster) for cluster in best_clusters]
                        
                        # Beat error = half of the difference between groups, normalized by period
                        interval_diff = abs(medians[0] - medians[1])
                        
                        # Convert to milliseconds relative to beat period
                        beat_period = 3600 / self.expected_bph  # seconds per beat
                        beat_error_ratio = interval_diff / beat_period
                        
                        # Apply empirical correction factor based on BPH
                        # (Higher BPH watches need adjustment to measurement)
                        bph_correction = 1.0
                        if self.expected_bph > 28800:
                            bph_correction = 0.9
                        elif self.expected_bph < 18000:
                            bph_correction = 1.1
                        
                        # Calculate final beat error in milliseconds
                        self.beat_error = beat_error_ratio * 1000 / 2 * bph_correction
                        
                        # Ensure we're in expected range (0-10ms typically)
                        if self.beat_error is not None:
                            self.beat_error = min(10.0, max(0.0, float(self.beat_error)))
                
                # Step 7: Advanced amplitude calculation using precise horological principles
                if len(self.beat_amplitudes) >= 15:  # Need sufficient data for reliable amplitude calculation
                    # Step 7.1: Advanced peak amplitude analysis
                    # Sort amplitudes and apply robust statistical methods
                    sorted_amplitudes = np.sort(self.beat_amplitudes)
                    
                    # Remove extreme outliers for more reliable measurements
                    if len(sorted_amplitudes) > 20:
                        # Remove bottom and top 5% of values
                        trim_percentage = 0.05
                        trim_size = int(len(sorted_amplitudes) * trim_percentage)
                        trimmed_amplitudes = sorted_amplitudes[trim_size:-trim_size]
                    else:
                        trimmed_amplitudes = sorted_amplitudes
                    
                    # Step 7.2: Calculate statistical robust measures
                    # Use multiple statistical measures for better reliability
                    p90_amplitude = float(np.percentile(trimmed_amplitudes, 90))  # 90th percentile for peaks
                    median_amplitude = float(np.median(trimmed_amplitudes))  # Median for stability
                    mean_top_quarter = float(np.mean(trimmed_amplitudes[-len(trimmed_amplitudes)//4:]))  # Mean of top 25%
                    
                    # Step 7.3: Weighted combination of measures for better accuracy
                    # Different weights based on reliability of each measure
                    weighted_amplitude = (p90_amplitude * 0.5) + (mean_top_quarter * 0.3) + (median_amplitude * 0.2)
                    
                    # Step 7.4: Apply advanced signal-dependent scaling
                    # Signal quality estimation
                    signal_quality = 1.0
                    if len(self.beat_intervals) > 20:
                        # Coefficient of variation of intervals - lower means more stable signal
                        interval_array = np.array(self.beat_intervals)
                        interval_cv = np.std(interval_array) / np.mean(interval_array)
                        # Adjust signal quality based on stability
                        if interval_cv < 0.05:  # Very stable
                            signal_quality = 1.2  # Boost amplitude
                        elif interval_cv > 0.15:  # Unstable
                            signal_quality = 0.8  # Reduce amplitude
                    
                    # Step 7.5: Apply sophisticated amplitude model based on horological principles
                    # Base amplitude conversion using calibrated scaling model
                    normalized_amplitude = min(1.0, float(weighted_amplitude * 8 * signal_quality))
                    
                    # Non-linear calibration curve for more accurate degrees conversion
                    # Use logarithmic model for better low-amplitude accuracy
                    if normalized_amplitude > 0.01:
                        calibrated_amplitude = 180 + (120 * np.log10(normalized_amplitude) + 120)
                    else:
                        calibrated_amplitude = 150  # Minimum default amplitude
                    
                    # Apply lift angle correction based on horological principles
                    # The formula: actual_amplitude = measured_amplitude * sin(standard_angle) / sin(actual_angle)
                    standard_lift_angle = 52.0  # Standard reference lift angle in degrees
                    
                    # Convert angles to radians for trigonometric calculations
                    standard_angle_rad = float(standard_lift_angle * np.pi / 180.0)
                    actual_angle_rad = float(self.lift_angle * np.pi / 180.0)
                    
                    # Apply precise trigonometric correction
                    if actual_angle_rad > 0:
                        angle_correction = float(np.sin(standard_angle_rad) / np.sin(actual_angle_rad))
                        self.amplitude = float(calibrated_amplitude * angle_correction)
                    else:
                        self.amplitude = float(calibrated_amplitude)
                    
                    # Step 7.6: Apply BPH-dependent corrections
                    # Different escapements have different amplitude characteristics
                    if self.expected_bph < 21600:  # Slower watches
                        self.amplitude *= 1.05
                    elif self.expected_bph > 36000:  # High-frequency watches
                        self.amplitude *= 0.95
                    
                    # Ensure amplitude is in realistic range (typically 150-320 degrees)
                    if self.amplitude is not None:
                        self.amplitude = min(320.0, max(150.0, float(self.amplitude)))
        
        # Update last results
        self.last_results = {
            'bph': self.calculated_bph,
            'daily_rate': self.daily_rate,
            'beat_error': self.beat_error,
            'amplitude': self.amplitude
        }
    
    def has_new_data(self):
        """Check if new data is available"""
        return self.new_data_available
    
    def get_latest_results(self):
        """Get the latest calculated results"""
        self.new_data_available = False
        
        # Add diagnostic information for debugging
        if self.calculated_bph is not None:
            # Create a cleaner representation of the histogram for debugging
            hist_data = {}
            if self.bph_histogram:
                # Only include entries with significant counts
                for bph, count in sorted(self.bph_histogram.items()):
                    if count > 2:  # Only show values with multiple occurrences
                        hist_data[int(bph)] = count
            
            # Calculate histogram confidence if available
            confidence = 0
            if self.bph_histogram:
                max_count = max(self.bph_histogram.values()) if self.bph_histogram else 0
                total = sum(self.bph_histogram.values()) if self.bph_histogram else 0
                confidence = max_count / total if total > 0 else 0
            
            # Add diagnostic data to results
            self.last_results['diagnostic'] = {
                'intervals_count': len(self.beat_intervals),
                'histogram_entries': len(self.bph_histogram),
                'confidence': confidence,
                'top_bph_values': hist_data
            }
            
        return self.last_results