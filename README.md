<div align="center">
  <img width="300" height="300" src="assets/icons/icon.ico" alt="Package Logo">
  <h1><b>PyTimegrapher</b></h1>
</div>

This application is a timegrapher for mechanical watches that analyzes watch accuracy through your computer's microphone.

## Description

The application uses your computer's microphone to listen to the ticking of a mechanical watch and calculate the following metrics:

- **BPH (Beats Per Hour)**: The number of beats per hour of the watch
- **Daily Rate**: The deviation in seconds per day from exact time
- **Beat Error**: The error in the symmetry of the beats
- **Amplitude**: The oscillation amplitude of the balance wheel in degrees

## Usage

### Important Notes

> **NOTE**: This application requires access to your computer's microphone.

To use this application with a real watch:

1. Download the code to your local computer
2. Install dependencies: `pip install -r requirements.txt` or `pip install PyQt5 pyqtgraph numpy scipy pyaudio pandas pyqtdarktheme`
3. Run the application: `python main.py`

### Usage Instructions

1. Place your watch close to your computer's microphone
2. Ensure your environment is quiet with minimal background noise
3. Select the BPH mode (auto-detect or preset) and lift angle for your watch
4. Press 'Start Recording' to begin analysis
5. Wait at least 10-15 seconds for accurate measurements

## Configuration

### BPH Modes

The application offers two modes for BPH detection:
- **Auto-detect** (default): Automatically detects the closest BPH from common presets
- **Preset Selection**: Manually select from common BPH values

### Lift Angle

Typical lift angle is:
- 52° for most modern watches
- 45° for vintage watches

### Common BPH Presets

- 12,000 BPH: Some vintage pocket watches
- 14,400 BPH: Certain vintage watches
- 18,000 BPH (5 beats per second): Many vintage watches
- 19,800 BPH: Specific vintage calibers
- 21,600 BPH (6 beats per second): Mid-range watches
- 25,200 BPH (7 beats per second): Some modern watches
- 28,800 BPH (8 beats per second): High-quality modern watches
- 36,000 BPH (10 beats per second): High-frequency watches
- 43,200 BPH (12 beats per second): Specialized high-beat movements
- 72,000 BPH (20 beats per second): Ultra high-frequency watches

## Technologies Used

- Python
- PyQt5 and PyQtGraph for the desktop user interface
- PyAudio for audio recording
- NumPy and SciPy for signal analysis
- Pandas for data management

## Key Features

- Native and professional desktop user interface
- Real-time daily rate deviation graph
- Simple configuration for different types of watches 
- Color-coded daily rate display
- High-precision watch accuracy calculations
- Automatic BPH detection with nearest preset matching