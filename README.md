# Guitar-TECHS Dataset Alignment

## Overview

This script aligns the multimodal data in the Guitar-TECHS dataset by correcting synchronization issues between the video and audio modalities. The dataset originally has misaligned recordings, so this script ensures that all signals are properly synchronized by adjusting time lags detected between different modalities while preserving the original folder structure.

## Features

- **Recursively Processes All Files**: The script searches the specified dataset directory and aligns all detected audio and video files.
- **Extracts Audio from Videos**: Uses `pydub` to extract audio from video files.
- **Computes Lag Between Signals**: Uses cross-correlation to compute time offsets between signals.
- **Aligns Videos**: Adjusts video timestamps by trimming or padding based on detected audio lags.
- **Aligns Audio Files**: Trims or pads microphone-recorded audio to match reference signals.
- **Maintains Directory Structure**: Keeps the original dataset structure when saving aligned files.
- **Processes Multiple Files in Batch**: Aligns all files automatically.

## Dataset Structure

```
Guitar-TECHS/
├── P1/
│   ├── chords/
│   │   ├── audio/
│   │   ├── midi/
│   │   ├── video/
│   ├── scales/
│   ├── singlenotes/
│   ├── techniques/
│
├── P2/
│   ├── chords/
│   ├── scales/
│   ├── singlenotes/
│   ├── techniques/
│
├── P3/music/
│   ├── audio/
│   ├── midi/
│   ├── video/
```

## Installation

Ensure you have the necessary dependencies installed. You can set up a virtual environment and install them using:

```sh
python -m virtualenv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

## Usage

Run the script using the following command:

```sh
python align_GT4D.py --input_directory /path/to/dataset --output_directory /path/to/output
```

### Arguments:

- `--input_directory`: Path to the dataset directory (e.g., `/scratch/GuitarTECHS/`).
- `--output_directory`: Path where aligned files will be saved.

## Alignment Process

### **Recursive File Processing**

The script scans all subdirectories within `input_directory`, detects video and audio files, and processes them while keeping the original folder structure intact in `output_directory`.

### **Video Alignment**

1. Extracts audio from `ego` and `exo` videos.
2. Computes the lag between extracted audio and reference audio (`directinput` audio).
3. Adjusts video timestamps:
   - **If video lags**: Trims the beginning.
   - **If video is ahead**: Pads the video with a black screen.
4. Saves the adjusted video with aligned audio.

### **Audio Alignment**

1. Loads misaligned microphone-recorded audio (`micamp`).
2. Computes lag against `directinput` audio.
3. Adjusts timing:
   - **If micamp lags**: Adds silence padding.
   - **If micamp is ahead**: Trims the start.
4. Saves the corrected audio file.

## Example Output

The script will generate aligned video and audio files inside the specified output directory, preserving the original structure:

```
/path/to/output/
├── P1/
│   ├── chords/
│   │   ├── audio/
│   │   ├── video/
│   ├── scales/
│   ├── singlenotes/
│   ├── techniques/
│
├── P2/
│   ├── chords/
│   ├── scales/
│   ├── singlenotes/
│   ├── techniques/
│
├── P3/music/
│   ├── audio/
│   ├── video/
```

## License

This script is provided for research purposes. Please cite the dataset's corresponding paper if used in academic work.

## Contact

For any questions or issues, please reach out to the dataset maintainers.

