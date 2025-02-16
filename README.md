# Guitar-TECHS Dataset Alignment

## Overview

This script aligns the multimodal data in the Guitar-TECHS dataset by correcting synchronization issues between the video and audio modalities. The dataset originally has misaligned recordings, so this script ensures that all signals are properly synchronized by adjusting time lags detected between different modalities.

## Features

- **Extracts Audio from Videos**: Uses `pydub` to extract audio from video files.
- **Computes Lag Between Signals**: Uses cross-correlation to compute time offsets between signals.
- **Aligns Videos**: Adjusts video timestamps by trimming or padding based on detected audio lags.
- **Aligns Audio Files**: Trims or pads microphone-recorded audio to match reference signals.
- **Processes Multiple Files in Batch**: Aligns all files automatically.

## Dataset Structure

The script assumes the following dataset structure:

```
Guitar-TECHS/
├── audio/
│   ├── directinput/
│   │   ├── directinput_01.wav
│   │   ├── directinput_02.wav
│   │   ├── ...
│   ├── micamp/
│   │   ├── micamp_01.wav
│   │   ├── micamp_02.wav
│   │   ├── ...
│
├── video/
│   ├── ego/
│   │   ├── ego_01.mp4
│   │   ├── ego_02.mp4
│   │   ├── ...
│   ├── exo/
│   │   ├── exo_01.mp4
│   │   ├── exo_02.mp4
│   │   ├── ...
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
python align_GT4D.py --input_directory /path/to/dataset --output_directory /path/to/output --start 1 --end 12
```

### Arguments:

- `--input_directory`: Path to the dataset directory.
- `--output_directory`: Path where aligned files will be saved.
- `--start`: Starting file index (default: 1).
- `--end`: Ending file index (default: 12).

## Alignment Process

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

The script will generate aligned video and audio files inside the specified output directory:

```
/path/to/output/
├── audio/
│   ├── micamp/
│   │   ├── micamp_01.wav
│   │   ├── micamp_02.wav
│   │   ├── ...
│
├── video/
│   ├── ego/
│   │   ├── ego_01.mp4
│   │   ├── ego_02.mp4
│   │   ├── ...
│   ├── exo/
│   │   ├── exo_01.mp4
│   │   ├── exo_02.mp4
│   │   ├── ...
```

## License

This script is provided for research purposes. Please cite the dataset's corresponding paper if used in academic work.

## Contact

For any questions or issues, please reach out to the dataset maintainers.

