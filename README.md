# Guitar-TECHS Dataset

## Overview

Guitar-TECHS is a multimodal dataset designed for guitar-related machine perception research. It features various guitar techniques, musical excerpts, chords, and scales performed by diverse musicians in different recording conditions. The dataset includes multiple perspectives and modalities, such as direct input audio, microphone recordings, MIDI annotations, and video recordings.

## Dataset Structure

The dataset is structured as follows:

```
Guitar-TECHS/
├── audio/
│   ├── directinput/
│   │   ├── directinput_01.wav
│   │   ├── directinput_02.wav
│   │   ├── ...
│   │   ├── directinput_12.wav
│   ├── micamp/
│   │   ├── micamp_01.wav
│   │   ├── micamp_02.wav
│   │   ├── ...
│   │   ├── micamp_12.wav
│
├── video/
│   ├── ego/
│   │   ├── ego_01.mp4
│   │   ├── ego_02.mp4
│   │   ├── ...
│   │   ├── ego_12.mp4
│   ├── exo/
│   │   ├── exo_01.mp4
│   │   ├── exo_02.mp4
│   │   ├── ...
│   │   ├── exo_12.mp4
│
├── metadata/
│   ├── bmps.txt
│   ├── midi/
│   │   ├── midi_01.mid
│   │   ├── midi_02.mid
│   │   ├── ...
│   │   ├── midi_12.mid
```

- **Audio**

  - `directinput/`: Contains direct guitar input recordings (`directinput_01.wav` to `directinput_12.wav`)
  - `micamp/`: Contains recordings captured via microphones and amplifiers (`micamp_01.wav` to `micamp_12.wav`)

- **Video**

  - `ego/`: Egocentric video recordings from the performer’s perspective (`ego_01.mp4` to `ego_12.mp4`)
  - `exo/`: Exocentric video recordings from an external perspective (`exo_01.mp4` to `exo_12.mp4`)

- **Metadata**

  - `bmps.txt`: Contains the beats per minute (BPM) values for each performance, mapping file numbers to their corresponding tempo.
  - `midi/`: MIDI annotations corresponding to each performance (`midi_01.mid` to `midi_12.mid`)

## Setup

To set up the environment, create a virtual environment and install dependencies from `requirements.txt`:

```sh
python -m virtualenv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

## Usage

Run the script using the following command:

```sh
python visualize_GT4D.py --directory /path/to/dataset --filename 01 --output_filename output_01_start0%_end5%.mp4
```

#### Arguments:

- `--directory`: Path to the dataset directory
- `--filename`: Number of the file to process (e.g., `01` for `ego_01.mp4`, `exo_01.mp4`, etc.)
- `--output_filename` (optional): Name of the generated output video
- `--start_pct` (optional): Start percentage of the video to process (default: 0)
- `--end_pct` (optional): End percentage of the video to process (default: 100)

### Features

- Extracts and normalizes audio from video files
- Synchronizes multimodal data
- Generates dynamic overlays for MIDI and audio visualization
- Outputs a composite video with all aligned data

### Example Output

The final output will be a video file (`output_01_start0%_end5%.mp4`) where:

- The top part shows side-by-side ego and exo videos
- The middle part displays MIDI tablature representation
- The bottom part contains a real-time waveform representation of the audio

## License

This dataset and visualization tool are provided for research purposes. Please cite the corresponding paper if you use this resource in your work.

## Contact

For any questions or issues, please reach out to the dataset maintainers.

