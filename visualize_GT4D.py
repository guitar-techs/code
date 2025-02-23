import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import librosa
import pretty_midi
from pydub import AudioSegment
import fire

from moviepy.editor import VideoFileClip, clips_array
from moviepy.video.VideoClip import VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from scipy.signal import correlate
from itertools import product
from io import BytesIO
from PIL import Image

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------
class Config:
    TIME_RESOLUTION: int = 100  # frames per second for the tablature matrix
    VIDEO_CODEC: str = "libx264"
    AUDIO_CODEC: str = "aac"
    FIG_SIZE_AUDIO: Tuple[float, float] = (10, 2)
    FIG_SIZE_MIDI: Tuple[float, float] = (10, 2)
    # Guitar strings in order (from high to low pitch) and standard tuning (MIDI note numbers)
    GUITAR_STRINGS: Tuple[str, ...] = ('e', 'B', 'G', 'D', 'A', 'E')
    STANDARD_TUNING: Dict[str, int] = {
        "E": 40,  # E2
        "A": 45,  # A2
        "D": 50,  # D3
        "G": 55,  # G3
        "B": 59,  # B3
        "e": 64,  # E4
    }

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a NumPy RGB array.

    Args:
        fig (plt.Figure): The matplotlib figure to convert.

    Returns:
        np.ndarray: The rendered figure as an RGB image.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image_array = np.array(image)
    buffer.close()
    plt.close(fig)
    return image_array


def get_audio_from_video(video_path: Path) -> Tuple[np.ndarray, int]:
    """
    Extract and normalize audio from a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        Tuple[np.ndarray, int]: Normalized audio samples and the audio frame rate.
    """
    try:
        audio_segment = AudioSegment.from_file(str(video_path), format="mp4")
    except Exception as e:
        logging.error(f"Error reading audio from {video_path}: {e}")
        raise

    audio_samples = np.array(audio_segment.get_array_of_samples())
    
    # Reshape stereo audio to (n_samples, channels)
    if audio_segment.channels > 1:
        audio_samples = audio_samples.reshape((-1, audio_segment.channels))

    # Normalize the audio to [-1, 1] using original bit depth range
    max_val = np.iinfo(audio_segment.array_type).max
    min_val = np.iinfo(audio_segment.array_type).min
    normalized_audio = audio_samples / max(abs(max_val), abs(min_val))
    return normalized_audio, audio_segment.frame_rate


def flatten_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Flatten multi-channel signals into separate single-channel signals.

    Args:
        signals (Dict[str, np.ndarray]): Dictionary of (possibly multi-channel) signals.

    Returns:
        Dict[str, np.ndarray]: Dictionary with each channel separated (keys appended with channel number).
    """
    flattened: Dict[str, np.ndarray] = {}
    for name, signal in signals.items():
        # Handle mono signals
        if signal.ndim == 1:
            flattened[f"{name}_ch0"] = signal
        # Handle multi-channel signals
        else:
            for ch in range(signal.shape[1]):
                flattened[f"{name}_ch{ch}"] = signal[:, ch]
    return flattened


def midi_note_to_string_fret(guitar_string: str, note: pretty_midi.Note) -> int:
    """
    Calculate the fret number for a given MIDI note on a specific guitar string.

    Args:
        guitar_string (str): The guitar string name (e.g., 'E', 'A', 'D', etc.).
        note (pretty_midi.Note): A PrettyMIDI Note object.

    Returns:
        int: The fret number if playable on the string, or -1 if not.
    """
    if guitar_string not in Config.STANDARD_TUNING:
        raise ValueError(
            f"Invalid string name: {guitar_string}. "
            f"Must be one of: {Config.GUITAR_STRINGS}"
        )
    
    base_pitch = Config.STANDARD_TUNING.get(guitar_string)
    fret = note.pitch - base_pitch
    return fret if 0 <= fret <= 24 else -1


def build_tab_matrix(midi_data: pretty_midi.PrettyMIDI,
                     time_resolution: int = Config.TIME_RESOLUTION) -> np.ndarray:
    """
    Build a tablature matrix from MIDI data for guitar instruments.

    Each row corresponds to a guitar string (ordered as in Config.GUITAR_STRINGS) and each column
    represents a time slice. The matrix is initialized with -2. A fret number (0-24) indicates a played note,
    and -1 marks the note onset.

    Args:
        midi_data (pretty_midi.PrettyMIDI): Parsed MIDI data.
        time_resolution (int): Frames per second in the tablature matrix.

    Returns:
        np.ndarray: 2D tablature matrix.
    """
    total_time = midi_data.get_end_time()
    matrix_width = int(total_time * time_resolution)
    tab_matrix = {string: -2 * np.ones(matrix_width) for string in Config.GUITAR_STRINGS}

    for instrument in midi_data.instruments:
        if instrument.name not in Config.GUITAR_STRINGS:
            raise ValueError(f"Invalid instrument name: {instrument.name}")
        for note in instrument.notes:
            start_idx = int(note.start * time_resolution)
            end_idx = int(note.end * time_resolution)
            fret = midi_note_to_string_fret(instrument.name, note)
            tab_matrix[instrument.name][start_idx:end_idx] = fret
            tab_matrix[instrument.name][start_idx] = -1  # Mark note onset

    return np.vstack([v for v in tab_matrix.values()])



def create_audio_plot_frame(t: float, signals: Dict[str, np.ndarray], sample_rate: int) -> np.ndarray:
    """ 
    Generate an audio plot frame with:
    - 1-second window centered at t
    - Y-axis fixed to [-1, 1]
    - 10 equal vertical divisions
    - Visible plot borders
    - Static vertical grid
    """
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE_AUDIO)
    
    # Calculate fixed 1-second window (0.5s on each side of t)
    window_size = 1.0
    start_time = t - 0.5
    end_time = t + 0.5
    
    # Adjust window boundaries if needed
    min_time = 0
    max_time = max(len(s)/sample_rate for s in signals.values())
    
    if start_time < min_time:
        end_time += (min_time - start_time)
        start_time = min_time
    if end_time > max_time:
        start_time -= (end_time - max_time)
        end_time = max_time
    
    # Calculate absolute boundaries for 10 divisions
    division_size = (end_time - start_time) / 10  # Dynamic division size
    x_min = start_time
    x_max = x_min + (10 * division_size)  # Force exact 10 divisions
    
    # Plot signals with precise alignment
    for name, signal in signals.items():
        start_sample = int(x_min * sample_rate)
        end_sample = int(x_max * sample_rate)
        time_axis = np.linspace(x_min, x_max, end_sample - start_sample)
        ax.plot(time_axis, signal[start_sample:end_sample], 
                label=name, linewidth=0.75)
    
    # Formatting requirements
    ax.set_ylim(-1, 1)
    ax.set_xlim(x_min, x_max)
    ax.axvline(x=t, color='green', linestyle='--', label='Center (t)')
    
    # Configure grid and ticks
    ax.xaxis.set_major_locator(MultipleLocator(division_size))
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    
    # Maintain plot borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        
    ax.tick_params(axis='both', which='both',
                   labelleft=False, labelbottom=False,
                   length=0)
    
    ax.legend(loc="upper right")
    fig.tight_layout()
    
    return fig_to_array(fig)



def create_midi_plot_frame(t: float, tab_matrix: np.ndarray,
                           time_resolution: int = Config.TIME_RESOLUTION) -> np.ndarray:
    """
    Generate an image frame plotting the MIDI tablature at time t without the green center line.

    Args:
        t (float): Current time in seconds.
        tab_matrix (np.ndarray): 2D tablature matrix.
        time_resolution (int): Time resolution used in the tablature matrix.

    Returns:
        np.ndarray: RGB image array of the MIDI plot.
    """
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE_MIDI)
    
    # Determine the center index and the window of data to display
    center_idx = int(t * time_resolution)
    half_window = time_resolution // 2
    start = max(center_idx - half_window, 0)
    end = center_idx + half_window
    
    # Extract the relevant segment of the tab_matrix
    segment = tab_matrix[:, start:end] if end <= tab_matrix.shape[1] else tab_matrix[:, start:]

    # Display the segment as an image
    ax.imshow(np.zeros_like(segment), aspect="auto", cmap="Greys", interpolation="nearest")

    for (row, col), value in np.ndenumerate(segment):
        if value >= 0:
            ax.text(col, row, f"{int(value)}", ha="center", va="center", fontsize=8)
        elif value == -1:
            ax.text(col, row, "V", ha="center", va="center", fontsize=8)

    # Add a vertical green dashed line at the center of the displayed segment.
    ax.axvline(x=center_idx - start, color='green', linestyle='--', label="Center (t)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig_to_array(fig)


# -------------------------------------------------------------------
# Main Video Composition Function
# -------------------------------------------------------------------
def concatenate_videos(directory: str, content: str, filename: str, output_filename: str = None,
                       start_pct: float = 0, end_pct: float = 100) -> None:
    """
    Concatenate ego and exo videos with dynamic audio and MIDI overlays.
    Optionally, only a specified percentage (of total duration) will be processed.
    
    Args:
        directory (str): Base directory containing subdirectories.
        content (str): The type of content to be visualized (i.e. 'techniques', 'music', 'chords', etc.)
        filename (str): Base filename (without extension) for the input files.
        output_filename (str): Name of the output video file.
        start_pct (float): Start percentage (0-100) of the video to process.
        end_pct (float): End percentage (0-100) of the video to process.
    """
    base_dir = Path(directory)
    # Build file paths.
    ego_video_path = base_dir / content / "video" / "ego" / f"ego_{filename}.mp4"
    exo_video_path = base_dir / content / "video" / "exo" / f"exo_{filename}.mp4"
    di_audio_path = base_dir / content / "audio" / "directinput" / f"directinput_{filename}.wav"
    amp_audio_path = base_dir / content / "audio" / "micamp" / f"micamp_{filename}.wav"
    midi_path = base_dir / content / "midi" / f"midi_{filename}.mid"

    # Verify required video files exist.
    if not ego_video_path.exists():
        raise FileNotFoundError(f"Ego video file not found: {ego_video_path}")
    if not exo_video_path.exists():
        raise FileNotFoundError(f"Exo video file not found: {exo_video_path}")

    # Extract audio from video files.
    ego_audio, egosr = get_audio_from_video(ego_video_path)
    exo_audio, exosr = get_audio_from_video(exo_video_path)

    if egosr != exosr:
        raise ValueError("Sample rates for ego and exo videos do not match.")

    # Load additional audio files.
    try:
        di_audio, _ = librosa.load(str(di_audio_path), sr=egosr)
        amp_audio, _ = librosa.load(str(amp_audio_path), sr=egosr)
    except Exception as e:
        logging.error(f"Error loading additional audio files: {e}")
        raise

    # Load MIDI data and construct tablature matrix.
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in midi_data.instruments:
            if instrument.name not in Config.GUITAR_STRINGS:
                raise ValueError(f"Invalid instrument name: {instrument.name}")
    except Exception as e:
        logging.error(f"Error loading MIDI file {midi_path}: {e}")
        raise

    tab_matrix = build_tab_matrix(midi_data, time_resolution=Config.TIME_RESOLUTION)

    # Prepare dictionary of signals and log pairwise lags.
    signals = {
        "ego": ego_audio,
        "exo": exo_audio,
        "direct_input": di_audio,
        "mic_amp": amp_audio,
    }

    # Load and resize video clips.
    try:
        ego_clip = VideoFileClip(str(ego_video_path))
        exo_clip = VideoFileClip(str(exo_video_path))
    except Exception as e:
        logging.error(f"Error loading video clips: {e}")
        raise

    # Resize the clips to have the same height.
    target_height = min(ego_clip.h, exo_clip.h)
    ego_clip_resized = ego_clip.resize(height=target_height)
    exo_clip_resized = exo_clip.resize(height=target_height)
    
    # Arrange the clips side by side.
    combined_clip = clips_array([[ego_clip_resized, exo_clip_resized]])
    
    full_duration = combined_clip.duration

    # Validate and compute the time range.
    if not (0 <= start_pct < end_pct <= 100):
        raise ValueError("start_pct must be >= 0, end_pct <= 100, and start_pct < end_pct.")
    start_time = full_duration * (start_pct / 100)
    end_time = full_duration * (end_pct / 100)
    segment_duration = end_time - start_time

    # Dynamically generate the output filename if not provided
    if output_filename is None:
        output_filename = f"output_{filename}_start{int(start_pct)/100:.2f}_end{int(end_pct)/100:.2f}.mp4"

    # Extract only the desired segment from the base video.
    base_segment = combined_clip.subclip(start_time, end_time)

    # Flatten the signals.
    flattened_signals = flatten_signals(signals)

    # Create dynamic overlay clips for audio and MIDI over the segment.
    # The lambda functions now add start_time so that the plotting matches the absolute timeline.
    audio_overlay = VideoClip(
        lambda t: create_audio_plot_frame(t + start_time, flattened_signals, egosr),
        duration=segment_duration
    ).set_position(("center", combined_clip.h)).resize(width=combined_clip.w)

    midi_overlay = VideoClip(
        lambda t: create_midi_plot_frame(t + start_time, tab_matrix, Config.TIME_RESOLUTION),
        duration=segment_duration
    ).set_position(("center", combined_clip.h)).resize(width=combined_clip.w)

    # Compose the overlays with the base video segment.
    final_video = CompositeVideoClip([
        base_segment.set_position(("center", 0)),
        midi_overlay.set_position(("center", base_segment.h)),
        audio_overlay.set_position(("center", base_segment.h + midi_overlay.h))
    ], size=(base_segment.w, base_segment.h + midi_overlay.h + audio_overlay.h))

    # Set the duration of the final composite to the desired segment duration
    final_video = final_video.set_duration(segment_duration)

    # Write the output file.
    try:
        final_video.write_videofile(output_filename, codec=Config.VIDEO_CODEC, audio_codec=Config.AUDIO_CODEC)
    except Exception as e:
        logging.error(f"Error writing final video: {e}")
        raise

    logging.info(f"Video with dynamic plots saved to {output_filename}")

# -------------------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        fire.Fire(concatenate_videos)
    except Exception as ex:
        logging.exception("An error occurred during execution")
        sys.exit(1)
