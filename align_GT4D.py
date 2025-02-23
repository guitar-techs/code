import os
import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv
import soundfile as sf
from scipy.signal import correlate, hilbert
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, ColorClip, AudioClip, concatenate_videoclips

# ================================
# CONFIGURATION
# ================================
OUTLIER_THRESHOLD = 5.0  # seconds; adjust as needed

# ================================
# FUNCTION DEFINITIONS
# ================================

def get_audio_from_vid(vid_path):
    """
    Extract audio from a video file and normalize it to the range [-1, 1].

    Parameters:
        vid_path (str): Path to the video file.

    Returns:
        tuple: (normalized_audio, frame_rate)
            - normalized_audio (np.ndarray): Audio signal array (mono or multi-channel).
            - frame_rate (int): Audio sample rate.
    """
    audio = AudioSegment.from_file(vid_path, format="mp4")
    x = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        # Reshape so that each column is one channel.
        x = x.reshape((-1, audio.channels))
    max_val = np.iinfo(audio.array_type).max
    min_val = np.iinfo(audio.array_type).min
    normalized_x = x / max(abs(max_val), abs(min_val))
    return normalized_x, audio.frame_rate


def compute_lag(signal1, signal2, sr):
    """
    Compute the time lag between two signals using cross-correlation.

    The lag is defined as the time offset corresponding to the maximum absolute correlation.

    Parameters:
        signal1 (np.ndarray): First signal array.
        signal2 (np.ndarray): Second signal array.
        sr (int): Sampling rate of the signals.

    Returns:
        float: Lag in seconds (positive if signal1 lags behind signal2).
    """
    corr = correlate(signal1, signal2, mode='full', method='fft')
    lag_index = np.argmax(np.abs(corr)) - (len(signal2) - 1)
    lag_seconds = lag_index / sr
    return lag_seconds


def load_signal(signal_path, sr=None):
    """
    Load an audio signal using librosa.

    Parameters:
        signal_path (str): Path to the audio file.
        sr (int, optional): Desired sampling rate. If None, the native sampling rate is used.

    Returns:
        tuple: (y, sr)
            - y (np.ndarray): Loaded audio signal (mono).
            - sr (int): Sampling rate of the loaded audio.
    """
    y, sr = librosa.load(signal_path, sr=sr, mono=False)
    return y, sr


def video_alignment(video_path, input_directory, output_directory, egoexo):
    """
    Process a single video file for audio alignment.

    Steps:
      1. Extract audio from the video and load the reference audio signal.
      2. Compute the initial lag between the video's audio and the reference signal.
      3. If the lag exceeds the outlier threshold, skip alignment.
      4. Otherwise, adjust the video by trimming (if lag > 0) or padding (if lag < 0).
      5. Write the adjusted video to the output path.
      6. Verify the final lag after adjustment.

    Parameters:
        video_path (str): Full path to the video file.
        input_directory (str): Base directory containing the video files.
        output_directory (str): Base directory where the aligned video will be saved.
        egoexo (str): Specifies the video type ("ego" or "exo").

    Returns:
        tuple: (initial_lag, final_lag, tolerance)
            - final_lag is None if alignment was skipped.
    """
    # Determine reference audio path dynamically
    reference_signal_path = video_path.replace("video", "audio").replace(egoexo, "directinput").replace(f"{egoexo}_", "directinput_").replace(".mp4", ".wav")

    # Preserve the original folder structure in the output directory
    relative_path = os.path.relpath(video_path, input_directory)
    output_video_path = os.path.join(output_directory, relative_path)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    video_signal, sr = get_audio_from_vid(video_path)
    reference_signal, _ = librosa.load(reference_signal_path, sr=sr, mono=True)

    video_signal_s = video_signal - np.mean(video_signal, axis=0, keepdims=True)
    video_signal_s /= np.std(video_signal_s, axis=0, keepdims=True)
    reference_signal_s = reference_signal - np.mean(reference_signal)
    reference_signal_s /= np.std(reference_signal_s)
    video_envelope = np.abs(hilbert(video_signal_s, axis=0))
    reference_envelope = np.abs(hilbert(reference_signal_s))
    
    if video_signal.ndim == 1:
        initial_lag = compute_lag(video_envelope, reference_envelope, sr)
    else:
        channel_lags = [compute_lag(video_envelope[:, ch], reference_envelope, sr)
                        for ch in range(video_signal_s.shape[1])]
        initial_lag = np.mean(channel_lags)
    
    if abs(initial_lag) > OUTLIER_THRESHOLD:
        print(f"File {video_path}: Outlier detected with initial lag {initial_lag:.4f} s. Skipping alignment.")
        input()
        tolerance = 1.0 / VideoFileClip(video_path).fps
        return initial_lag, None, tolerance
    
    clip = VideoFileClip(video_path)
    fps = clip.fps
    tolerance = 1.0 / fps
    frame_duration = 1.0 / fps
    
    if initial_lag > 0:
        adjusted_shift = math.ceil(initial_lag / frame_duration) * frame_duration
        print(f"File {video_path}: Audio lags behind by {initial_lag:.4f} s; trimming first {adjusted_shift:.4f} s.")
        shifted_clip = clip.subclip(adjusted_shift, clip.duration)
    elif initial_lag < 0:
        pad_duration = abs(initial_lag)
        adjusted_pad = math.ceil(pad_duration / frame_duration) * frame_duration
        print(f"File {video_path}: Audio is ahead by {pad_duration:.4f} s; adding {adjusted_pad:.4f} s of padding.")
        black_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=adjusted_pad)
        black_clip = black_clip.set_fps(clip.fps)
        silent_audio = AudioClip(lambda t: 0, duration=adjusted_pad, fps=sr)
        black_clip = black_clip.set_audio(silent_audio)
        shifted_clip = concatenate_videoclips([black_clip, clip])
    else:
        print(f"File {video_path}: No shift required; audio is aligned.")
        shifted_clip = clip

    shifted_clip.write_videofile(
        output_video_path,
        codec="libx264",
        audio_codec="alac",
        temp_audiofile="temp_audio.m4a",
        remove_temp=True,
        audio_fps=sr,
        ffmpeg_params=[
            "-crf", "18",
            "-preset", "slow",
            "-avoid_negative_ts", "make_zero",
            "-sample_fmt", "s16"
        ]
    )
    
    aligned_audio, _ = get_audio_from_vid(output_video_path)
    aligned_audio -= np.mean(aligned_audio, axis=0, keepdims=True)
    aligned_audio /= np.std(aligned_audio, axis=0, keepdims=True)
    aligned_audio_env = np.abs(hilbert(aligned_audio, axis=0))
    if aligned_audio.ndim == 1:
        new_lag = compute_lag(aligned_audio_env, reference_envelope, sr)
    else:
        new_channel_lags = [compute_lag(aligned_audio_env[:, ch], reference_envelope, sr)
                            for ch in range(aligned_audio.shape[1])]
        new_lag = np.mean(new_channel_lags)
    
    print(f"File {video_path} : Final lag: {new_lag:.4f} s (tolerance: {tolerance:.4f} s)\n")
    return initial_lag, new_lag, tolerance


def audio_alignment(misaligned_path, input_directory, output_directory, tolerance=0.01):
    """
    Process a single audio file for alignment with a reference signal.

    Steps:
      1. Load the misaligned audio signal and the corresponding reference signal.
      2. Compute the initial lag between the signals.
      3. Align the audio by applying zero-padding (if lag > 0) or trimming (if lag < 0).
      4. Save the aligned audio signal.
      5. Re-compute the final lag after alignment.

    Parameters:
        misaligned_path (str): Full path to the misaligned audio file.
        input_directory (str): Base directory containing the audio files.
        content (str): The type of content to be visualized (i.e. 'techniques', 'music', 'chords', etc.)
        output_directory (str): Base directory where the aligned audio will be saved.
        tolerance (float): Acceptable lag tolerance (default is 0.01 s).

    Returns:
        tuple: (initial_lag, final_lag, tolerance)
    """
    # Determine the reference signal path based on the misaligned file's structure
    reference_path = misaligned_path.replace("micamp", "directinput")

    # Preserve the folder structure in output
    relative_path = os.path.relpath(misaligned_path, input_directory)
    output_path = os.path.join(output_directory, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    signal, sr = load_signal(misaligned_path)
    ref_signal, _ = librosa.load(reference_path, sr=sr, mono=True)
    signal_s = signal - np.mean(signal, axis=1, keepdims=True)
    signal_s /= np.std(signal_s, axis=1, keepdims=True) 
    signal_s = signal_s.T
    ref_signal_s = ref_signal - np.mean(ref_signal)
    ref_signal_s /= np.std(ref_signal_s)
    
    if signal.ndim == 1:
        initial_lag = compute_lag(signal_s, ref_signal_s, sr)
    else:
        channel_lags = [compute_lag(signal_s[:, ch], ref_signal_s, sr)
                        for ch in range(signal_s.shape[1])]
        initial_lag = np.mean(channel_lags)
    
    # Apply alignment
    if initial_lag > 0:
        pad_samples = int(math.ceil(initial_lag * sr))
        aligned_signal = np.pad(signal, ((0, 0), (pad_samples, 0)), mode='constant')
        print(f"File {misaligned_path}: Audio lags behind by {initial_lag:.4f} s; padding {pad_samples} samples.")
    elif initial_lag < 0:
        trim_samples = int(math.ceil(abs(initial_lag) * sr))
        aligned_signal = signal[:, trim_samples:]
        print(f"File {misaligned_path}: Audio is ahead by {abs(initial_lag):.4f} s; trimming {trim_samples} samples.")
    else:
        aligned_signal = signal.copy()
        print(f"File {misaligned_path}: No shift required; audio is aligned.")
    
    sf.write(output_path, aligned_signal.T, sr)

    # Verify final lag
    aligned_signal, _ = librosa.load(output_path, sr=sr, mono=False)
    if aligned_signal.ndim == 1:
        aligned_signal = np.expand_dims(aligned_signal, axis=0)
    new_lag = np.mean([compute_lag(aligned_signal[ch, :], ref_signal_s, sr) for ch in range(aligned_signal.shape[0])])
    print(f"File {misaligned_path}: Final lag: {new_lag:.4f} s (tolerance: {tolerance:.4f} s)\n")

    return initial_lag, new_lag, tolerance


def process_video_files(input_directory, output_directory, file_paths):
    """
    Process multiple video files for alignment across both "ego" and "exo" types.

    Parameters:
        input_directory (str): Base directory containing the video and audio files.
        output_directory (str): Base directory where aligned videos will be saved.
        file_paths (list of str): List of full paths to video files.

    Returns:
        dict: Dictionary containing processed file paths, initial lags, final lags, and tolerance value.
    """
    processed_file_paths = []
    initial_lags = []
    final_lags = []
    tolerance_value = None

    for file_path in file_paths:
        # Determine whether it's an "ego" or "exo" video based on filename
        egoexo = "ego" if "ego_" in file_path else "exo"

        # Construct output path
        relative_path = os.path.relpath(file_path, input_directory)
        output_video_path = os.path.join(output_directory, relative_path)
        
        # Perform video alignment
        init_lag, final_lag, tol = video_alignment(file_path, input_directory, output_directory, egoexo)

        if final_lag is not None:
            processed_file_paths.append(output_video_path)
            initial_lags.append(init_lag)
            final_lags.append(final_lag)
            tolerance_value = tol
    
    return {
        "processed_file_paths": processed_file_paths,
        "initial_lags": initial_lags,
        "final_lags": final_lags,
        "tolerance": tolerance_value
    }


def process_audio_files(input_directory, output_directory, file_paths, tolerance=0.01):
    """
    Process multiple audio files for alignment.

    Parameters:
        input_directory (str): Base directory containing the audio files.
        output_directory (str): Base directory where aligned audio will be saved.
        file_paths (list of str): List of full paths to misaligned audio files.
        tolerance (float): Acceptable lag tolerance (default is 0.01 s).

    Returns:
        dict: Dictionary containing initial and final lags for the processed audio files.
    """
    initial_lags = []
    final_lags = []

    for file_path in file_paths:
        init_lag, final_lag, _ = audio_alignment(file_path, input_directory, output_directory, tolerance)
        initial_lags.append(init_lag)
        final_lags.append(final_lag)
    
    return {
        "initial_lags": initial_lags,
        "final_lags": final_lags
    }


def main(input_directory, output_directory):
    # Lists to store absolute paths of audio and video files
    micamp_files = []
    ego_files = []
    exo_files = []

    # Recursively find all files
    for root, _, files in os.walk(input_directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.startswith("micamp_") and file.endswith(".wav"):
                micamp_files.append(file_path)
            elif file.startswith("ego_") and file.endswith(".mp4"):
                ego_files.append(file_path)
            elif file.startswith("exo_") and file.endswith(".mp4"):
                exo_files.append(file_path)

    print("Starting video alignment processing...\n")
    video_results = process_video_files(input_directory, output_directory, ego_files + exo_files)
    print("Video alignment processing completed.\n")

    print("Starting audio alignment processing...\n")
    audio_results = process_audio_files(input_directory, output_directory, micamp_files)
    print("Audio alignment processing completed.\n")
    
    return {
        "video_results": video_results,
        "audio_results": audio_results
    }


if __name__ == "__main__":
    import fire
    fire.Fire(main)
