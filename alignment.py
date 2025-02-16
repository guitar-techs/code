import os
import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv
import soundfile as sf
from scipy.signal import correlate
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
    y, sr = librosa.load(signal_path, sr=sr, mono=True)
    return y, sr

def video_alignment(file_id, input_directory, output_directory, egoexo="exo"):
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
        file_id (str): Identifier for the file.
        input_directory (str): Base directory containing the video and audio files.
        output_directory (str): Base directory where the aligned video will be saved.
        egoexo (str): Specifies the video type ("ego" or "exo").

    Returns:
        tuple: (initial_lag, final_lag, tolerance)
            - final_lag is None if alignment was skipped.
    """
    video_path = os.path.join(input_directory, "video", egoexo, f"{egoexo}_{file_id}.mp4")
    reference_signal_path = os.path.join(input_directory, "audio", "directinput", f"directinput_{file_id}.wav")
    output_dir = os.path.join(output_directory, "video", egoexo)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{egoexo}_{file_id}.mp4")
    
    video_signal, sr = get_audio_from_vid(video_path)
    reference_signal, _ = librosa.load(reference_signal_path, sr=sr, mono=True)
    
    if video_signal.ndim == 1:
        initial_lag = compute_lag(video_signal, reference_signal, sr)
    else:
        channel_lags = [compute_lag(video_signal[:, ch], reference_signal, sr)
                        for ch in range(video_signal.shape[1])]
        initial_lag = np.mean(channel_lags)
    
    if abs(initial_lag) > OUTLIER_THRESHOLD:
        print(f"File {file_id} ({egoexo}): Outlier detected with initial lag {initial_lag:.4f} s. Skipping alignment.")
        tolerance = 1.0 / VideoFileClip(video_path).fps
        return initial_lag, None, tolerance
    
    clip = VideoFileClip(video_path)
    fps = clip.fps
    tolerance = 1.0 / fps
    frame_duration = 1.0 / fps
    
    if initial_lag > 0:
        adjusted_shift = math.ceil(initial_lag / frame_duration) * frame_duration
        print(f"File {file_id} ({egoexo}): Audio lags behind by {initial_lag:.4f} s; trimming first {adjusted_shift:.4f} s.")
        shifted_clip = clip.subclip(adjusted_shift, clip.duration)
    elif initial_lag < 0:
        pad_duration = abs(initial_lag)
        adjusted_pad = math.ceil(pad_duration / frame_duration) * frame_duration
        print(f"File {file_id} ({egoexo}): Audio is ahead by {pad_duration:.4f} s; adding {adjusted_pad:.4f} s of padding.")
        black_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=adjusted_pad)
        black_clip = black_clip.set_fps(clip.fps)
        silent_audio = AudioClip(lambda t: 0, duration=adjusted_pad, fps=sr)
        black_clip = black_clip.set_audio(silent_audio)
        shifted_clip = concatenate_videoclips([black_clip, clip])
    else:
        print(f"File {file_id} ({egoexo}): No shift required; audio is aligned.")
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
    if aligned_audio.ndim == 1:
        new_lag = compute_lag(aligned_audio, reference_signal, sr)
    else:
        new_channel_lags = [compute_lag(aligned_audio[:, ch], reference_signal, sr)
                            for ch in range(aligned_audio.shape[1])]
        new_lag = np.mean(new_channel_lags)
    
    print(f"File {file_id} ({egoexo}): Final lag: {new_lag:.4f} s (tolerance: {tolerance:.4f} s)\n")
    return initial_lag, new_lag, tolerance

def audio_alignment(file_id, input_directory, output_directory, tolerance=0.01):
    """
    Process a single audio file for alignment with a reference signal.

    Steps:
      1. Load the misaligned audio signal and the corresponding reference signal.
      2. Compute the initial lag between the signals.
      3. Align the audio by applying zero-padding (if lag > 0) or trimming (if lag < 0).
      4. Save the aligned audio signal.
      5. Re-compute the final lag after alignment.

    Parameters:
        file_id (str): Identifier for the file.
        input_directory (str): Base directory containing the audio files.
        output_directory (str): Base directory where the aligned audio will be saved.
        tolerance (float): Acceptable lag tolerance (default is 0.01 s).

    Returns:
        tuple: (initial_lag, final_lag, tolerance)
    """
    misaligned_path = os.path.join(input_directory, "audio", "micamp", f"micamp_{file_id}.wav")
    reference_path = os.path.join(input_directory, "audio", "directinput", f"directinput_{file_id}.wav")
    output_dir = os.path.join(output_directory, "audio", "micamp")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"micamp_{file_id}.wav")
    
    signal, sr = load_signal(misaligned_path)
    ref_signal, _ = load_signal(reference_path, sr=sr)
    
    init_lag = compute_lag(signal, ref_signal, sr)
    
    if init_lag > 0:
        pad_samples = int(math.ceil(init_lag * sr))
        aligned_signal = np.concatenate((np.zeros(pad_samples), signal))
        print(f"File {file_id}: Audio lags behind by {init_lag:.4f} s; padding with {pad_samples} zeros.")
    elif init_lag < 0:
        trim_samples = int(math.ceil(abs(init_lag) * sr))
        aligned_signal = signal[trim_samples:]
        print(f"File {file_id}: Audio is ahead by {abs(init_lag):.4f} s; trimming first {trim_samples} samples.")
    else:
        aligned_signal = signal.copy()
        print(f"File {file_id}: No shift required; audio is aligned.")
    
    sf.write(output_path, aligned_signal, sr)
    aligned_signal, sr_aligned = librosa.load(output_path)
    ref_signal, _ = librosa.load(reference_path, sr=sr_aligned)
    new_lag = compute_lag(aligned_signal, ref_signal, sr)
    print(f"File {file_id}: Final lag: {new_lag:.4f} s (tolerance: {tolerance:.4f} s)\n")
    
    return init_lag, new_lag, tolerance

def process_video_files(input_directory, output_directory, file_ids):
    """
    Process multiple video files for alignment across both "ego" and "exo" types.

    Parameters:
        input_directory (str): Base directory containing the video and audio files.
        output_directory (str): Base directory where aligned videos will be saved.
        file_ids (list of str): List of file identifiers.

    Returns:
        dict: Dictionary containing processed file IDs, initial lags, final lags, and tolerance value.
    """
    processed_file_ids = []
    initial_lags = []
    final_lags = []
    tolerance_value = None

    for fid in file_ids:
        for video_type in ["ego", "exo"]:
            init_lag, final_lag, tol = video_alignment(fid, input_directory, output_directory, egoexo=video_type)
            if final_lag is not None:
                processed_file_ids.append(f"{video_type}_{fid}")
                initial_lags.append(init_lag)
                final_lags.append(final_lag)
                tolerance_value = tol
    
    return {
        "processed_file_ids": processed_file_ids,
        "initial_lags": initial_lags,
        "final_lags": final_lags,
        "tolerance": tolerance_value
    }

def process_audio_files(input_directory, output_directory, file_ids, tolerance=0.01):
    """
    Process multiple audio files for alignment.

    Parameters:
        input_directory (str): Base directory containing the audio files.
        output_directory (str): Base directory where aligned audio will be saved.
        file_ids (list of str): List of file identifiers.
        tolerance (float): Acceptable lag tolerance (default is 0.01 s).

    Returns:
        dict: Dictionary containing initial and final lags for the processed audio files.
    """
    initial_lags = []
    final_lags = []

    for fid in file_ids:
        init_lag, final_lag, _ = audio_alignment(fid, input_directory, output_directory, tolerance)
        initial_lags.append(init_lag)
        final_lags.append(final_lag)
    
    return {
        "initial_lags": initial_lags,
        "final_lags": final_lags
    }

def main(input_directory, output_directory, start=1, end=12):
    """
    Main function to process video and audio files for alignment.

    Parameters:
        input_directory (str): Base directory containing the source files.
        output_directory (str): Base directory where aligned files will be saved.
        start (int): Starting file id number (inclusive).
        end (int): Ending file id number (exclusive). File ids will be formatted as two-digit numbers.
    """
    file_ids = [f"{i:02d}" for i in range(start, end+1)]
    
    print("Starting video alignment processing...\n")
    video_results = process_video_files(input_directory, output_directory, file_ids)
    print("Video alignment processing completed.\n")

    print("Starting audio alignment processing...\n")
    audio_results = process_audio_files(input_directory, output_directory, file_ids)
    print("Audio alignment processing completed.\n")
    
    return {
        "video_results": video_results,
        "audio_results": audio_results
    }

if __name__ == "__main__":
    import fire
    fire.Fire(main)
