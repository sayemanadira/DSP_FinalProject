import numpy as np
import librosa as lb
import time
import pyaudio
import wave
import sys
import pyaudio
from scipy.signal import medfilt
import scipy.io.wavfile as wavfile
import csv
import cProfile
import pstats
import io
import time

# Constants
CHUNK = L = 2048
min_Ha = 2  # Minimum analysis hop size
L_ola = 256
Hs = L // 4
Hs_ola = L_ola // 2
alpha = 0.75
window = np.hanning(L)
output_buffer = np.zeros(L)
prev_fft = None
prev_phase = np.zeros(L//2 + 1)

def estimateIF(S, sr, hop_samples):
    '''
    Estimates the instantaneous frequencies in a STFT matrix.
    
    Inputs
    S: the STFT matrix, should only contain the lower half of the frequency bins
    sr: sampling rate
    hop_samples: the hop size of the STFT analysis in samples
    
    Returns a matrix containing the estimated instantaneous frequency at each time-frequency bin.
    This matrix should contain one less column than S.
    '''
    hop_sec = hop_samples / sr
    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1,1))    
    unwrapped = np.angle(S[:,1:]) - np.angle(S[:,0:-1]) - w_nom * hop_sec
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / hop_sec
    return w_if


def invert_stft(S, hop_length, window):
    '''
    Reconstruct a signal from a modified STFT matrix.
    
    Inputs
    S: modified STFT matrix
    hop_length: the synthesis hop size in samples
    window: an array specifying the window used for FFT analysis
    
    Returns a time-domain signal y whose STFT is closest to S in squared error distance.
    '''
    
    L = len(window)
    
    # construct full stft matrix
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0:S.shape[0],:] = S
    Sfull[S.shape[0]:,:] = np.conj(np.flipud(S[1:fft_size//2,:]))
    
    # compute inverse FFTs
    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:,i] = np.fft.ifft(Sfull[:,i])
    frames = np.real(frames) # remove imaginary components due to numerical roundoff
    
    # synthesis frames
    den = calc_sum_squared_window(window, hop_length)
    frames = frames * window.reshape((-1,1)) / den.reshape((-1,1))
    
    # reconstruction
    y = np.zeros(hop_length*(frames.shape[1]-1) + L)
    for i in range(frames.shape[1]):
        offset = i * hop_length
        y[offset:offset+L] += frames[:,i]
    
    return y

def harmonic_percussive_separation(x, sr=22050, fft_size = 2048, hop_length=512, lh=6, lp=6):
    window = np.hanning(fft_size)
    X = lb.core.stft(x, n_fft=fft_size, hop_length=512, window=window, center=False)
    Y = np.abs(X)
    Yh = medfilt(Y, (1, 2*lh+1))
    Yp = medfilt(Y, (2*lp+1, 1))
    Mh = (Yh > Yp)
    Mp = np.logical_not(Mh)
    Xh = X * Mh
    Xp = X * Mp
    xh = invert_stft(Xh, hop_length, window)
    xp = invert_stft(Xp, hop_length, window)
    
    return xh, xp, Xh, Xp

def calc_sum_squared_window(window, hop_length):
    '''
    Calculates the denominator term for computing synthesis frames.
    
    Inputs
    window: array specifying the window used in FFT analysis
    hop_length: the synthesis hop size in samples
    
    Returns an array specifying the normalization factor.
    '''
    assert (len(window) % hop_length == 0), "Hop length does not divide the window evenly."
    
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i*hop_length)
        
    return den

def float2pcm(sig, dtype='int16'):
    # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


# Pre-compute values
# Load audio file
file_name = 'samples/runtime_samples/fred_10sec.wav'
audio_data, audio_sr = lb.load(file_name)
xh, _, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)

# if max(abs(xh)) > 1:
#     xh = xh / max(abs(xh))
# # elif max(abs(xp)) > 1:
# #     xp = xp / max(abs(xp))

# xh = float2pcm(xh).astype(np.int16)
omega_nom = np.arange(L//2 + 1) * 2 * np.pi * audio_sr / L
den = np.sum(window**2)  # Simplified denominator calculation

def phase_vocoder_processing(S_mag_lookup, S_phase_lookup, w_if_lookup, pos, sr):
    """Process one frame EXACTLY as in hybrid_opt.py"""
    global prev_phase
    
    timings = {}
    Ha = int(Hs/alpha)
    
    # 1. Frame lookup and phase calculation
    start = time.perf_counter()
    nn_frame = int(round(pos / min_Ha))  # Nearest neighbor for magnitude
    lb_frame = int(pos/min_Ha)           # Lower bound for phase increment
    timings['analysis_window'] = time.perf_counter() - start

    start = time.perf_counter()
    if pos == 0:
        phase = S_phase_lookup[:, 0]
    else:
        phase_inc = w_if_lookup[:, lb_frame-1] * (Hs / audio_sr)
        phase = prev_phase + phase_inc
    timings['phase_calc'] = time.perf_counter() - start

    # 2. Reconstruction
    start = time.perf_counter()
    S_mod = S_mag_lookup[:, nn_frame] * np.exp(1j * phase)
    frame = np.fft.irfft(S_mod) * (window.reshape((-1,1))/den.reshape((-1,1))).flatten()
    timings['reconstruction'] = time.perf_counter() - start
    frame = float2pcm(frame)
    
    prev_phase = phase
    return frame, timings


def run_processing(xh, num_runs=100):
    """Complete processing pipeline matching hybrid_opt.py"""
    # Precompute STFT tables exactly as in original
    S_lookup = lb.core.stft(xh, n_fft=L, hop_length=min_Ha, center=False)
    S_mag_lookup = np.abs(S_lookup)
    S_phase_lookup = np.angle(S_lookup)
    w_if_lookup = estimateIF(S_lookup, audio_sr, min_Ha)
    
    cumulative_times = {
        'analysis_window': [],
        'phase_calc': [],
        'reconstruction': [],
        'total_frames': 0
    }
    
    for _ in range(num_runs):
        pos = 0
        run_times = {'analysis_window': 0.0, 'phase_calc': 0.0, 'reconstruction': 0.0}
        frame_count = 0
        
        while pos <= len(xh) - L:
            frame, timings = phase_vocoder_processing(S_mag_lookup, S_phase_lookup, 
                w_if_lookup, pos, audio_sr
            )
            
            # # Overlap-add (matches original)
            # output_buffer[:-Hs] = output_buffer[Hs:]
            # output_buffer[-Hs:] = 0
            # output_buffer += frame
            
            for key in run_times:
                run_times[key] += timings[key]
            frame_count += 1
            pos += int(Hs/alpha)
        
        for key in run_times:
            cumulative_times[key].append(run_times[key])
        cumulative_times['total_frames'] = frame_count
    
    # Reporting (matches hybrid_opt.py format)
    print("Timing Analysis Results:")
    for stage in ['analysis_window', 'phase_calc', 'reconstruction']:
        avg = np.mean(cumulative_times[stage]) * 1000
        std = np.std(cumulative_times[stage]) * 1000
        print(f"{stage}: {avg:.3f} ± {std:.3f} ms")
    print(f"Total frames: {cumulative_times['total_frames']}")



# def run_timing_analysis(num_runs=100):
#     """Run timing analysis on the phase vocoder components.
#     Measures cumulative runtime per stage (analysis, FFT, phase mod, reconstruction)
#     for the entire audio input, averaged over `num_runs` runs.
#     """
#     global pos, prev_fft, prev_phase
    
#     # Initialize accumulators for cumulative times (per run)
#     cumulative_times = {
#         'analysis_window': [],
#         'fft': [],
#         'phase_mod': [],
#         'reconstruction': [],
#         'total_frames': 0  # Just for reference (same for all runs)
#     }
    
#     for run in range(num_runs):
#         # Reset state for each run
#         pos = 0
#         prev_fft = None
#         prev_phase = np.zeros(L//2 + 1)
        
#         # Initialize per-run cumulative timings
#         run_totals = {
#             'analysis_window': 0.0,
#             'fft': 0.0,
#             'phase_mod': 0.0,
#             'reconstruction': 0.0,
#             'total_frames': 0
#         }
        
#         # Process all frames in the audio input
#         while pos <= len(xh) - L:
#             _, S, prev_phase, run_timings = phase_vocoder_processing(
#                 xh[pos:pos+L], window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr
#             )
            
#             # Accumulate timings for this frame
#             for key in run_totals:
#                 run_totals[key] += run_timings[key]
#                 if key == 'total_frames':
#                     run_totals[key] += 1
            
#             prev_fft = S
#             pos += int(Hs/alpha)
        
#         # Store cumulative times for this run
#         for key in run_totals:
#             cumulative_times[key].append(run_totals[key])
        
#         # Store total frames processed (same for all runs)
#         # if run == 0:
#         #     cumulative_times['total_frames'] = len(cumulative_times['analysis_window'])
    
#     # Calculate and print statistics
#     print("\nPhase Vocoder Timing Analysis (Cumulative per Audio Input):")
#     print(f"Total frames processed per run: {run_totals['total_frames']}")
#     print(f"Number of runs averaged: {num_runs}")
#     print("\nAverage Cumulative Time per Stage (ms):")
    
#     for stage in ['analysis_window', 'fft', 'phase_mod', 'reconstruction']:
#         avg_time_ms = np.mean(cumulative_times[stage]) * 1000
#         std_dev_ms = np.std(cumulative_times[stage]) * 1000
#         print(
#             f"{stage.replace('_', ' ').title():<18} "
#             f"Avg: {avg_time_ms:.3f} ms ± {std_dev_ms} ms"
#         )
    
#     # Optional: Print total processing time (sum of all stages)
#     total_avg = sum(np.mean(cumulative_times[stage]) for stage in [
#         'analysis_window', 'fft', 'phase_mod', 'reconstruction'
#     ])
#     print(f"\nTotal Avg. Processing Time per Run: {total_avg:.3f} seconds")
# def run_timing_analysis(num_runs=100):
#     """Run timing analysis on the phase vocoder components"""
#     global pos, prev_fft, prev_phase
    
#     # Initialize timing accumulators
#     timings = {
#         'analysis_window': 0.0,
#         'fft': 0.0,
#         'phase_mod': 0.0,
#         'reconstruction': 0.0,
#         'total_frames': 0
#     }
    
#     for run in range(num_runs):
#         pos = 0
#         prev_fft = None
#         prev_phase = np.zeros(L//2 + 1)
        
#         while pos <= len(xh) - L:
#             # Run processing and collect timings
#             _, S, prev_phase, run_timings = phase_vocoder_processing(
#                 xh[pos:pos+L], window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr
#             )
            
#             # Accumulate timings
#             for key in run_timings:
#                 timings[key] += run_timings[key]
#             timings['total_frames'] += 1
            
#             prev_fft = S
#             pos += int(Hs/alpha)
    
#     # Calculate and print average timings
#     print("\nPhase Vocoder Timing Analysis:")
#     print(f"Total frames processed: {timings['total_frames']}")
#     print(f"Number of runs: {num_runs}")
#     print("\nAverage times per frame (ms):")
#     for key in ['analysis_window', 'fft', 'phase_mod', 'reconstruction']:
#         avg_time = timings[key] * 1000
#         print(f"{key.replace('_', ' ').title():<15} {avg_time:.3f} ms")
    
#     total_time = sum(timings[k] for k in ['analysis_window', 'fft', 'phase_mod', 'reconstruction'])
#     print(f"\nTotal processing time: {total_time:.3f} seconds")

# def run_timing_analysis(num_runs=100):
#     """Run timing analysis on the phase vocoder components"""
#     global pos, prev_fft, prev_phase
    
#     # Initialize timing accumulators
#     timings = {
#         'analysis_window': [],
#         'fft': [],
#         'phase_mod': [],
#         'reconstruction': []
#     }
    
#     # Prepare audio data
#     # xh = float2pcm(audio_data).astype(np.int16)
#     pos = 0
#     cumul_analy = 0
#     cumul_fft = 0
#     cumul_phase = 0
#     cumul_recon = 0
    
#     for _ in range(num_runs):
#         while pos <= len(xh) - L:
#             pos = 0
#             prev_fft = None
#             prev_phase = np.zeros(L//2 + 1)
        
#             # Run processing and collect timings
#             _, S, prev_phase, run_timings = phase_vocoder_processing(
#                 xh[pos:pos+L], window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr
#             )
#             # print("Getting culmutatative runtion")
        

#             cumul_analy += run_timings['analysis_window']
#             cumul_fft += run_timings['fft']
#             cumul_phase += run_timings['phase_mod']
#             cumul_recon += run_timings['reconstruction']
#             # Accumulate timings    
        
#             prev_fft = S
#             pos += int(Hs/alpha)
    

#         timings['analysis_window'].append(cumul_analy)
#         timings['fft'].append(cumul_fft)
#         timings['phase_mod'].append(cumul_phase)
#         timings['reconstruction'].append(cumul_recon)

#         print("Reseting...")
#         pos = 0
#         prev_fft = None
#         prev_phase = np.zeros(L//2 + 1)

#     # Calculate and print average timings
#     print("\nPhase Vocoder Timing Analysis (averages):")
#     for key in timings:
#         avg_time = np.mean(timings[key]) * 1000  # Convert to milliseconds
#         print(f"{key.replace('_', ' ').title():<15} {avg_time:.3f} ms")

# if __name__ == "__main__":
#     # Run the timing analysis
#     run_timing_analysis(num_runs=100)
    
#     # Clean up audio resources
#     p = pyaudio.PyAudio()
#     p.terminate()

if __name__ == "__main__":
    # Run the timing analysis
    run_processing(xh, num_runs=200)
    
    # Clean up audio resources
    p = pyaudio.PyAudio()
    p.terminate()