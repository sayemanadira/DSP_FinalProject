import wave
import sys
import pyaudio
import numpy as np
import keyboard 
import librosa as lb
from scipy.signal import medfilt
import scipy.io.wavfile as wavfile
import scipy
import csv

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

def calc_sum_squared_window_any(window, hop_length):
    """
    Calculates the denominator for OLA normalization for any hop length.
    Unlike the original version, it does not require hop_length to divide len(window).
    """
    win_len = len(window)
    max_offset = win_len + hop_length
    den = np.zeros(max_offset)

    for shift in range(0, max_offset, hop_length):
        start = shift
        end = shift + win_len
        den[start:end] += window**2

    return den[:win_len]

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
    """
    Perform Harmonic/Percussive source separation using median filtering.
    Inputs
    x: input audio signal
    sr: sampling rate
    fft_size: size of the FFT window
    hop_length: hop size for STFT
    lh: half-length of the median filter for harmonic components (in frames)
    lp: half-length of the median filter for percussive components (in frequency bins)

    Outputs
    xh: harmonic component of the signal
    xp: percussive component of the signal
    Xh: STFT of the harmonic component
    Xp: STFT of the percussive component"""

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

def float2pcm(sig, dtype='int16'):
    """Convert floating point signal to PCM format.
    Inputs:
    sig: floating point input signal (numpy array)
    dtype: target PCM format (default: int16)
    Outputs:
    sig_pcm: PCM signal (numpy array)"""
    
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def on_alpha_change(e):
    '''Callback function to handle alpha changes via keyboard input.'''
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.01
    elif e.name == 'down' and alpha > 0.10:
        alpha -=0.01
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

keyboard.on_press(on_alpha_change)

def manual_stft_numpy(xh, beta, L=2048, sr=22050):
    """Manual STFT implementation using NumPy.
    Inputs:
    xh: input signal
    beta: overlap factor (e.g., 0.25 for 75% overlap)
    L: FFT size (default: 2048)
    sr: sampling rate (default: 22050)
    
    Outputs:
    S_lookup: STFT matrix (numpy array)"""

    Ha_lookup = int(round(beta * L))
    window = scipy.signal.windows.hann(L, sym=False)
    n_frames = int(np.round((len(xh) - L) / Ha_lookup))
    k_bins = 1 + L // 2
    S_lookup = np.zeros((k_bins, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * Ha_lookup
        end = start + L
        if end > len(xh):
            break
        S_lookup[:, i] = np.fft.rfft(xh[start:end] * window)
    return S_lookup

file_name = sys.argv[1]
audio_data, audio_sr = lb.load(file_name)

# Constants
CHUNK = L = 2048
L_ola = 256
Hs = L // 4
Hs_ola = L_ola // 2
alpha = 1.00
window = scipy.signal.windows.hann(L, sym=False)
window_ola = scipy.signal.windows.hann(L_ola, sym=False)
output_buffer = np.zeros(int(L))
normalization = np.zeros(int(L))
prev_fft = None
prev_phase = np.zeros(L//2 + 1)
runtimes = []
pos = 0
pos_ola = 0
omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L

# Determines the phase vocoder look-up analysis hopsize e.g. beta = 0.125 is 12.5% overlap
beta = 0.25

# Seperate harmonic and percussive components
xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)

#Phase vocoder Look-up
Ha_lookup = int(round((beta)*L))
den = calc_sum_squared_window(window, Hs)

S_lookup = manual_stft_numpy(xh, beta)
S_phase_lookup = np.angle(S_lookup)
S_mag_lookup = np.abs(S_lookup)

w_if_lookup = estimateIF(S_lookup, audio_sr, Ha_lookup)
prev_phase = None
ratio = Hs//Hs_ola

# To save audio file
output_filename = f"output/output_{beta}.wav"  # Name of the output file
output_frames = []  # List to store audio frames

p = pyaudio.PyAudio()

print("Playing audio. Press:")
print("- UP arrow to increase stretch factor")
print("- DOWN arrow to decrease stretch factor")
print("- CTRL+C to stop")
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=audio_sr,
                output=True,
                frames_per_buffer=512)

try:
    while pos <= len(xh) - L:
        Ha = int(round(Hs/alpha))
        Ha_ola = int(round(Hs_ola/alpha))
        
        if pos == 0:
            prev_phase = S_phase_lookup[:, 0]
            S_mod = S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
        else:
            #Get FRAME index (current frame)
            frame_idx = min(int(round(pos / Ha_lookup)), S_mag_lookup.shape[1] - 1)
            
            # Get PHASE TRANSITION index (previous to current frame)
            phase_trans_idx = min(int(round((pos - Ha) / Ha_lookup)), w_if_lookup.shape[1] - 1)
            
            phase_increment = w_if_lookup[:, phase_trans_idx] * (Hs / audio_sr)
            prev_phase += phase_increment  # Update phase correctly for current alpha
            S_mod = S_mag_lookup[:, frame_idx] * np.exp(1j * prev_phase) # Modified STFT frame

        pv_frame_mod = np.fft.irfft(S_mod) # Inverse FFT to get time-domain frame

        # Updates the buffer by shifting and adding the new frame
        output_buffer[:-Hs] = output_buffer[Hs:]
        output_buffer[-Hs:] = 0
        output_buffer += pv_frame_mod * ((window.reshape((-1,1))/(den.reshape((-1,1)))).flatten())

        # OLA for percussive component
        for i in range(ratio):
            ola_win_synth = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola] * window_ola
            offset = i * Hs_ola
            output_buffer[offset:offset + L_ola] += ola_win_synth
            

        output_buffer = np.clip(output_buffer, -1.0, 1.0)  # Float32 clipping
        stream.write(float2pcm(output_buffer[:Hs]).astype(np.int16).tobytes())  # Convert to int16 at last moment

        # Store for WAV file
        output_frames.append(float2pcm(output_buffer[:Hs]).astype(np.int16).copy()) 

        pos += Ha

except KeyboardInterrupt:
    print("\nStream stopped by user!")
stream.stop_stream()
stream.close()
p.terminate

# Save to WAV file if we captured any audio
if output_frames:
    # Concatenate all frames
    full_audio = np.concatenate(output_frames)
    
    # Save as WAV
    wavfile.write(output_filename, audio_sr, full_audio)
    print(f"\nSaved processed audio to {output_filename}")
