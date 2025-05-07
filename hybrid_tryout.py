import wave
import sys
import pyaudio
import numpy as np
import keyboard
from scipy.signal import medfilt
import librosa as lb

# Parameters
L = 2048  # Frame size
Hs = L // 4  # Synthesis hop
Ha_ola = L // 2  # Analysis hop for OLA
alpha = 1.0  # Initial stretch factor
window = np.hanning(L)

# Buffers
output_buffer = np.zeros(L * 2, dtype=np.float32)

# Phase vocoder state
prev_phase = np.zeros(L//2 + 1)
phase_accum = np.zeros(L//2 + 1)
prev_mag = None

def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.0:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.1:
        alpha -= 0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

def process_harmonic(x):
    """Phase vocoder processing for harmonic components"""
    global prev_phase, phase_accum, prev_mag
    
    # STFT with librosa for proper handling
    S = lb.stft(np.array(x), n_fft=L, hop_length=Hs, win_length=L, window='hann')
    mag = np.abs(S)
    phase = np.angle(S)
    
    if prev_mag is not None:
        # Phase difference calculation
        delta_phase = phase - prev_phase
        omega = 2 * np.pi * Hs * np.arange(L//2 + 1) / L
        delta_phase = delta_phase - omega[:, np.newaxis]  # Add dimension for broadcasting
        delta_phase = np.mod(delta_phase + np.pi, 2*np.pi) - np.pi
        phase_accum += delta_phase * alpha
    
    stretched = mag * np.exp(1j * phase_accum)
    prev_phase = phase
    prev_mag = mag
    return lb.istft(stretched, hop_length=Hs, win_length=L, window='hann')

def process_percussive(x):
    """Simplified OLA processing for percussive components"""
    if alpha != 1.0:
        # Resample for time-stretching
        target_length = int(len(x) * alpha)
        x_resampled = lb.resample(x, orig_sr=len(x), target_sr=target_length)
        if len(x_resampled) > L:
            x_resampled = x_resampled[:L]
        elif len(x_resampled) < L:
            x_resampled = np.pad(x_resampled, (0, L - len(x_resampled)))
        return x_resampled * window
    return x * window

def hpss(x):
    """Real-time HPSS with proper dimension handling"""
    S = lb.stft(x, n_fft=L, hop_length=Hs, win_length=L, window='hann')
    mag = np.abs(S)
    
    # Ensure kernel size is smaller than spectrogram
    kernel_size = min(13, mag.shape[1] - 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1  # Ensure odd kernel size
    
    H = medfilt(mag, (1, kernel_size)) * np.exp(1j * np.angle(S))
    P = S - H
    return lb.istft(H)[0], lb.istft(P)[0]

keyboard.on_press(on_alpha_change)

with wave.open(sys.argv[1], 'rb') as wf:
    sr = wf.getframerate()
    pya = pyaudio.PyAudio()
    
    stream = pya.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sr,
                    output=True,
                    frames_per_buffer=Hs)

    print("Hybrid OLA/Phase Vocoder. Press:")
    print("- UP/DOWN arrows to adjust stretch factor")
    print("- CTRL+C to stop")

    pos = 0
    try:
        while pos <= wf.getnframes() - L:
            wf.setpos(pos)
            data = wf.readframes(L)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            x /= 32768.0  # Normalize to [-1, 1]
            
            # Separation
            h, p = hpss(x)
            
            # Process components
            harmonic = process_harmonic(h)
            percussive = process_percussive(p)
            
            # Overlap-add to output
            output_buffer[:-Hs] = output_buffer[Hs:]
            output_buffer[-Hs:] = 0
            output_buffer[:len(harmonic)] += harmonic
            output_buffer[:len(percussive)] += percussive
            
            # Output
            stream.write(output_buffer[:Hs].astype(np.float32).tobytes())
            
            pos += int(Ha_ola / alpha)
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        pya.terminate()
        keyboard.unhook_all()