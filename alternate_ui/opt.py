import wave
import sys
import pyaudio
import numpy as np
import librosa as lb
from scipy.signal import medfilt
import tkinter as tk
from tkinter import ttk
import threading

# Constants
CHUNK = L = 2048
L_ola = 256
Hs = L // 4
Hs_ola = L_ola // 2
window = np.hanning(L)
output_buffer = np.zeros(int(L))
prev_phase = np.zeros(L//2 + 1)

# Global variables
processing_active = False
stream = None
pos = 0
prev_fft = None
current_min_alpha = 5.0  # Default value
current_alpha = 1.0      # Default playback speed
audio_ended = False

def calc_sum_squared_window(window, hop_length):
    assert (len(window) % hop_length == 0), "Hop length does not divide the window evenly."
    
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i*hop_length)
        
    return den

def estimateIF(S, sr, hop_samples):
    hop_sec = hop_samples / sr
    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1,1))    
    unwrapped = np.angle(S[:,1:]) - np.angle(S[:,0:-1]) - w_nom * hop_sec
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / hop_sec
    return w_if

def invert_stft(S, hop_length, window):
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
    frames = np.real(frames)
    
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

def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# Load audio file
file_name = sys.argv[1]
audio_data, audio_sr = lb.load(file_name)

# Process audio
xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)

if max(abs(xh)) > 1:
    xh = xh / max(abs(xh))
elif max(abs(xp)) > 1:
    xp = xp / max(abs(xp))

xh = float2pcm(xh).astype(np.int16)
xp = float2pcm(xp).astype(np.int16)

omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L
den = calc_sum_squared_window(window, Hs)

# Prepare lookup tables
min_alpha = 5  # Starting value
min_Ha = int(Hs / min_alpha)
S_lookup = lb.core.stft(audio_data, n_fft=L, hop_length=min_Ha, center=False)
S_phase_lookup = np.angle(S_lookup)
S_mag_lookup = np.abs(S_lookup)
w_if_lookup = estimateIF(S_lookup, audio_sr, min_Ha)

def audio_processing_thread():
    global pos, prev_fft, prev_phase, output_buffer, processing_active, current_min_alpha, current_alpha, audio_ended
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=audio_sr,
                    output=True)
    
    audio_ended = False
    
    while processing_active and not audio_ended:
        if pos <= len(xh) - L:
            Ha = int(Hs/current_alpha)
            Ha_ola = int(Hs_ola/current_alpha)
            

            if pos == 0:
                prev_phase = S_phase_lookup[:, 0]
                S_mod = S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
            else:
                curr_frame = int(round(pos / min_Ha))  # lookup is always based on min_Ha
                phase_increment = w_if_lookup[:, curr_frame - 1] * (Hs / (current_alpha * audio_sr))
                prev_phase += phase_increment  # Update phase correctly for current alpha
                S_mod = S_mag_lookup[:, curr_frame] * np.exp(1j * prev_phase)

            pv_frame_mod = np.fft.irfft(S_mod)
            

        
            # pv_frame_mod = np.fft.irfft(X_mod)
            pv_frame_mod = float2pcm(np.array(pv_frame_mod))

            
            # overlap-add to output buffer
            output_buffer[:-Hs] = output_buffer[Hs:]
            output_buffer[-Hs:] = 0
            output_buffer += pv_frame_mod * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()

            ratio = Hs//Hs_ola
            ola_y = np.zeros(L)
            for i in range(ratio):
                ola_win = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola]
                ola_win_synth = ola_win * np.hanning(L_ola)
                offset = i * Hs_ola
                ola_y[offset:offset + L_ola] += ola_win_synth
        
            output_buffer += ola_y

            output_buffer = np.clip(output_buffer, -32768, 32767)  # 16-bit range
            # runtimes.append(end_time - start_time)
            stream.write(output_buffer[:Hs].astype(np.int16).tobytes())
            # print(pos//Ha)
            # prev_fft = S
            pos += Ha
        else:
            # Reached end of audio
            audio_ended = True
    
    # Clean up when processing stops
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if audio_ended:
        # Update UI on main thread
        root.after(0, playback_ended)

def playback_ended():
    global processing_active
    processing_active = False
    play_button.config(text="Play")
    status_label.config(text="Playback completed")

def update_min_alpha(val):
    global current_min_alpha
    current_min_alpha = float(val)
    min_alpha_label.config(text=f"H_(a, precomp): {int(Hs/current_min_alpha):.2f}")

def update_alpha(val):
    global current_alpha
    current_alpha = float(val)
    alpha_label.config(text=f"Speed: {np.log(current_alpha):.2f}x")

def toggle_playback():
    global processing_active, pos, prev_phase, output_buffer, prev_fft, audio_ended
    
    if not processing_active:
        # Start playback
        processing_active = True
        audio_ended = False
        play_button.config(text="Stop")
        status_label.config(text="Playing...")
        
        # Reset audio position and buffers
        pos = 0
        prev_phase = np.zeros(L//2 + 1)
        output_buffer = np.zeros(L)
        prev_fft = None
        
        # Start processing thread
        audio_thread = threading.Thread(target=audio_processing_thread)
        audio_thread.daemon = True
        audio_thread.start()
    else:
        # Stop playback
        processing_active = False
        play_button.config(text="Play")
        status_label.config(text="Stopped")

def on_closing():
    global processing_active
    processing_active = False
    root.destroy()

# Create the GUI
root = tk.Tk()
root.title("Audio Time-Scale Modification")

# Play/Stop button
play_button = tk.Button(root, text="Play", command=toggle_playback, font=('Arial', 14), width=10)
play_button.pack(pady=10)

# Status label
status_label = tk.Label(root, text="Ready", font=('Arial', 12))
status_label.pack(pady=5)

# Time resolution slider
min_alpha_frame = tk.Frame(root)
min_alpha_frame.pack(pady=5, fill=tk.X, padx=10)

min_alpha_label = tk.Label(min_alpha_frame, text="Time Resolution: 5.00")
min_alpha_label.pack(side=tk.TOP, anchor=tk.W)

min_alpha_slider = ttk.Scale(min_alpha_frame, from_=1.0, to=15.00, orient=tk.HORIZONTAL, 
                            command=update_min_alpha)
min_alpha_slider.set(5.0)
min_alpha_slider.pack(fill=tk.X)

# Playback speed slider
alpha_frame = tk.Frame(root)
alpha_frame.pack(pady=5, fill=tk.X, padx=10)

alpha_label = tk.Label(alpha_frame, text="Playback Speed: 1.00x")
alpha_label.pack(side=tk.TOP, anchor=tk.W)

alpha_slider = ttk.Scale(alpha_frame, from_=0.3, to=2, orient=tk.HORIZONTAL, 
                        command=update_alpha)
alpha_slider.set(1.0)
alpha_slider.pack(fill=tk.X)

# Handle window closing
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()