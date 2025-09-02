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


CHUNK = L = 2048
L_ola = 256
Hs = L // 4
Hs_ola = L_ola // 2
alpha = 1.00
window = scipy.signal.windows.hann(L, sym=False)
window_ola = scipy.signal.windows.hann(L_ola, sym=False)
output_buffer = np.zeros(int(L))
prev_fft = None
prev_phase = np.zeros(L//2 + 1)
runtimes = []


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

def float2pcm(sig, dtype='int16'):
    # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# file_name = 'samples/fred_10sec.wav'
file_name = sys.argv[1]
audio_data, audio_sr = lb.load(file_name)


xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)

omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L
den = calc_sum_squared_window(window, Hs)
# norm = (window / np.sqrt(np.mean(window**2)))



def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.01
    elif e.name == 'down' and alpha > 0.10:
        alpha -=0.01
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

keyboard.on_press(on_alpha_change)

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

pos = 0
pos_ola = 0

saved_frames = []
Ha = 0

ratio = Hs//Hs_ola

try:
    while pos <= len(xh) - L:
        Ha = int(round(Hs/alpha))
        Ha_ola = int(round(Hs_ola/alpha))
        
        # start_time = time.perf_counter()

        # Phase Vocoder       
        # STFT processing
        pv_win = xh[pos:pos+L]
        S = np.fft.rfft(pv_win)
        magnitude = np.abs(S)
        if prev_fft is not None:
            dphi = np.angle(S) - np.angle(prev_fft)
            dphi = dphi - omega_nom * (Ha/audio_sr)
            dphi = (dphi + np.pi) % (2*np.pi) - np.pi
            w_if = omega_nom + dphi * (audio_sr/Ha)
            prev_phase += w_if * (Hs/audio_sr)
        else:
            prev_phase = np.angle(S)
        
        X_mod = magnitude * np.exp(1j * prev_phase)
        pv_frame_mod = np.fft.irfft(X_mod)

        #shift and add to stream
        output_buffer[:-Hs] = output_buffer[Hs:]
        output_buffer[-Hs:] = 0
        output_buffer += pv_frame_mod * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
        # output_buffer += pv_frame_mod * norm


        for i in range(ratio):
            ola_win_synth = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola] * window_ola
            offset = i * Hs_ola
            output_buffer[offset:offset + L_ola] += ola_win_synth
    
        output_buffer = np.clip(output_buffer, -1.0, 1.0)  # FLoat range
        stream.write(float2pcm(output_buffer[:Hs]).astype(np.int16).tobytes())

        # To save
        saved_frames.append(output_buffer[:Hs].copy())

        # print(output_buffer)
        prev_fft = S
        pos += Ha

except KeyboardInterrupt:
    print("\nStream stopped by user!")
stream.stop_stream()
stream.close()
p.terminate

saving_filepath = 'output/'
if saved_frames:
    # Concatenate all frames
    full_audio = np.concatenate(saved_frames)
    # Ensure audio is in range [-1, 1]
    full_audio = np.clip(full_audio, -1.0, 1.0)
    # Convert to int16 for WAV file
    full_audio_int16 = float2pcm(full_audio, dtype='int16')
    # Save as WAV
    wavfile.write(saving_filepath + "realtime_test.wav", audio_sr, full_audio_int16)
    print(f"\nSaved processed audio to {saving_filepath}")
