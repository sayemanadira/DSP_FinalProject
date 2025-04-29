import wave
import sys
import pyaudio
import numpy as np
import keyboard 
import librosa as lb
from scipy.signal import medfilt


CHUNK = L = 2048
L_ola = 256
Hs = L // 4
Hs_ola = L // 2
alpha = 1.0
window = np.hanning(L)
output_buffer = np.zeros(int(L))
prev_fft = None
prev_phase = np.zeros(L//2 + 1)


def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.10:
        alpha -=0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

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
    num = window.reshape((-1,1))
    den = calc_sum_squared_window(window, hop_length)
    #den = np.square(window) + np.square(np.roll(window, hop_length))
    frames = frames * window.reshape((-1,1)) / den.reshape((-1,1))
    #frames = frames * window.reshape((-1,1))
    
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

keyboard.on_press(on_alpha_change)

with wave.open(sys.argv[1], 'rb') as wf:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    print("Playing audio. Press:")
    print("- UP arrow to increase stretch factor")
    print("- DOWN arrow to decrease stretch factor")
    print("- CTRL+C to stop")

    # Get total frames and calculate positions
    sr = wf.getframerate()
    num_samples = wf.getnframes()
    omega_nom = np.arange(L//2 + 1) * 2 * np.pi * sr / L  # update with real sample rate
    pos_PV = 0  # Start at beginning
    pos_OLA = 0
    analysis_frames_ola = []
    try:
        while pos_PV <= num_samples - CHUNK:
            wf.setpos(pos_PV)
            # Read CHUNK frames (with overlap)
            data = wf.readframes(CHUNK)
            xh, xp, _, _ = harmonic_percussive_separation(data)
            Ha = int(Hs/alpha)

            #OLA within the working window
            while pos_OLA <= L - L_ola:
                ola_Ha = int(Hs_ola/alpha)
                frame = xp[pos_OLA:pos_OLA+L_ola]
                analysis_frames_ola.append(frame)
                pos_OLA += ola_Ha
            
            if len(x) < L:
                x = np.pad(x, (0, L - len(x)))  # zero-pad if too short
            
            frame = x[:L] * window
            S = np.fft.rfft(frame)

            # Phase Vocoder analysis
            if prev_fft is None:
                w_if = np.zeros_like(omega_nom)
            else:
                dphi = np.angle(S) - np.angle(prev_fft)
                dphi = dphi - omega_nom * (Ha / sr)
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                w_if = omega_nom + dphi * (sr / Ha)

            prev_phase = prev_phase + w_if * (Hs / sr)

            X_mod = np.abs(S) * np.exp(1j * prev_phase)
            frame_mod = np.fft.irfft(X_mod)

            # getting synthesis frames for OLA
            systhesis_frames_ola = np.array(analysis_frames_ola).T * np.hann(L_ola).reshape((-1,1))

            # Overlap-add
            output_buffer[:-Hs] = output_buffer[Hs:]  # shift left
            output_buffer[-Hs:] = 0
            output_buffer += (frame_mod * window + systhesis_frames_ola)  # apply synthesis window

            # Output
            output_int16 = np.clip(output_buffer[:Hs], -32768, 32767).astype(np.int16)
            stream.write(output_int16.tobytes())

            prev_fft = S
            pos += Ha
            
            

            

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.close()
        p.terminate()
        keyboard.unhook_all()