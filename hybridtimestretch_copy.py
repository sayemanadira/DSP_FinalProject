import wave
import sys
import pyaudio
import numpy as np
import keyboard
from scipy.signal import medfilt

# Parameters
L = 2048  # Frame size
Hs = L // 4  # Synthesis hop size (512)
alpha = 1.0  # Time-stretch factor

# Window functions
window = np.hanning(L)

# State variables
prev_fft = None
prev_phase = np.zeros(L // 2 + 1)
output_buffer = np.zeros(L + Hs)  # Buffer needs to be L + Hs

def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.10:
        alpha -= 0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

keyboard.on_press(on_alpha_change)

with wave.open(sys.argv[1], 'rb') as wf:
    sr = wf.getframerate()
    omega_nom = np.arange(L//2 + 1) * 2 * np.pi * sr / L
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=sr,
        output=True
    )
    
    print("Playing audio (Hybrid PV + OLA). Press:")
    print("- UP arrow to increase stretch factor")
    print("- DOWN arrow to decrease stretch factor")
    print("- CTRL+C to stop")

    num_samples = wf.getnframes()
    pos = 0

    try:
        while pos <= num_samples - L:
            wf.setpos(pos)
            data = wf.readframes(L)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            
            if len(x) < L:
                x = np.pad(x, (0, L - len(x)))
            
            Ha = int(np.round(Hs / alpha))  # Analysis hop size
            
            # Apply window
            frame = x * window
            
            # Compute STFT
            S = np.fft.rfft(frame)
            mag = np.abs(S)
            
            # --- Harmonic/Percussive Separation ---
            # Simple median filtering approach (like HPSS)
            mag_2d = mag.reshape(1, -1)  # Make it 2D for medfilt
            harm_mag = medfilt(mag_2d, (1, 17))  # Horizontal median filter (harmonic)
            perc_mag = medfilt(mag_2d, (9, 1))   # Vertical median filter (percussive)
            
            # Create masks (soft masks for smoother transitions)
            harm_mask = harm_mag / (harm_mag + perc_mag + 1e-10)
            perc_mask = 1 - harm_mask
            
            # --- Phase Vocoder for Harmonic Components ---
            if prev_fft is None:
                w_if = np.zeros_like(omega_nom)
            else:
                dphi = np.angle(S) - np.angle(prev_fft)
                dphi = dphi - omega_nom * (Ha / sr)
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                w_if = omega_nom + dphi * (sr / Ha)
            
            prev_phase = prev_phase + w_if * (Hs / sr)
            
            # Apply harmonic mask and phase vocoder
            X_harm = (mag * harm_mask) * np.exp(1j * prev_phase)
            
            # --- OLA for Percussive Components ---
            # Use original phase for percussive components
            X_perc = (mag * perc_mask) * np.exp(1j * np.angle(S))
            
            # Combine both components
            X_combined = X_harm + X_perc
            
            # Inverse STFT
            frame_out = np.fft.irfft(X_combined) * window
            
            # Overlap-add - CORRECTED VERSION
            # Shift the buffer left by Hs samples
            output_buffer[:-Hs] = output_buffer[Hs:]
            # Zero out the rightmost Hs samples
            output_buffer[-Hs:] = 0
            # Add the new frame
            output_buffer[:L] += frame_out
            
            # Output the first Hs samples
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