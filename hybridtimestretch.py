import wave
import sys
import pyaudio
import numpy as np
import keyboard
from scipy.signal import medfilt

# Parameters
L = 1024  # Frame size
Hs = L // 4  # Hop size
alpha = 1.0  # Time-stretch factor

# Window
window = np.hanning(L)

# State variables
prev_fft = None
prev_phase = np.zeros(L//2 + 1)
output_buffer = np.zeros(L * 2)

def hpss(x, L=1024, Hs=256, sr=44100, harm_filter=31, perc_filter=21):
    """Improved HPSS implementation"""
    # STFT
    X = np.fft.rfft(x * window)
    mag = np.abs(X)
    
    # Create spectrogram context
    spectrogram = np.vstack([mag]*5)  # Simulate context with current frame
    
    # Median filtering
    harm_mag = medfilt(spectrogram, (1, harm_filter))
    perc_mag = medfilt(spectrogram, (perc_filter, 1))
    
    # Binary masks from center frame
    center = 2
    Mh = (harm_mag[center] > perc_mag[center]).astype(float)
    Mp = 1 - Mh
    
    return X * Mh, X * Mp

def phase_vocoder(X, prev_fft, prev_phase, Hs, alpha, sr, omega_nom):
    """Phase vocoder processing"""
    if prev_fft is None:
        return np.angle(X), X
    
    dphi = np.angle(X) - np.angle(prev_fft)
    dphi = dphi - omega_nom * (Hs/alpha / sr)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    w_if = omega_nom + dphi * (sr / (Hs/alpha))
    phase = prev_phase + w_if * (Hs / sr)
    
    return phase, X

def process_frame(x, prev_fft, prev_phase, Hs, alpha, sr, omega_nom):
    """Process a single frame"""
    # HPSS separation
    Xh, Xp = hpss(x)
    
    # Phase vocoder for harmonic
    phase, prev_fft = phase_vocoder(Xh, prev_fft, prev_phase, Hs, alpha, sr, omega_nom)
    Xh_mod = np.abs(Xh) * np.exp(1j * phase)
    xh = np.fft.irfft(Xh_mod).real * window
    
    # Pure OLA for percussive
    xp = np.fft.irfft(Xp).real * window
    
    return xh, xp, prev_fft, phase

def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.10:
        alpha -= 0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

def main():
    global alpha
    
    p = None
    stream = None
    
    try:
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
            
            print("Beatbox-optimized Hybrid Time Stretching")
            print("UP/DOWN arrows to adjust speed, CTRL+C to stop")
            
            keyboard.on_press(on_alpha_change)
            
            pos = 0
            num_samples = wf.getnframes()
            
            while pos <= num_samples - L:
                # Read and prepare frame
                wf.setpos(pos)
                data = wf.readframes(L)
                x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if len(x) < L:
                    x = np.pad(x, (0, L - len(x)))
                
                Ha = int(np.round(Hs / alpha))
                
                # Process frame
                xh, xp, prev_fft, prev_phase = process_frame(
                    x, prev_fft, prev_phase, Hs, alpha, sr, omega_nom
                )
                
                # Overlap-add
                output_buffer[:-Hs] = output_buffer[Hs:]
                output_buffer[-Hs:] = 0
                output_buffer[:L] += (0.5 * xh + 1.5 * xp)  # Emphasize percussive
                
                # Output
                stream.write(
                    np.clip(output_buffer[:Hs], -32768, 32767).astype(np.int16).tobytes()
                )
                
                pos += Ha
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        # Proper cleanup
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()
        keyboard.unhook_all()

if __name__ == "__main__":
    main()