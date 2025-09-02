import wave
import sys
import pyaudio
import numpy as np
import keyboard
import librosa as lb

def float2pcm(sig, dtype='int16'):
    # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


CHUNK = L = 2048
Hs = L // 4
window = np.hanning(L)
output_buffer = np.zeros(L)
alpha = 1.0

prev_fft = None
prev_phase = np.zeros(L//2 + 1)

def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.0:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.10:
        alpha -= 0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

keyboard.on_press(on_alpha_change)

file_name = sys.argv[1]
audio_data, audio_sr = lb.load(file_name)
omega_nom = np.arange(L//2 + 1) * 2 * np.pi * audio_sr / L  # update with real sample rate
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=audio_sr,
                output=True,
                frames_per_buffer=512)

print("Playing audio (Phase Vocoder). Press:")
print("- UP arrow to increase stretch factor")
print("- DOWN arrow to decrease stretch factor")
print("- CTRL+C to stop")

num_samples = len(audio_data)
pos = 0

try:
    pos = 0
    while pos <= num_samples - CHUNK:
        x = audio_data[pos:pos + L]
        Ha = int(np.round(Hs / alpha))
        
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)))  # zero-pad if too short
        
        frame = x[:L] * window
        S = np.fft.rfft(frame)

        # Phase Vocoder analysis
        if prev_fft is None:
            w_if = np.zeros_like(omega_nom)
        else:
            dphi = np.angle(S) - np.angle(prev_fft)
            dphi = dphi - omega_nom * (Ha / audio_sr)
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            w_if = omega_nom + dphi * (audio_sr / Ha)

        prev_phase = prev_phase + w_if * (Hs / audio_sr)

        X_mod = np.abs(S) * np.exp(1j * prev_phase)
        frame_mod = np.fft.irfft(X_mod)

        # Overlap-add
        output_buffer[:-Hs] = output_buffer[Hs:]  # shift left
        output_buffer[-Hs:] = 0
        output_buffer += frame_mod * window  # apply synthesis window

        # Output
        output_int16 = np.clip(output_buffer[:Hs], -1.0, 1.0)
        stream.write(float2pcm(output_int16).tobytes())

        prev_fft = S
        pos += Ha

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    stream.close()
    p.terminate()
    keyboard.unhook_all()
