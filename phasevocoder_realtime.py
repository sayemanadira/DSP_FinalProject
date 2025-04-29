import wave
import sys
import pyaudio
import numpy as np
import keyboard

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

with wave.open(sys.argv[1], 'rb') as wf:
    sr = wf.getframerate()
    omega_nom = np.arange(L//2 + 1) * 2 * np.pi * sr / L  # update with real sample rate
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=sr,
        output=True
    )
    print("Playing audio (Phase Vocoder). Press:")
    print("- UP arrow to increase stretch factor")
    print("- DOWN arrow to decrease stretch factor")
    print("- CTRL+C to stop")

    num_samples = wf.getnframes()
    pos = 0

    try:
        while pos <= num_samples - CHUNK:
            wf.setpos(pos)
            data = wf.readframes(CHUNK)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)

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
                dphi = dphi - omega_nom * (Ha / sr)
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                w_if = omega_nom + dphi * (sr / Ha)

            prev_phase = prev_phase + w_if * (Hs / sr)

            X_mod = np.abs(S) * np.exp(1j * prev_phase)
            frame_mod = np.fft.irfft(X_mod)

            # Overlap-add
            output_buffer[:-Hs] = output_buffer[Hs:]  # shift left
            output_buffer[-Hs:] = 0
            output_buffer += frame_mod * window  # apply synthesis window

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
