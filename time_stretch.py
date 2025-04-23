import wave
import sys
import pyaudio
import numpy as np
import keyboard 

CHUNK = L = 2048
Hs = L // 2
alpha = 1.0
window = np.hanning(L)
output_buffer = None

def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.05
    elif e.name == 'down' and alpha > 0.10:
        alpha -=0.05
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)

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
    num_samples = wf.getnframes()
    pos = 0  # Start at beginning

    try:
        while pos <= num_samples - CHUNK:
            # Move file pointer to current position
            wf.setpos(pos)
            
            # Read CHUNK frames (with overlap)
            data = wf.readframes(CHUNK)
            
            x = np.frombuffer(data, dtype=np.int16)
            
            Ha = int(np.round(Hs/alpha))

            # 3. Apply OLA
            analysis_buffer = x * window
            synthesis_buffer = analysis_buffer
            
            numFrames = (num_samples - L) // Ha + 1
            output_buffer = np.zeros(Hs * (numFrames-1) + L)

            # 4. Overlap-add to output
            output_buffer[:-Hs] = output_buffer[Hs:]  # Shift buffer (first half = the second half)
            output_buffer[-Hs:] = 0
            output_buffer[:L] += synthesis_buffer

            # 5. Stream the output
            stream.write(output_buffer[:Hs].astype(np.int16).tobytes())

            # Advance by analysis hop size
            pos += Ha
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.close()
        p.terminate()
        keyboard.unhook_all()
