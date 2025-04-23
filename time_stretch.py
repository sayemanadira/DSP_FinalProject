import wave
import sys
import pyaudio
import numpy as np
import keyboard 

L = 2048
CHUNK = L
HOP = L // 2
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
    num_frames = wf.getnframes()
    frame_rate = wf.getframerate()
    duration = num_frames / frame_rate

    print("Playing audio. Press:")
    print("- UP arrow to increase stretch factor")
    print("- DOWN arrow to decrease stretch factor")
    print("- CTRL+C to stop")

    # Get total frames and calculate positions
    total_frames = wf.getnframes()
    pos = 0  # Start at beginning
    output_buffer = np.zeros(int(HOP * duration))

    try:
        while pos <= total_frames - CHUNK:
            # Move file pointer to current position
            wf.setpos(pos)
            
            # Read CHUNK frames (with overlap)
            data = wf.readframes(CHUNK)
            
            # Process data (e.g., apply FFT, volume, effects)
            x = np.frombuffer(data, dtype=np.int16)
            
            Ha = int(np.round(HOP/alpha))

            # 3. Apply OLA
            analysis_buffer = x * window
            synthesis_buffer = analysis_buffer  # (Add your processing here)
            
            # 4. Overlap-add to output
            output_buffer[:-HOP] = output_buffer[HOP:]  # Shift buffer
            output_buffer[-HOP:] = 0
            output_buffer[:L] += synthesis_buffer


            # 5. Stream the output
            stream.write(output_buffer[:HOP].astype(np.int16).tobytes())

            # Advance by analysis hop size
            pos += Ha
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.close()
        p.terminate()
        keyboard.unhook_all()
