import wave
import sys
import pyaudio
import numpy as np
import keyboard 

CHUNK = 1024
volume = 1.0  # Initial volume (1.0 = 100%)

def on_volume_change(e):
    global volume
    if e.name == 'up' and volume < 2.0:  # Max 2.0 (200% volume)
        volume += 0.1
    elif e.name == 'down' and volume > 0.0:  # Min 0.0 (mute)
        volume -= 0.1
    print(f"\rCurrent volume: {volume:.1f}", end="", flush=True)

# Set up keyboard listeners
keyboard.on_press(on_volume_change)

if len(sys.argv) < 2:
    print(f'Plays a wave file. Usage: {sys.argv[0]} filename.wav')
    sys.exit(-1)

with wave.open(sys.argv[1], 'rb') as wf:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    print("Playing audio. Press:")
    print("- UP arrow to increase volume")
    print("- DOWN arrow to decrease volume")
    print("- CTRL+C to stop")

    try:
        while len(data := wf.readframes(CHUNK)):
            # Convert bytes to numpy array and adjust volume
            audio_array = np.frombuffer(data, dtype=np.int16)
            adjusted = np.clip(audio_array * volume, -32768, 32767).astype(np.int16) # -32768 to 32767 are the split range of int16's possible values
            stream.write(adjusted.tobytes())
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.close()
        p.terminate()
        keyboard.unhook_all()  # Clean up keyboard listeners