import wave
import sys
import pyaudio
import numpy as np
import keyboard 
import librosa as lb
import scipy.io as sio


CHUNK = 1024
volume = 1.0  # Initial volume (1.0 = 100%)
# count = 0
def on_volume_change(e):
    global volume
    if e.name == 'up' and volume < 2.0:  # Max 2.0 (200% volume)
        volume += 0.1
    elif e.name == 'down' and volume > 0.1:  # Min 0.0 (mute)
        volume -= 0.1
    print(f"\rCurrent volume: {volume:.1f}", end="", flush=True)

def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


# Set up keyboard listeners
keyboard.on_press(on_volume_change)

if len(sys.argv) < 2:
    print(f'Plays a wave file. Usage: {sys.argv[0]} filename.wav')
    sys.exit(-1)



# try:
#     wf = wave.open(sys.argv[1], 'rb')
# except:  # Catch absolutely everything
#     import sys
#     exc_type, exc_value, exc_traceback = sys.exc_info()
#     print(f"EXACT ERROR: {exc_type.__name__} - {exc_value}")
#     raise
# except wave.Error as e:
#     if 'unknown format' in str(e):
#         format_tag = str(e).split(':')[-1].strip()  # Extract the format number
#         print(f"Unsupported wave format detected: {format_tag}")
#         data, data_sr = lb.load(sys.argv[1])
#         data_int = [float2pcm(e) for e in data]
#         sio.wavfile.write(str(sys.argv[1]) + "_int.wav", data_sr, data_int)
#         wf = wave.open(str(sys.argv[1]) + "_int.wav", 'rb')


# p = pyaudio.PyAudio()
# stream = p.open(
#     format=p.get_format_from_width(wf.getsampwidth()),
#     channels=wf.getnchannels(),
#     rate=wf.getframerate(),
#     output=True
# )

# print("Playing audio. Press:")
# print("- UP arrow to increase volume")
# print("- DOWN arrow to decrease volume")
# print("- CTRL+C to stop")

# total_frames = wf.getnframes()
# pos = 0
# try:
#     while pos <= total_frames - CHUNK:
#         # count += 1
#         wf.setpos(pos)
#         data = wf.readframes(CHUNK)
#         # Convert bytes to numpy array and adjust volume
#         audio_array = np.frombuffer(data, dtype=np.int16)
#         adjusted = (audio_array * volume).astype(np.int16)
#         stream.write(adjusted.tobytes())
#         pos += CHUNK
#         # print(count)
# except KeyboardInterrupt:
#     print("\nStopped by user")
# finally:
#     stream.close()
#     p.terminate()
#     keyboard.unhook_all()  # Clean up keyboard listeners


