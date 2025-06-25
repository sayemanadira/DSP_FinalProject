import wave
import sys
import pyaudio
import numpy as np
import keyboard 
import librosa as lb
from scipy.signal import medfilt
import scipy.io.wavfile as wavfile
import time
import csv
import cProfile
import pstats
import io
import time

# Constants and initialization (keep your existing setup)
CHUNK = L = 2048
L_ola = 256
Hs = L // 4
Hs_ola = L_ola // 2
alpha = 1.5
window = np.hanning(L)
output_buffer = np.zeros(int(L))
prev_fft = None
prev_phase = np.zeros(L//2 + 1)

file_name = 'samples/fred_10sec.wav'
# file_name = sys.argv[1]
audio_data, audio_sr = lb.load(file_name)



def get_total_runtime(profiler):
    # Capture the stats output as a string
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.print_stats()
    output = stream.getvalue()
    
    # Extract the total time (e.g., "0.278 seconds")
    first_line = output.split('\n')[0]  # Get the first line
    total_time_str = first_line.split('in ')[1]  # Extract "0.278 seconds"
    total_time = float(total_time_str.split()[0])  # Convert to float
    
    return total_time

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


xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)

if max(abs(xh)) > 1:
    xh = xh / max(abs(xh))
elif max(abs(xp)) > 1:
    xp = xp / max(abs(xp))

xh = float2pcm(xh).astype(np.int16)
xp = float2pcm(xp).astype(np.int16)

omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L
den = calc_sum_squared_window(window, Hs)


def on_alpha_change(e):
    global alpha
    if e.name == 'up' and alpha < 2.00:
        alpha += 0.01
    elif e.name == 'down' and alpha > 0.10:
        alpha -=0.01
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)
                
def phase_vocoder_processing(xh_chunk, window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr):
    """Isolated Phase Vocoder processing for profiling"""
    # Grabbing corresponding analysis window

    start_time = time.monotonic()
    Ha = int(Hs/alpha)
    pv_win = xh_chunk * window
    end_time = time.monotonic()

    analysis_win_t = end_time - start_time
    
    start_time = time.monotonic()
    S = np.fft.rfft(pv_win)
    magnitude = np.abs(S)
    end_time = time.monotonic()

    stft_t = end_time - start_time
    
    start_time = time.monotonic()

    if prev_fft is not None:
        dphi = np.angle(S) - np.angle(prev_fft)
        dphi = dphi - omega_nom * (Ha/audio_sr)
        dphi = (dphi + np.pi) % (2*np.pi) - np.pi
        w_if = omega_nom + dphi * (audio_sr/Ha)
        prev_phase += w_if * (Hs/audio_sr)
    else:
        prev_phase = np.angle(S)

    end_time = time.monotonic()

    phase_mod_t = end_time - start_time

    start_time = time.monotonic()    
    X_mod = magnitude * np.exp(1j * prev_phase)
    phase_window = np.fft.irfft(X_mod)
    end_time = time.monotonic()

    reconst_t = end_time - start_time

    return phase_window, S, prev_phase, analysis_win_t, stft_t, reconst_t

def ola_processing(xp_chunk, Hs_ola, L_ola, alpha, Hs):
    """Isolated OLA processing for profiling"""
    Ha_ola = int(Hs_ola/alpha)
    ratio = Hs//Hs_ola
    ola_y = np.zeros(L)
    for i in range(ratio):
        ola_win = xp_chunk[Ha_ola*i : Ha_ola*i + L_ola]
        ola_win_synth = ola_win * np.hanning(L_ola)
        offset = i * Hs_ola
        ola_y[offset:offset + L_ola] += ola_win_synth
    return ola_y

keyboard.on_press(on_alpha_change)

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=audio_sr,
                output=True)

pos = 0

def main_processing_loop():
    global pos, prev_fft, prev_phase, output_buffer
    pos = 0  
    prev_fft = None
    prev_phase = np.zeros(L//2 + 1)
    output_buffer = np.zeros(L)
    
    # Initialize profilers
    # pv_profiler = cProfile.Profile()
    # ola_profiler = cProfile.Profile()
    total_profiler = cProfile.Profile()
    
    try:
        while pos <= len(xh) - L:
            # Process Phase Vocoder with profiling
            # pv_profiler.enable()
            pv_frame_mod, S, prev_phase, analysis_win_t, stft_t, reconst_t = phase_vocoder_processing(
                xh[pos:pos+L], window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr
            )
            # pv_profiler.disable()
            
            # Process OLA with profiling
            # ola_profiler.enable()
            ola_y = ola_processing(
                xp[pos:pos+L], Hs_ola, L_ola, alpha, Hs
            )
            # ola_profiler.disable()
            
            # Buffer management
            output_buffer[:-Hs] = output_buffer[Hs:]
            output_buffer[-Hs:] = 0
            output_buffer += pv_frame_mod * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
            output_buffer += ola_y
            output_buffer = np.clip(output_buffer, -32768, 32767)
            
            # stream.write(output_buffer[:Hs].astype(np.int16).tobytes())
            prev_fft = S
            pos += int(Hs/alpha)

    except KeyboardInterrupt:
        print("\nStream stopped by user!")

    # Reset position
    pos = 0  
    prev_fft = None
    prev_phase = np.zeros(L//2 + 1)
    output_buffer = np.zeros(L)


    # try:
    #     total_profiler.enable()
    #     while pos <= len(xh) - L:
    #         # Process Phase Vocoder with profiling
    #         pv_frame_mod, S, prev_phase = phase_vocoder_processing(
    #             xh[pos:pos+L], window, prev_fft, prev_phase, omega_nom, Hs, alpha, audio_sr
    #         )
            
    #         # Process OLA with profiling
    #         ola_y = ola_processing(
    #             xp[pos:pos+L], Hs_ola, L_ola, alpha, Hs
    #         )
            
    #         # Buffer management
    #         output_buffer[:-Hs] = output_buffer[Hs:]
    #         output_buffer[-Hs:] = 0
    #         output_buffer += pv_frame_mod * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
    #         output_buffer += ola_y
    #         output_buffer = np.clip(output_buffer, -32768, 32767)
            
    #         # stream.write(output_buffer[:Hs].astype(np.int16).tobytes())
    #         prev_fft = S
    #         pos += int(Hs/alpha)
    #     total_profiler.disable()

    # except KeyboardInterrupt:
    #     print("\nStream stopped by user!")

    
    return analysis_win_t, stft_t, reconst_t


# Main execution
if __name__ == "__main__":
    # Audio setup
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=audio_sr,
                    output=True)
    
    # print("Playing audio. Press:")
    # print("- UP arrow to increase stretch factor")
    # print("- DOWN arrow to decrease stretch factor")
    # print("- CTRL+C to stop")
    avg_pv_analysis = []
    avg_pv_fft = []
    avg_pv_reconst = []
    # Run processing with profiling
    for _ in range(0,100):
        analysis_win_t, stft_t, reconst_t = main_processing_loop()
        # avg_pv_runtime = get_total_runtime(pv_prof)
        # avg_ola_runtime = get_total_runtime(ola_prof)
        # avg_runtime_tot.append(get_total_runtime(pv_prof))
        avg_pv_analysis.append(analysis_win_t)
        avg_pv_fft.append(stft_t)
        avg_pv_reconst.append(avg_pv_reconst)
    
    avg_pv_analysis = np.mean(avg_pv_analysis)
    avg_pv_fft = np.mean(avg_pv_fft)
    avg_pv_reconst = np.mean(avg_pv_reconst)
    print(f"\nAvg PV analysis frame: {avg_pv_analysis:.3f} seconds for alpha: {alpha}")
    print(f"\nAvg PV fft: {avg_pv_fft:.3f} seconds for alpha: {alpha}")
    print(f"\nAvg PV reconstruct: {avg_pv_reconst:.3f} seconds for alpha: {alpha}")
    # avg_pv, avg_ola = measure_average_runtime(num_trials=100)
    
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # # Save profiling results
    # print("\nPhase Vocoder Profiling Results:")
    # pv_stats = pstats.Stats(pv_prof)
    # pv_stats.strip_dirs().sort_stats('cumulative')
    # pv_stats.print_stats(10)
    
    # print("\nOLA Profiling Results:")
    # ola_stats = pstats.Stats(ola_prof)
    # ola_stats.strip_dirs().sort_stats('cumulative')
    # ola_stats.print_stats(10)

    # print("\nTotal Profiling Results:")
    # total_stats = pstats.Stats(total_prof)
    # total_stats.strip_dirs().sort_stats('cumulative')
    # total_stats.print_stats(10)
    
    # # Save to files
    # with open(f'runtime_analysis/cProf_profiling/phase_vocoder_profile_alpha={alpha}.prof', 'w') as f:
    #     pv_stats = pstats.Stats(pv_prof, stream=f)
    #     pv_stats.strip_dirs().sort_stats('cumulative')
    #     pv_stats.print_stats()
    
    # with open(f'runtime_analysis/cProf_profiling/ola_profile_alpha={alpha}.prof', 'w') as f:
    #     ola_stats = pstats.Stats(ola_prof, stream=f)
    #     ola_stats.strip_dirs().sort_stats('cumulative')
    #     ola_stats.print_stats()
    
    # with open(f'runtime_analysis/cProf_profiling/total_profile_alpha={alpha}.prof', 'w') as f:
    #     total_stats = pstats.Stats(total_prof, stream=f)
    #     total_stats.strip_dirs().sort_stats('cumulative')
    #     total_stats.print_stats()


