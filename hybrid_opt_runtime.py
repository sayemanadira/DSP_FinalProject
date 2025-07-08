import time
import statistics
import numpy as np
import librosa as lb
from scipy.signal import medfilt
import cProfile
import pstats
import os

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

def estimateIF(S, sr, hop_samples):
    '''
    Estimates the instantaneous frequencies in a STFT matrix.
    
    Inputs
    S: the STFT matrix, should only contain the lower half of the frequency bins
    sr: sampling rate
    hop_samples: the hop size of the STFT analysis in samples
    
    Returns a matrix containing the estimated instantaneous frequency at each time-frequency bin.
    This matrix should contain one less column than S.
    '''
    hop_sec = hop_samples / sr
    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1,1))    
    unwrapped = np.angle(S[:,1:]) - np.angle(S[:,0:-1]) - w_nom * hop_sec
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / hop_sec
    return w_if

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

def measure_main_loop(audio_data, audio_sr, iterations=100):
    """Measure only the main while loop runtime"""
    # Precompute everything outside the measurement
    print("Precomputing data for measurements...")
    
    # Harmonic-percussive separation (precomputed once)
    xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)
    xh = float2pcm(xh).astype(np.int16)
    xp = float2pcm(xp).astype(np.int16)
    
    # STFT precomputations
    L = 2048
    Hs = L // 4
    window = np.hanning(L)
    beta = 0.125
    Ha_lookup = int(round(beta*L))
    S_lookup = lb.core.stft(audio_data, n_fft=L, hop_length=Ha_lookup, center=False)
    S_phase_lookup = np.angle(S_lookup)
    S_mag_lookup = np.abs(S_lookup)
    w_if_lookup = estimateIF(S_lookup, audio_sr, Ha_lookup)
    omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L
    den = calc_sum_squared_window(window, Hs)
    
    pv_runtimes = []
    ola_runtimes = []
    tot_runtimes = []
    
    print(f"\nMeasuring main loop performance for {iterations} iterations...")
    

    for iteration in range(iterations):
        # Reset loop variables
        output_buffer = np.zeros(int(L))
        pos = 0
        prev_phase = None
        ola_t = 0
        pv_t = 0
        
        # Only measure the while loop
        start_tot = time.perf_counter()
        # Main loop from hybrid_opt.py
        while pos <= len(xh) - L:
            alpha = 1.75  # Fixed for consistent measurement
            Ha = int(Hs/alpha)
            Ha_ola = int(Hs_ola/alpha)

            start_t = time.perf_counter()
            if pos == 0:
                prev_phase = S_phase_lookup[:, 0]
                S_mod = S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
            else:
                nn_frame = int(round(pos / Ha_lookup))
                lb_frame = int(pos/Ha_lookup)
                phase_increment = w_if_lookup[:, lb_frame-1] * (Hs / audio_sr)
                prev_phase += phase_increment
                S_mod = S_mag_lookup[:, nn_frame] * np.exp(1j * prev_phase)

            pv_frame_mod = np.fft.irfft(S_mod) * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
            
            pv_t += time.perf_counter() - start_t

            # pv_frame_mod = float2pcm(np.array(pv_frame_mod))
            # output_buffer[:-Hs] = output_buffer[Hs:]
            # output_buffer[-Hs:] = 0
            # output_buffer += pv_frame_mod


            start_t = time.perf_counter()
            ratio = Hs//Hs_ola
            ola_y = np.zeros(L)
            for i in range(ratio):
                ola_win = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola]
                ola_win_synth = ola_win * np.hanning(L_ola)
                offset = i * Hs_ola
                ola_y[offset:offset + L_ola] += ola_win_synth
            ola_t += time.perf_counter() - start_t
            # output_buffer += ola_y
            # output_buffer = np.clip(output_buffer, -32768, 32767)
            pos += Ha
        
        # end_time = time.time()
        tot_runtimes.append(time.perf_counter() - start_tot)
        pv_runtimes.append(pv_t)
        ola_runtimes.append(ola_t)
        # print(f"Iteration {iteration+1}/{iterations}: {tot_runtimes[-1]:.5f} seconds")
    
    # Calculate statistics
    avg_runtime_pv = statistics.mean(pv_runtimes)
    avg_runtime_ola = statistics.mean(ola_runtimes)
    std_de_pv = statistics.stdev(pv_runtimes) if len(pv_runtimes) > 1 else 0
    std_de_ola = statistics.stdev(ola_runtimes) if len(ola_runtimes) > 1 else 0
    print("\nMain Loop Results:")
    print(f"Average runtime (PV): {avg_runtime_pv:.5f} seconds, std ")
    print(f"Standard deviation (PV): {std_de_pv:.5f} seconds")
    print(f"Average runtime (OLA): {avg_runtime_ola:.5f} seconds")
    print(f"Standard deviation (OLA): {std_de_ola:.5f} seconds")
    print(f"Avg loop time for {iterations} iterations: {statistics.mean(tot_runtimes):.5f} seconds")
    

def profile_main_loop(audio_data, audio_sr, output_file="main_loop_profile.prof"):
    """Profile only the main while loop"""
    print("\nProfiling main loop...")
    
    # Precompute data
    xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)
    xh = float2pcm(xh).astype(np.int16)
    xp = float2pcm(xp).astype(np.int16)
    
    L = 2048
    Hs = L // 4
    window = np.hanning(L)
    beta = 0.125
    Ha_lookup = int(round(beta*L))
    S_lookup = lb.core.stft(audio_data, n_fft=L, hop_length=Ha_lookup, center=False)
    S_phase_lookup = np.angle(S_lookup)
    S_mag_lookup = np.abs(S_lookup)
    w_if_lookup = estimateIF(S_lookup, audio_sr, Ha_lookup)
    
    # Setup profiling
    def run_loop():
        output_buffer = np.zeros(int(L))
        pos = 0
        prev_phase = None
        
        while pos <= len(xh) - L:
             alpha = 0.75
             Ha = int(round(Hs/alpha))
             Ha_ola = int(Hs_ola/alpha)
            
            # start_time = time.perf_counter()

            # Phase Vocoder       
            # STFT processing
             pv_win = xh[pos:pos+L] * window
             S = np.fft.rfft(pv_win)
             magnitude = np.abs(S)
             if prev_fft is not None:
                dphi = np.angle(S) - np.angle(prev_fft)
                dphi = dphi - omega_nom * (Ha/audio_sr)
                dphi = (dphi + np.pi) % (2*np.pi) - np.pi
                w_if = omega_nom + dphi * (audio_sr/Ha)
                prev_phase += w_if * (Hs/audio_sr)
             else:
                prev_phase = np.angle(S)
            
             X_mod = magnitude * np.exp(1j * prev_phase)
             pv_frame_mod = np.fft.irfft(X_mod)

            #shift and add to stream
             output_buffer[:-Hs] = output_buffer[Hs:]
             output_buffer[-Hs:] = 0
             output_buffer += pv_frame_mod * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()


             ratio = Hs//Hs_ola
             ola_y = np.zeros(L)
             for i in range(ratio):
                ola_win = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola]
                ola_win_synth = ola_win * np.hanning(L_ola)
                offset = i * Hs_ola
                ola_y[offset:offset + L_ola] += ola_win_synth
        
             output_buffer += ola_y

             output_buffer = np.clip(output_buffer, -32768, 32767)  # 16-bit range
        # runtimes.append(end_time - start_time)
        # stream.write(output_buffer[:Hs].astype(np.int16).tobytes())

        # saved_frames.append(output_buffer[:Hs].astype(np.int16).copy())

        prev_fft = S
        pos += Ha
            # alpha = 1.25
            # Ha = int(Hs/alpha)
            # Ha_ola = int(Hs_ola/alpha)
            
            # if pos == 0:
            #     prev_phase = S_phase_lookup[:, 0]
            #     S_mod = S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
            # else:
            #     nn_frame = int(round(pos / Ha_lookup))
            #     lb_frame = int(pos/Ha_lookup)
            #     phase_increment = w_if_lookup[:, lb_frame-1] * (Hs / audio_sr)
            #     prev_phase += phase_increment
            #     S_mod = S_mag_lookup[:, nn_frame] * np.exp(1j * prev_phase)

            # pv_frame_mod = np.fft.irfft(S_mod) * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
            # pv_frame_mod = float2pcm(np.array(pv_frame_mod))
            
            # output_buffer[:-Hs] = output_buffer[Hs:]
            # output_buffer[-Hs:] = 0
            # output_buffer += pv_frame_mod

            # ratio = Hs//Hs_ola
            # ola_y = np.zeros(L)
            # for i in range(ratio):
            #     ola_win = xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + L_ola]
            #     ola_win_synth = ola_win * np.hanning(L_ola)
            #     offset = i * Hs_ola
            #     ola_y[offset:offset + L_ola] += ola_win_synth
        
            # output_buffer += ola_y
            # output_buffer = np.clip(output_buffer, -32768, 32767)
            
            # pos += Ha
    
    # Run profiling
    cProfile.runctx('run_loop()', globals(), locals(), output_file)
    
    # Print results
    stats = pstats.Stats(output_file)
    stats.sort_stats('cumtime').print_stats(20)

if __name__ == "__main__":
    # Load audio file (replace with your 10-second file)
    audio_file = "samples/runtime_samples/fred_10sec.wav"
    audio_data, audio_sr = lb.load(audio_file)
    
    # Global constants from hybrid_opt.py
    global L, Hs, Hs_ola, den
    L = 2048
    L_ola = 256
    Hs = L // 4
    Hs_ola = L_ola // 2
    window = np.hanning(L)
    den = calc_sum_squared_window(window, Hs)
    
    # Run measurements
    measure_main_loop(audio_data, audio_sr, iterations=200)
    
    # Run profiling
    # profile_main_loop(audio_data, audio_sr)