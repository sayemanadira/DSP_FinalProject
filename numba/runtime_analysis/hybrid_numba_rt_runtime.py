import time
import statistics
import numpy as np
import librosa as lb
from scipy.signal import medfilt
import cProfile
import pstats
import os
from numba import njit

@njit(fastmath=True)
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

@njit(fastmath=True)
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

@njit(fastmath=True)
def pv_numba(omega_nom, Ha, prev_fft, prev_phase, S):
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
    return X_mod, prev_phase

@njit(fastmath=True)
def pv_lookup_numba(pos, S_phase_lookup, S_mag_lookup, prev_phase, Ha_lookup, w_if_lookup, Hs, audio_sr):
    if pos == 0:
        prev_phase += S_phase_lookup[:, 0]  # Copy values
        S_mod = S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
    else:
        nn_frame = int(round(pos / Ha_lookup))
        lb_frame = int(pos/Ha_lookup)
        phase_increment = w_if_lookup[:, lb_frame-1] * (Hs / audio_sr)
        prev_phase += phase_increment
        S_mod = S_mag_lookup[:, nn_frame] * np.exp(1j * prev_phase)
    return S_mod, prev_phase


@njit(fastmath=True, cache=True)
def ola_numba(xp, pos, Ha_ola, L_ola, window_OLA, output_buffer, ratio):
    temp = np.empty(L_ola)  # Reused buffer
    for i in range(ratio):
        temp[:] = xp[pos + Ha_ola*i : pos + Ha_ola*i + L_ola]  # Fill existing array
        temp *= window_OLA
        output_buffer[i*Hs_ola : i*Hs_ola + L_ola] += temp
    return output_buffer

        
@njit(nogil=True, fastmath=True)
def float2pcm(sig, dtype='int16'):
    # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def measure_main_loop_OPT(audio_data, audio_sr, alpha, iterations=100):
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
    L_ola = 256
    Hs_ola = L_ola // 2
    ratio = Hs//Hs_ola
    window = np.hanning(L)
    window_OLA = np.hanning(L_ola)

    #Phase vocoder look-up
    beta = 0.125
    Ha_lookup = int(round(beta*L))
    S_lookup = lb.core.stft(audio_data, n_fft=L, hop_length=Ha_lookup, center=False)
    S_phase_lookup = np.angle(S_lookup)
    S_mag_lookup = np.abs(S_lookup)
    w_if_lookup = estimateIF(S_lookup, audio_sr, Ha_lookup)
    omega_nom = np.arange(L//2 + 1) * 2 *np.pi * audio_sr / L

    #OLA look-up
    ratio = Hs//Hs_ola
    Ha_lookup_ola = int(round(beta*L_ola))
    precomp_pos = 0
    OLA_frame_lookup = []
    while precomp_pos <= len(xp) - L_ola:
        OLA_frame_lookup.append(xp[precomp_pos:precomp_pos+L_ola] * window_OLA)
        precomp_pos += Ha_lookup_ola

    OLA_offsets = [i*Hs_ola for i in range(ratio)]

    den = calc_sum_squared_window(window, Hs)
    
    pv_runtimes = []
    ola_runtimes = []
    tot_runtimes = []
    
    print(f"\nMeasuring main loop performance for {iterations} iterations...")
    

    for iteration in range(iterations):
        # Reset loop variables
        output_buffer = np.zeros(int(L))
        pos = 0
        prev_phase = np.zeros(S_lookup.shape[0])
        ola_t = 0
        pv_t = 0
        
        # Only measure the while loop
        start_tot = time.perf_counter()
        # Main loop from hybrid_opt.py
        while pos <= len(xh) - L:
            Ha = int(Hs/alpha)
            Ha_ola = int(Hs_ola/alpha)

            start_t = time.perf_counter()
            X_mod, prev_phase = pv_lookup_numba(
                pos, S_phase_lookup, S_mag_lookup, 
                prev_phase, Ha_lookup, w_if_lookup,
                Hs, audio_sr  # Add these parameters
            )
            pv_frame_mod = np.fft.irfft(X_mod) * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
            
            pv_t += time.perf_counter() - start_t

            # pv_frame_mod = float2pcm(np.array(pv_frame_mod))
            # output_buffer[:-Hs] = output_buffer[Hs:]
            # output_buffer[-Hs:] = 0
            # output_buffer += pv_frame_mod


            start_t = time.perf_counter()
            
            # TODO: uncomment when you want OLA LOOK-UP.
            # nn_frame_OLA = int(round(pos/Ha_lookup_ola))            # read_start = pos + ola_relative_offsets[i]
    
            # for offset in OLA_offsets:
            #     output_buffer[offset: offset+L_ola] += OLA_frame_lookup[nn_frame_OLA]
            
            # TODO: uncomment when you want OLA BASELINE.
            output_buffer = ola_numba(xp, pos, Ha_ola, L_ola, window_OLA, output_buffer, ratio)

            ola_t += time.perf_counter() - start_t
            # output_buffer += ola_y
            # output_buffer = np.clip(output_buffer, -32768, 32767)
            pos += Ha
        
        tot_runtimes.append(time.perf_counter() - start_tot)
        pv_runtimes.append(pv_t)
        ola_runtimes.append(ola_t)
        # print(f"Iteration {iteration+1}/{iterations}: {tot_runtimes[-1]:.5f} seconds")
    
    # Calculate statistics
    avg_runtime_pv = np.mean(pv_runtimes[-100:])
    avg_runtime_ola = np.mean(ola_runtimes[-100:])
    std_de_pv = np.std(pv_runtimes[-100:]) if len(pv_runtimes[-100:]) > 1 else 0
    std_de_ola = np.std(ola_runtimes[-100:]) if len(ola_runtimes[-100:]) > 1 else 0
    print("\nOPT - Main Loop Results:")
    print(f"Average runtime (PV): {avg_runtime_pv:.5f} seconds, std ")
    print(f"Standard deviation (PV): {std_de_pv:.5f} seconds")
    print(f"Average runtime (OLA): {avg_runtime_ola:.5f} seconds")
    print(f"Standard deviation (OLA): {std_de_ola:.5f} seconds")
    print(f"Num of iteratons: {len(pv_runtimes[-100:])}")
    print(f"Avg total time for {iterations} iterations: {statistics.mean(tot_runtimes[-100:]):.5f} seconds")
    print(f"Std dev total time for {iterations} iterations: {statistics.stdev(tot_runtimes[-100:]):.5f} seconds")

print("Precomputing data for measurements...")
    
# Harmonic-percussive separation (precomputed once)


def measure_main_loop_BASELINE(audio_data, audio_sr, alpha, iterations=100):
    """Measure only the main while loop runtime"""
    # Precompute everything outside the measurement
    xh, xp, _, _ = harmonic_percussive_separation(x=audio_data, sr=audio_sr)
    xh = float2pcm(xh).astype(np.int16)
    xp = float2pcm(xp).astype(np.int16)

    # STFT precomputations
    L = 2048
    Hs = L // 4
    L_ola = 256
    Hs_ola = L_ola // 2
    ratio = Hs//Hs_ola
    window = np.hanning(L)
    window_OLA = np.hanning(L_ola)
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
        prev_phase = np.zeros(L//2 + 1)
        ola_t = 0
        pv_t = 0
        prev_fft = None
        
        # Only measure the while loop
        start_tot = time.perf_counter()
        # Main loop from hybrid_opt.py
        while pos <= len(xh) - L:
            Ha = int(round(Hs/alpha))
            Ha_ola = int(Hs_ola/alpha)
            
            # start_time = time.perf_counter()

            start_t_pv = time.perf_counter()
            # Phase Vocoder       
            pv_win = xh[pos:pos+L] * window
            S = np.fft.rfft(pv_win)
            X_mod, prev_phase = pv_numba(omega_nom, Ha, prev_fft, prev_phase, S)
            pv_frame_mod = np.fft.irfft(X_mod) * (window.reshape((-1, 1))/den.reshape((-1,1))).flatten()
            pv_t += time.perf_counter() - start_t_pv

            start_t_ola = time.perf_counter()
            output_buffer = ola_numba(xp, pos, Ha_ola, L_ola, window_OLA, output_buffer, ratio)
            ola_t += time.perf_counter() - start_t_ola

            prev_fft = S
            pos += Ha
        
        # end_time = time.time()
        tot_runtimes.append(time.perf_counter() - start_tot)
        pv_runtimes.append(pv_t)
        ola_runtimes.append(ola_t)
        # print(f"Iteration {iteration+1}/{iterations}: {tot_runtimes[-1]:.5f} seconds")
    
    # Calculate statistics
    avg_runtime_pv = np.mean(pv_runtimes[-100:])
    avg_runtime_ola = np.mean(ola_runtimes[-100:])
    std_de_pv = np.std(pv_runtimes[-100:]) if len(pv_runtimes[-100:]) > 1 else 0
    std_de_ola = np.std(ola_runtimes[-100:]) if len(ola_runtimes[-100:]) > 1 else 0
    print("\nBASELINE - Main Loop Results:")
    print(f"Average runtime (PV): {avg_runtime_pv:.5f} seconds, std ")
    print(f"Standard deviation (PV): {std_de_pv:.5f} seconds")
    print(f"Average runtime (OLA): {avg_runtime_ola:.5f} seconds")
    print(f"Standard deviation (OLA): {std_de_ola:.5f} seconds")
    print(f"Num of iteratons: {len(pv_runtimes[-100:])}")
    print(f"Avg total time for {iterations} iterations: {statistics.mean(tot_runtimes[-100:]):.5f} seconds")
    print(f"Std dev total time for {iterations} iterations: {statistics.stdev(tot_runtimes[-100:]):.5f} seconds")


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
    measure_main_loop_OPT(audio_data, audio_sr, 2.00, iterations=200)
    measure_main_loop_BASELINE(audio_data, audio_sr, 2.00,iterations=200)
    