import numpy as np
import wave
import pyaudio
import threading
import librosa as lb
from scipy.signal import medfilt
import time
import csv

import subprocess
import tempfile
import os
import subprocess
import tempfile
import os, sys

# === PyInstaller Helper ===
def get_resource_path(relative_path):
    """Get the absolute path to a resource, whether frozen or not."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def convert_to_pcm16_wav(input_path):
    # Create a temporary output file with .wav suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    output_path = tmp.name

    # FFmpeg command to convert to 16-bit PCM WAV, mono, 22.5kHz
    ffmpeg_path = get_resource_path("ffmpeg")
    
    cmd = [
        ffmpeg_path,
        "-y",                   # overwrite existing file
        "-i", input_path,       # input file
        "-t", "30",
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ac", "1",             # mono
        "-ar", "22050",         # sample rate 22.5
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed for {input_path}: {e}")
        os.unlink(output_path)  # Clean up temp file
        raise

    return output_path

class EngineBase:
    def __init__(self, filename, fft_size=2048, on_complete=None):
        # self.filename = convert_to_pcm16_wav(filename)
        self.filename = filename
        self.L = fft_size
        self.Hs = self.L // 4
        self.window = np.hanning(self.L)
        self.alpha = 1.0
        
        self.chunk_size = 512
        
        self.output_buffer = np.zeros(self.L)
        self.audio_data = None
        self.audio_sr = None
        self.stream = None
        self.p = None
        self.prev_phase = np.zeros(self.L // 2 + 1)
        self.prev_fft = None
        self.running = False
        self.thread = None
        
        self.on_complete = on_complete
        self.complete = False
        self.reset_state()

    def set_alpha(self, a):
        self.alpha = a #ax(0.1, min(a, 4.0))

    def load_audio(self):
        self.audio_data, self.audio_sr = lb.load(self.filename, sr=None)
        # print(self.audio_sr)

    def setup_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_sr,
            output=True,
            frames_per_buffer=self.chunk_size,
        )

    def close_audio_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def float2pcm(self, sig, dtype='int16'):
        # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
        sig = np.asarray(sig)
        dtype = np.dtype(dtype)
        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

    def pcm2float(self, sig):
        return sig.astype(np.float32) / (2 ** 15)

    def calc_sum_squared_window(self, window, hop_length):
        assert len(window) % hop_length == 0, "Hop length must divide window length."
        numShifts = len(window) // hop_length
        den = np.zeros_like(window)
        for i in range(numShifts):
            den += np.roll(np.square(window), i * hop_length)
        return den

    def reset_state(self):
        
        self.wf = wave.open(self.filename, 'rb')
        # self.audio_sr = self.wf.getframerate()
        _, self.audio_sr = lb.load(self.filename)
        
        # with wave.open(self.filename, 'rb') as wf:
        #     assert self.wf.getsampwidth() == 2  # 2 bytes = 16 bits
        #     assert self.wf.getnchannels() == 1
        self.setup_audio_stream()
        
        self.output_buffer = np.zeros(self.L)
        self.prev_phase = np.zeros(self.L // 2 + 1)
        self.prev_fft = None
        
    def set_paused(self, paused):
        try:
            if self.stream:
                if paused and self.stream.is_active():
                    self.stream.stop_stream()
                elif not paused and not self.stream.is_active():
                    self.stream.start_stream()
        except OSError as e:
            print(f"Audio stream error: {e}")
            # Optionally reinitialize the stream
            self.reinitialize_stream()
            
    def on_complete_post(self):
        self.on_complete()
    
    def start(self):
        self.reset_state()
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _run(self):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _run method")


class OLAEngine(EngineBase):
    def __init__(self, filename, on_complete=None):
        super().__init__(filename, fft_size=256, on_complete=on_complete)
        self.Hs = self.L // 2
        

    def _run(self):

        num_samples = self.wf.getnframes()
        pos = 0

        try:
            while self.running and pos <= num_samples - self.L:
                self.wf.setpos(pos)
                data = self.wf.readframes(self.L)
                x = np.frombuffer(data, dtype=np.int16)

                Ha = int(round(self.Hs / self.alpha))

                analysis_buffer = x * self.window
                synthesis_buffer = analysis_buffer

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer[:self.L] += synthesis_buffer

                # self.stream.write(np.clip(self.output_buffer[:self.Hs], -32768, 32767).astype(np.int16).tobytes())
                self.stream.write(self.output_buffer[:self.Hs].astype(np.int16).tobytes())
                # fade_win = np.hanning(self.Hs * 2)[self.Hs:]  # smooth fade-out
                # chunk = self.output_buffer[:self.Hs] * fade_win
                # self.stream.write(chunk.astype(np.int16).tobytes())
                
                pos += Ha

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:  # call callback
                self.on_complete_post()


class PVEngine(EngineBase):
    def __init__(self, filename, on_complete=None):
        super().__init__(filename, fft_size=2048, on_complete=on_complete)
        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L
        
    def _run(self):
        num_samples = self.wf.getnframes()
        pos = 0

        try:
            while self.running and pos <= num_samples - self.L:
                self.wf.setpos(pos)
                data = self.wf.readframes(self.L)
                x = np.frombuffer(data, dtype=np.int16).astype(np.float32)

                Ha = int(np.round(self.Hs / self.alpha))

                if len(x) < self.L:
                    x = np.pad(x, (0, self.L - len(x)))

                frame = x[:self.L] * self.window
                S = np.fft.rfft(frame)

                if self.prev_fft is None:
                    w_if = np.zeros_like(self.omega_nom)
                else:
                    dphi = np.angle(S) - np.angle(self.prev_fft)
                    dphi = dphi - self.omega_nom * (Ha / self.audio_sr)
                    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                    w_if = self.omega_nom + dphi * (self.audio_sr / Ha)

                self.prev_phase = self.prev_phase + w_if * (self.Hs / self.audio_sr)

                X_mod = np.abs(S) * np.exp(1j * self.prev_phase)
                frame_mod = np.fft.irfft(X_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += frame_mod * self.window

                output_int16 = np.clip(self.output_buffer[:self.Hs], -32768, 32767).astype(np.int16)
                self.stream.write(output_int16.tobytes())

                self.prev_fft = S
                pos += Ha

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True

def invert_stft(S, hop_length, window):
    L = len(window)
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0:S.shape[0], :] = S
    Sfull[S.shape[0]:, :] = np.conj(np.flipud(S[1:fft_size // 2, :]))

    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:, i] = np.fft.ifft(Sfull[:, i])
    frames = np.real(frames)

    den = calc_sum_squared_window(window, hop_length)
    frames = frames * window.reshape((-1, 1)) / den.reshape((-1, 1))

    y = np.zeros(hop_length * (frames.shape[1] - 1) + L)
    for i in range(frames.shape[1]):
        offset = i * hop_length
        y[offset:offset + L] += frames[:, i]
    return y


def calc_sum_squared_window(window, hop_length):
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i * hop_length)
    return den


def harmonic_percussive_separation(x, sr=22050, fft_size=2048, hop_length=512, lh=6, lp=6):
    window = np.hanning(fft_size)
    X = lb.core.stft(x, n_fft=fft_size, hop_length=hop_length, window=window, center=False)
    Y = np.abs(X)
    Yh = medfilt(Y, (1, 2 * lh + 1))
    Yp = medfilt(Y, (2 * lp + 1, 1))
    Mh = (Yh > Yp)
    Mp = np.logical_not(Mh)
    Xh = X * Mh
    Xp = X * Mp
    xh = invert_stft(Xh, hop_length, window)
    xp = invert_stft(Xp, hop_length, window)
    return xh, xp


class HybridEngine(EngineBase):
    def __init__(self, filename, on_complete=None):
        super().__init__(filename, fft_size=2048, on_complete=on_complete)
        
        self.omega_nom = None
        self.den = None
        self.xh = None
        self.xp = None
        self.separate_hpss()
    
    def reset_state(self):
        super().reset_state()
        self.L_ola = 256
        self.Hs_ola = self.L_ola // 2
        # self.runtimes = []
        self.setup_audio_stream()

    def separate_hpss(self):
        x, self.audio_sr = lb.load(self.filename)
        
        xh, xp = harmonic_percussive_separation(x, self.audio_sr)
        if max(abs(xh)) > 1: 
            xh /= max(abs(xh))
        if max(abs(xp)) > 1: 
            xp /= max(abs(xp))
        
        self.xh = xh
        self.xp = xp

        # self.xh = self.float2pcm(xh).astype(np.int16)
        # self.xp = self.float2pcm(xp).astype(np.int16)

        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L
        self.den = self.calc_sum_squared_window(self.window, self.Hs)

    def _run(self):
        """Threading implementation for consistency with base class"""
        
        pos = 0
        try:
            while self.running and pos <= len(self.xh) - self.L:
                Ha = int(self.Hs / self.alpha)
                Ha_ola = int(self.Hs_ola / self.alpha)

                # Phase Vocoder (harmonic)
                pv_win = self.xh[pos:pos + self.L] * self.window
                S = np.fft.rfft(pv_win)

                if self.prev_fft is not None:
                    dphi = np.angle(S) - np.angle(self.prev_fft)
                    dphi = (dphi - self.omega_nom * (Ha / self.audio_sr) + np.pi) % (2 * np.pi) - np.pi
                    w_if = self.omega_nom + dphi * (self.audio_sr / Ha)
                    self.prev_phase += w_if * (self.Hs / self.audio_sr)
                else:
                    self.prev_phase = np.angle(S)

                X_mod = np.abs(S) * np.exp(1j * self.prev_phase)
                pv_frame_mod = np.fft.irfft(X_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += pv_frame_mod * (self.window / self.den)

                # OLA (percussive)
                ola_y = np.zeros(self.L)
                ratio = self.Hs // self.Hs_ola
                for i in range(ratio):
                    start_i = pos + (Ha_ola * i)
                    if start_i + self.L_ola > len(self.xp):
                        continue
                    ola_win = self.xp[start_i:start_i + self.L_ola]
                    ola_y[i * self.Hs_ola:i * self.Hs_ola + self.L_ola] += ola_win * np.hanning(self.L_ola)

                self.output_buffer += ola_y

                self.output_buffer = np.clip(self.output_buffer, -1.0, 1.0)
                # self.stream.write(np.clip(self.output_buffer[:self.Hs], -32768, 32767).astype(np.int16).tobytes())
                
                # fade_win = np.hanning(self.Hs * 2)[self.Hs:]  # smooth fade-out
                # chunk = self.output_buffer[:self.Hs] * fade_win
                # self.stream.write(chunk.astype(np.int16).tobytes())
                self.stream.write(self.float2pcm(self.output_buffer[:self.Hs]).astype(np.int16).tobytes())

                self.prev_fft = S
                pos += Ha
                # self.runtimes.append(time.perf_counter() - start)

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:
                self.on_complete_post()
                

class OPTEngine(EngineBase):
    def __init__(self, filename, beta=0.25, on_complete=None):
        super().__init__(filename, fft_size=2048, on_complete=on_complete)

        self.beta = beta
        self.L_ola = 256
        self.Hs_ola = self.L_ola // 2
        self.prev_phase = np.zeros(self.L//2 + 1)
        self.S_lookup = None
        self.S_phase_lookup = None
        self.S_mag_lookup = None
        self.w_if_lookup = None
        self.xh = None
        self.xp = None

        self.prepare_hpss()

    def reset_state(self):
        super().reset_state()
        self.prev_phase = None
        self.setup_audio_stream()

    def prepare_hpss(self):
        x, self.audio_sr = lb.load(self.filename)

        # HPSS separation
        xh, xp = self.harmonic_percussive_separation(x, self.audio_sr)
        if np.max(np.abs(xh)) > 1:
            xh /= np.max(np.abs(xh))
        if np.max(np.abs(xp)) > 1:
            xp /= np.max(np.abs(xp))

        # self.xh = self.float2pcm(xh).astype(np.int16)
        # self.xp = self.float2pcm(xp).astype(np.int16)
        self.xh = xh
        self.xp = xp

        # Precompute STFT, phase and IF lookup for time-varying alpha
        Ha_lookup = int(round(self.beta * self.L))

        self.S_lookup = lb.core.stft(self.xh, n_fft=self.L, hop_length=Ha_lookup, win_length=self.L, center=False, dtype=np.complex64)
        self.S_phase_lookup = np.angle(self.S_lookup)
        self.S_mag_lookup = np.abs(self.S_lookup)
        self.w_if_lookup = self.estimate_instantaneous_frequency(self.S_lookup, self.audio_sr, Ha_lookup)

        self.den = self.calc_sum_squared_window(self.window, self.Hs)

    def harmonic_percussive_separation(self, x, sr, lh=6, lp=6):
        fft_size = self.L
        hop_length = self.Hs
        window = np.hanning(fft_size)

        X = lb.core.stft(x, n_fft=fft_size, hop_length=hop_length, window=window, center=False)
        Y = np.abs(X)

        Yh = medfilt(Y, (1, 2*lh+1))
        Yp = medfilt(Y, (2*lp+1, 1))

        Mh = Yh > Yp
        Mp = ~Mh

        Xh = X * Mh
        Xp = X * Mp

        xh = self.invert_stft(Xh, hop_length, window)
        xp = self.invert_stft(Xp, hop_length, window)

        return xh, xp

    def estimate_instantaneous_frequency(self, S, sr, hop_samples):
        hop_sec = hop_samples / sr
        fft_size = (S.shape[0] - 1) * 2
        w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
        w_nom = w_nom.reshape((-1, 1))
        unwrapped = np.angle(S[:, 1:]) - np.angle(S[:, 0:-1]) - w_nom * hop_sec
        wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
        return w_nom + wrapped / hop_sec

    def invert_stft(self, S, hop_length, window):
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
        den = self.calc_sum_squared_window(window, hop_length)
        frames = frames * window.reshape((-1,1)) / den.reshape((-1,1))
        
        # reconstruction
        y = np.zeros(hop_length*(frames.shape[1]-1) + L)
        for i in range(frames.shape[1]):
            offset = i * hop_length
            y[offset:offset+L] += frames[:,i]
        return y

    def _run(self):
        pos = 0
        Ha_lookup = int(self.beta* self.L)
        ratio = self.Hs // self.Hs_ola
        windowOLA = np.hanning(self.L_ola)

        # try:
        #     while self.running and pos <= len(self.xh) - self.L:
        #         Ha = int(round(self.Hs / self.alpha))
        #         Ha_ola = int(round(self.Hs_ola / self.alpha))

        
        #         if pos == 0:
        #             prev_phase = self.S_phase_lookup[:, 0]
        #             S_mod = self.S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
        #         else:
        #             nn_frame = int(round(pos / Ha_lookup))  # lookup is always based on min_Ha
        #             lb_frame = int(pos/Ha_lookup)
        #             phase_increment = self.w_if_lookup[:, lb_frame] * (self.Hs / self.audio_sr)
        #             prev_phase += phase_increment  # Update phase correctly for current alpha
        #             S_mod = self.S_mag_lookup[:, nn_frame] * np.exp(1j * prev_phase)

        #         pv_frame_mod = np.fft.irfft(S_mod)
        #         # if pos == 0:
        #         #     prev_phase = self.S_phase_lookup[:, 0]
        #         #     S_mod = self.S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
        #         # else:
        #         #     # Phase increment calculated with respect to Ha_lookup (fixed analysis hop)
        #         #     delta_phase = self.w_if_lookup[:, nn_frame] * (self.Hs / self.audio_sr)
        #         #     prev_phase += delta_phase
        #         #     S_mod = self.S_mag_lookup[:, nn_frame + 1] * np.exp(1j * prev_phase)

        #         # pv_frame_mod = np.fft.irfft(S_mod)
        #         self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
        #         self.output_buffer[-self.Hs:] = 0
        #         self.output_buffer += pv_frame_mod * (self.window.reshape((-1,1)) / self.den.reshape((-1,1))).flatten()

        #         # OLA synthesis
        #         for i in range(ratio):
        #             start_i = pos + Ha_ola * i
        #             if start_i + self.L_ola > len(self.xp):
        #                 continue
        #             ola_win = self.xp[start_i:start_i + self.L_ola]
        #             self.output_buffer[i * self.Hs_ola:i * self.Hs_ola + self.L_ola] += ola_win * windowOLA

        #         # self.output_buffer += ola_y
        #         self.output_buffer = np.clip(self.output_buffer, -1.0, 1.0)
        #         self.stream.write(self.float2pcm(self.output_buffer[:self.Hs]).astype(np.int16).tobytes())

        #         pos += Ha

        try:
            while self.running and pos <= len(self.xh) - self.L:
                Ha = int(round(self.Hs/self.alpha))
                Ha_ola = int(round(self.Hs_ola/self.alpha))
                
                if pos == 0:
                    prev_phase = self.S_phase_lookup[:, 0]
                    S_mod = self.S_mag_lookup[:, 0] * np.exp(1j * prev_phase)
                else:
                    nn_frame = int(round(pos / Ha_lookup))  # lookup is always based on min_Ha
                    lb_frame = int(pos/Ha_lookup)
                    phase_increment = self.w_if_lookup[:, nn_frame-1] * (self.Hs / self.audio_sr)
                    prev_phase += phase_increment  # Update phase correctly for current alpha
                    S_mod = self.S_mag_lookup[:, nn_frame] * np.exp(1j * prev_phase)

                pv_frame_mod = np.fft.irfft(S_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += pv_frame_mod * (self.window.reshape((-1, 1))/self.den.reshape((-1,1))).flatten()

                # #TODO: REMINDER - uncomment to try "OLA Lookup"
                # nn_frame_OLA = int(round(pos/Ha_lookup_ola))       
            
                # for offset in OLA_offsets:
                #     output_buffer[offset: offset+L_ola] += OLA_frame_lookup[nn_frame_OLA]
                    
                # #TODO: Uncomment when NO LOOK-UP
                for i in range(ratio):
                    ola_win_synth = self.xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + self.L_ola] * windowOLA
                    offset = i *self.Hs_ola
                    self.output_buffer[offset:offset + self.L_ola] += ola_win_synth
            

                # output_buffer = np.clip(output_buffer, -32768, 32767)  # 16-bit range
                # runtimes.append(end_time - start_time)
                self.output_buffer = np.clip(self.output_buffer, -1.0, 1.0)  # Float32 clipping
                self.stream.write(self.float2pcm(self.output_buffer[:self.Hs]).astype(np.int16).tobytes())  # Convert to int16 at last moment
                # stream.write(output_buffer[:Hs].astype(np.int16).tobytes())

                # Store for WAV file
                # output_frames.append(float2pcm(output_buffer[:Hs]).astype(np.int16).copy())  # Store the chunk
                # print(pos//Ha)
                # prev_fft = S
            
                # phase_dev = np.std(np.diff(prev_phase[:bass_bin]))
                # print(f"Bass phase deviation: {phase_dev:.3f} radians")
                pos += Ha
        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:
                self.on_complete_post()
