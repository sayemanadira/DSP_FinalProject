import numpy as np
import wave
import pyaudio
import threading
import librosa as lb
from scipy.signal import medfilt
import time
import csv


class EngineBase:
    def __init__(self, filename, fft_size=2048, on_complete=None):
        self.filename = filename
        self.L = fft_size
        self.Hs = self.L // 4
        self.window = np.hanning(self.L)
        self.alpha = 1.0
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

    def set_alpha(self, a):
        self.alpha = max(0.1, min(a, 2.0))

    def load_audio(self, mono=True):
        self.audio_data, self.audio_sr = lb.load(self.filename, sr=None, mono=mono)

    def setup_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_sr,
            output=True
        )

    def close_audio_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def float2pcm(self, sig):
        sig = np.asarray(sig)
        i = np.iinfo(np.int16)
        return (sig * (2**15)).clip(i.min, i.max).astype(np.int16)

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
    
    def start(self):
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
        wf = wave.open(self.filename, 'rb')
        self.audio_sr = wf.getframerate()
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=self.audio_sr,
            output=True
        )

        num_samples = wf.getnframes()
        pos = 0

        try:
            while self.running and pos <= num_samples - self.L:
                wf.setpos(pos)
                data = wf.readframes(self.L)
                x = np.frombuffer(data, dtype=np.int16)

                Ha = int(round(self.Hs / self.alpha))

                analysis_buffer = x * self.window
                synthesis_buffer = analysis_buffer

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer[:self.L] += synthesis_buffer

                self.stream.write(self.output_buffer[:self.Hs].astype(np.int16).tobytes())
                pos += Ha

        finally:
            self.close_audio_stream()
            wf.close()
            if self.on_complete:
                self.on_complete()
            complete = True


class PVEngine(EngineBase):
    def __init__(self, filename, on_complete=None):
        super().__init__(filename, fft_size=2048, on_complete=on_complete)
        self.omega_nom = None  # Will be initialized after sampling rate is known

    def _run(self):
        wf = wave.open(self.filename, 'rb')
        self.audio_sr = wf.getframerate()

        # Initialize omega_nom here (requires sr)
        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=self.audio_sr,
            output=True
        )

        num_samples = wf.getnframes()
        pos = 0

        try:
            while self.running and pos <= num_samples - self.L:
                wf.setpos(pos)
                data = wf.readframes(self.L)
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
            wf.close()
            if self.on_complete:
                self.on_complete()
            complete = True


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


def harmonic_percussive_separation(x, sr, fft_size=2048, hop_length=512, lh=6, lp=6):
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
        self.L_ola = 256
        self.Hs_ola = self.L_ola // 2
        self.omega_nom = None
        self.den = None
        self.xh = None
        self.xp = None
        self.runtimes = []

    def separate_hpss(self):
        x, self.audio_sr = lb.load(self.filename)
        
        xh, xp = harmonic_percussive_separation(x, self.audio_sr)
        if max(abs(xh)) > 1: 
            xh /= max(abs(xh))
        if max(abs(xp)) > 1: 
            xp /= max(abs(xp))

        self.xh = self.float2pcm(xh).astype(np.int16)
        self.xp = self.float2pcm(xp).astype(np.int16)

        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L
        self.den = self.calc_sum_squared_window(self.window, self.Hs)

    def play(self):
        """Direct play method for compatibility with existing code"""
        self.separate_hpss()
        self.setup_audio_stream()

        pos = 0
        try:
            while pos <= len(self.xh) - self.L:
                start = time.perf_counter()
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

                self.output_buffer = np.clip(self.output_buffer, -32768, 32767)
                self.stream.write(self.output_buffer[:self.Hs].astype(np.int16).tobytes())

                self.prev_fft = S
                pos += Ha
                self.runtimes.append(time.perf_counter() - start)

        except KeyboardInterrupt:
            print("\nPlayback interrupted.")

        finally:
            self.close_audio_stream()

            with open(f"runtimes_alpha_{self.alpha:.2f}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([[t] for t in self.runtimes])
                
            if self.on_complete:
                self.on_complete()
            complete = True

    def _run(self):
        """Threading implementation for consistency with base class"""
        self.play()