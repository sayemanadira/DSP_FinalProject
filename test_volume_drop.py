import numpy as np
import librosa

def test_volume_drop(beta, freq=1000, sr=22050):
    L = 2048
    Ha = int(beta * L)
    t = np.arange(L) / sr
    x = np.sin(2 * np.pi * freq * t)  # Test tone
    
    # STFT with specified hop
    S = librosa.stft(x, n_fft=L, hop_length=Ha, window='hann')
    
    # ISTFT (reconstruct without modification)
    y = librosa.istft(S, hop_length=Ha, window='hann')
    
    # Measure volume difference
    orig_power = np.mean(x**2)
    recon_power = np.mean(y**2)
    return recon_power / orig_power  # Ratio ≈ 1.0 means no volume change

print(f"β=0.1: Volume ratio = {test_volume_drop(0.1):.4f}")
print(f"β=0.25: Volume ratio = {test_volume_drop(0.25):.4f}")