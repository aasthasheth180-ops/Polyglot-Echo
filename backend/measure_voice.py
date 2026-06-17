# measure_voice.py
import librosa
import numpy as np

def measure_voice(audio_path: str, label: str = "") -> dict:
    """Measure the 3 key metrics for any voice clip. Reusable for anyone."""
    y, sr = librosa.load(audio_path, sr=None)
    
    rms = np.sqrt(np.mean(y**2))
    
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[voiced_flag]
    pitch_mean = np.nanmean(f0_voiced)
    pitch_std = np.nanstd(f0_voiced)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    result = {
        "rms": rms,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "centroid": centroid
    }
    
    print(f"--- {label} ---")
    print(f"RMS:              {rms:.4f}")
    print(f"Pitch mean:       {pitch_mean:.1f} Hz")
    print(f"Pitch std:        {pitch_std:.1f}")
    print(f"Spectral centroid:{centroid:.0f} Hz")
    print()
    
    return result


if __name__ == "__main__":
    # Step 1: measure your friend's real voice
    real = measure_voice("audio/friend_reference.wav", "Friend's real voice")
    
    # Step 2: measure the RAW clone (before enhancement)
    raw_clone = measure_voice("audio/friend_raw_clone.wav", "Raw XTTS output (unenhanced)")
    
    # Step 3: compute the gaps
    print("=== GAPS TO CORRECT ===")
    
    # Pitch gap → convert to semitones
    if raw_clone["pitch_mean"] > 0 and real["pitch_mean"] > 0:
        pitch_shift = 12 * np.log2(real["pitch_mean"] / raw_clone["pitch_mean"])
        print(f"Pitch shift needed: {pitch_shift:+.2f} semitones")
    
    # RMS target
    print(f"Target RMS: {real['rms']:.4f}")
    
    # Centroid gap
    centroid_gap = real["centroid"] - raw_clone["centroid"]
    print(f"Centroid gap: {centroid_gap:+.0f} Hz (positive = clone is too dull, needs boost)")