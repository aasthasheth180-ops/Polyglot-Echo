# backend/voice_enhancer.py
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import io

def enhance_cloned_voice(cloned_bytes: bytes, pitch_shift_steps: float = 0.0, target_rms: float = 0.0463) -> bytes:
    """
    Studio mixing board to polish the AI generated voice.
    """
    buffer = io.BytesIO(cloned_bytes)
    y, sr = sf.read(buffer)
    y = y.astype(np.float32)
    
    # Pitch correction (0.0 = no shift needed at this point)
    if pitch_shift_steps != 0.0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_steps)
    
    # Loudness normalization
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms > 0:
        scale = target_rms / current_rms
        y = y * scale
        y = np.clip(y, -1.0, 1.0)
        
    # Upsample to 44.1kHz
    y = librosa.resample(y, orig_sr=sr, target_sr=44100)
    sr = 44100
    
    # High-shelf EQ boost (single pass, no preemphasis)
    sos = signal.butter(2, 1500, btype='high', fs=sr, output='sos')
    y_highs = signal.sosfilt(sos, y)
    y = y + (0.22 * y_highs)
    y = np.clip(y, -1.0, 1.0)
    
    out = io.BytesIO()
    sf.write(out, y, sr, format="WAV")
    out.seek(0)
    return out.read()