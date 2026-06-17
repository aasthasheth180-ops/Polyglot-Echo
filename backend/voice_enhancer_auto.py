# backend/voice_enhancer_auto.py
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import io


def enhance_cloned_voice_auto(cloned_bytes: bytes, reference_audio_path: str) -> bytes:
    buffer = io.BytesIO(cloned_bytes)
    y, sr = sf.read(buffer)
    y = y.astype(np.float32)

    ref_y, ref_sr = librosa.load(reference_audio_path, sr=None)
    target_rms = np.sqrt(np.mean(ref_y**2))

    f0_ref, vf_ref, _ = librosa.pyin(ref_y, fmin=50, fmax=500, sr=ref_sr)
    target_pitch = np.nanmean(f0_ref[vf_ref])
    target_centroid = librosa.feature.spectral_centroid(y=ref_y, sr=ref_sr).mean()

    f0_clone, vf_clone, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    clone_pitch = np.nanmean(f0_clone[vf_clone])
    clone_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # STEP 1: Pitch correction
    if clone_pitch > 0 and target_pitch > 0:
        pitch_shift_steps = 12 * np.log2(target_pitch / clone_pitch)
        pitch_shift_steps = np.clip(pitch_shift_steps, -4, 4)
    else:
        pitch_shift_steps = 0.0

    print(f"[Auto] Pitch shift: {pitch_shift_steps:+.2f} semitones "
          f"(clone={clone_pitch:.1f}Hz -> target={target_pitch:.1f}Hz)")

    if abs(pitch_shift_steps) > 0.1:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_steps)

    # STEP 2: Loudness correction
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms > 0:
        y = y * (target_rms / current_rms)
        y = np.clip(y, -1.0, 1.0)

    print(f"[Auto] RMS target: {target_rms:.4f}")

    # STEP 3: Upsample
    y = librosa.resample(y, orig_sr=sr, target_sr=44100)
    sr = 44100

    # STEP 4: Harmonic-targeted brightness boost
    centroid_gap = target_centroid - clone_centroid

    if centroid_gap > 0:
        boost_amount = np.clip(centroid_gap / 1000 * 0.35, 0.0, 0.40)
        print(f"[Auto] Centroid gap: {centroid_gap:+.0f}Hz -> Harmonic boost: {boost_amount:.2f}")

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        sos = signal.butter(2, 1200, btype='high', fs=sr, output='sos')
        y_harmonic_highs = signal.sosfilt(sos, y_harmonic)

        y = y + (boost_amount * y_harmonic_highs)
        y = np.clip(y, -1.0, 1.0)
    else:
        print(f"[Auto] Centroid gap: {centroid_gap:+.0f}Hz -> no boost needed")

    out = io.BytesIO()
    sf.write(out, y, sr, format="WAV")
    out.seek(0)
    return out.read()