import os
import torch
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Execution Runtime: {DEVICE.upper()}")

print("[*] Loading Whisper weights into D: Drive Cache...")
# 🎯 Redirects downloads to D: drive to keep your C: drive safe!
model = whisper.load_model("small", device="cpu", download_root="D:/whisper_cache")
if DEVICE == "cuda":
    model = model.to(DEVICE)

def test_single_file(audio_path):
    if not os.path.exists(audio_path):
        print(f"[-] Error: {audio_path} not found.")
        return

    print(f"[*] Processing audio payload: {audio_path}")
    
    # Force strict Gujarati decoding parameters
    result = model.transcribe(
        audio_path,
        task="translate",                  # 🎯 Output directly to English
        language="gu",                    # ⚓ Source Anchor: Locks decoder to Gujarati phonemes
        temperature=0.0,                  # 🥶 No creative wandering
        compression_ratio_threshold=2.0,   # 🛑 Kills character repetition loops
        no_speech_threshold=0.5,          # 🤫 Filters line hiss
        condition_on_previous_text=False,  # 🧠 Memory Wipe: Clears out translation drift
        fp16=False
    )
    
    print("\n[✓] TARGET TRANSLATION RESULT:")
    print("-" * 50)
    print(f"Text Output: {result.get('text', '').strip()}")
    print(f"Detected Language Override: {result.get('language', '')}")
    print("-" * 50)

if __name__ == "__main__":
    test_single_file("audio/Guj.wav")