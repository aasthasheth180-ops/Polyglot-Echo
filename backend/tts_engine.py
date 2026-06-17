import os
import torch

# 1. Agree to the Terms of Service automatically so the script doesn't freeze
os.environ["COQUI_TOS_AGREED"] = "1"

# 2. Tell the computer to save the heavy voice weights to your spacious D: drive cache
os.environ["XDG_DATA_HOME"] = "D:/tts_cache"
os.environ["XDG_CACHE_HOME"] = "D:/tts_cache"

print("[*] Initializing Day 2 Setup: Redirecting storage pathways to D:/tts_cache...")

try:
    # 3. Bring in the TTS library toolbox
    from TTS.api import TTS
    
    # 4. Check if your graphics card (CUDA) is ready to help make the model fast
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Hardware acceleration selected: {DEVICE.upper()}")
    
    print("[*] Starting download for XTTS v2 model files. Please wait...")
    
    # 5. Tell the library to download the exact multilingual model we need onto your D: drive
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    
    print("[✓] SUCCESS: XTTS v2 model is fully downloaded and ready on your D: drive!")

except Exception as e:
    print(f"[-] An issue occurred during setup: {str(e)}")
    print("\n💡 TIP: If the terminal says 'ModuleNotFoundError', run this command first: pip install TTS")