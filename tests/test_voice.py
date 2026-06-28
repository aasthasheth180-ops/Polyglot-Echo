# test_all_languages.py
import os
import sys

# Ensure Python looks at the root project directory for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.tts_handler import VoiceCloner

def run_multilingual_master_test():
    print("🎙️ Starting Ultimate Multilingual Voice Calibration Test...")
    cloner = VoiceCloner()
    
    # 🎯 Map out the actual long translations so you hear your true cadence
    languages_to_test = {
        "English": {
            "code": "en",
            "text": "Good morning everyone! I hope you're having a wonderful day so far. I wanted to take a moment to talk with you about something that's been on my mind lately. You know, technology has changed so much over the past few years, and it keeps surprising me how incredible these voice synthesis models have become. I mean, just think about it – we can now create realistic human-like voices that sound natural and expressive."
        },
        "Hindi": {
            "code": "hi",
            "text": "शुभ प्रभात सभी को! मुझे आशा है कि आप सभी का दिन बहुत अच्छा चल रहा होगा। मैं आप सभी से कुछ बात करना चाहता था जो पिछले कुछ समय से मेरे दिमाग में है। आप जानते हैं, पिछले कुछ वर्षों में तकनीक में बहुत बदलाव आया है, और यह मुझे आश्चर्यचकित करता है कि ये आवाज संश्लेषण मॉडल कितने अविश्वसनीय हो गए हैं।"
        },
        "Gujarati": {
            "code": "gu",
            "text": "શુભ પ્રભાત સૌને! હું આશા રાખું છું કે તમારો દિવસ ખૂબ જ સુંદર જઈ રહ્યો હશે. હું તમારી સાથે એવી વાત કરવા માંગતો હતો જે છેલ્લા કેટલાક સમયથી મારા મગજમાં ચાલી રહી છે. તમે જાણો છો, છેલ્લા કેટલાક વર્ષોમાં ટેકનોલોજીમાં ઘણો બદલાવ આવ્યો છે, અને આ વોઈસ સિન્થેસિસ મોડલ્સ કેટલા અદ્ભુત બની ગયા છે તેનાથી મને આશ્ચર્ય થાય છે."
        },
        "Spanish": {
            "code": "es",
            "text": "¡Buenos días a todos! Espero que estén teniendo un día maravilloso. Quería tomarme un momento para hablar con ustedes sobre algo que ha estado en mi mente últimamente. Ya saben, la tecnología ha cambiado tanto en los últimos años, y me sigue sorprendiendo lo increíbles que se han vuelto estos modelos de síntesis de voz."
        }
    }
    
    reference_file = "audio/clip_1.wav"  # Your quiet, casual recording
    
    if not os.path.exists(reference_file):
        print(f"❌ ERROR: Put your recording at '{reference_file}' before running.")
        return

    # Run the loop cleanly
    for lang_name, lang_data in languages_to_test.items():
        output_file = f"audio/test_output_{lang_name}.wav"
        print(f"\n🌍 Processing Cloned Identity for: {lang_name}...")
        
        success = cloner.clone_voice(
            text_to_speak=lang_data["text"],
            reference_audio_path=reference_file,  # Clean string path
            output_wav_path=output_file,
            language=lang_data["code"]
        )
        
        if success:
            print(f"✓ Success! Generated {lang_name} audio: {output_file}")
        else:
            print(f"X Failed to generate {lang_name} audio.")

if __name__ == "__main__":
    run_multilingual_master_test()