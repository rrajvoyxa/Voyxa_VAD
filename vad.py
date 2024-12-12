#git clone https://github.com/snakers4/silero-vad.git
#pip install SpeechRecognition
#pip install pydub
#pip install omegaconf
#pip install pyaudio
#pip install torchaudio
#pip install numpy

import numpy as np
import torch
import pyaudio
import speech_recognition as sr

# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir="silero-vad",
    source="local",
    model='silero_vad',
    force_reload=True
)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# Helper function to convert int16 audio to float32
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    return sound

# Initialize the speech recognizer
r = sr.Recognizer()

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512  # 100ms chunks

# Function to process audio from the microphone
def vad_streaming_with_microphone():
    print("Listening... Press Ctrl+C to stop.")
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as mic:
            r.adjust_for_ambient_noise(mic)
            
            while True:
                # Capture audio from the microphone
                audio = r.listen(mic)
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                audio_float32 = int2float(audio_data)
                
                # Process with Silero VAD
                speech_timestamps = get_speech_timestamps(audio_float32, model, sampling_rate=SAMPLE_RATE)
                
                # Output results
                if speech_timestamps:
                    print("Speech detected!")
                else:
                    print("No speech detected.")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run VAD streaming
vad_streaming_with_microphone()
