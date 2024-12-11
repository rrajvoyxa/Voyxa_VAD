#pip install SpeechRecognition
#pip install pydub
#pip install omegaconf
#pip install pyaudio
#pip install torchaudio

import torch
from pydub import AudioSegment
import speech_recognition as sr
import io

device = torch.device('cpu')

model, decoder, utils = torch.hub.load(repo_or_dir='SYSTRAN/faster-whisper',
                                    #    'snakers4/silero-models',
                                       model='faster_whisper',
                                    #    'silero_stt',
                                       language='en',
                                       device=device)

# AVD

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as mic:
    r.adjust_for_ambient_noise(mic)
    print("Start Speaking....")
    while True:
        audio = r.listen(mic)
        audio = io.BytesIO(audio.get_wav_data())
        audio = AudioSegment.from_wav(audio)
        x     = torch.FloatTensor(audio.get_array_of_samples()).view(1, -1)
        x     = x.to(device)
        z     = model(x)
        print('You: ',decoder(z[0]))
