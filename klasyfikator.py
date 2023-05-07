import pyaudio
import numpy as np
import webrtcvad
from klasyfikator_funkcje import open_file, recognize_intonation

# import os
# print(os.path.abspath(''))

# wytrenowane modele gmm
gmms = open_file("GMM_EMOTIVE/gender/gmms")
gmms_female = open_file("GMM_EMOTIVE/f0/gmms_F_cms")   
gmms_male = open_file("GMM_EMOTIVE/f0/gmms_M_cms")

# VAD
vad = webrtcvad.Vad()
vad.set_mode(2)
frame_duration = 20  # 10 ms, *2 (bytes)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WINDOW_LEN = 2  # window length in seconds
CHUNK = 1000
CHUNK_vad = int(RATE * frame_duration / 1000)  # number of bytes in 10 ms

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

try:
    while True:
        frame = []
        is_speech = []
        frame_bytes = []
        # read data from mic (2 sec)
        for i in range(0, int(RATE / CHUNK * WINDOW_LEN)):
            data = stream.read(CHUNK)
            frame_bytes.append(data)
            numpydata = np.frombuffer(data, dtype=np.int16)
            frame.append(numpydata)
            for j in range(0, CHUNK, CHUNK_vad):
                is_speech.append(vad.is_speech(data[j : j + CHUNK_vad], RATE))
        true_num = is_speech.count(True)
        false_num = is_speech.count(False)
        if false_num < true_num:
            frame = np.hstack(frame)
            frame_float = np.array(frame).astype(np.float32)
            (
                audio_scores_gender,
                class_prediction_gender,
                audio_scores,
                class_prediction_intonation,
            ) = recognize_intonation(frame_float, gmms, gmms_female, gmms_male, fs=RATE)
            print(audio_scores_gender, class_prediction_gender)
            print(audio_scores, class_prediction_intonation)
        else:
            print("unvoiced segment")
except KeyboardInterrupt:
    pass
