import librosa
import numpy as np
import joblib

model = joblib.load("models/emotion_model.pkl")

file_path = "data/raw/Audio_Speech_Actors_01-24_16k/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(file_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

features = np.mean(mfcc, axis=1).reshape(1, -1)

prediction = model.predict(features)

print("Predicted Emotion:", prediction[0])