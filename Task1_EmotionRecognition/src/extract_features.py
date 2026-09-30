import librosa
import numpy as np
import pandas as pd
import glob
import os

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

files = glob.glob("data/raw/**/*.wav", recursive=True)

features = []
labels = []

for file in files:
    filename = os.path.basename(file)
    emotion_code = filename.split("-")[2]
    emotion = emotion_map[emotion_code]

    audio, sample_rate = librosa.load(file, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    features.append(mfcc_mean)
    labels.append(emotion)

df = pd.DataFrame(features)
df["emotion"] = labels

df.to_csv("data/features.csv", index=False)

print("Total files processed:", len(files))
print("Features saved to: data/features.csv")
print("Feature shape:", df.shape)