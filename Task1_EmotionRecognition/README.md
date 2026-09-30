# 🎤 Speech Emotion Recognition

A Speech Emotion Recognition project developed as part of the CodeAlpha Internship.

## 📌 Project Description

This application analyzes a speech audio recording and predicts the emotion expressed in the voice.

The application supports:
- 🎙️ Recording a voice
- 📁 Uploading an existing audio file
- ◀ Previous / Next recordings
- 🗑️ Deleting recordings
- 🤖 Emotion prediction
- 📊 Emotion probability percentages

## 🧠 Emotions Detected

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## 🛠️ Technologies Used

- Python
- Streamlit
- Librosa
- NumPy
- Pandas
- Scikit-learn
- SoundFile
- Joblib
- SVM
- MFCC

## 🎧 Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) speech dataset.

The raw dataset is not included in this repository.

## 🤖 Machine Learning Model

Audio features are extracted using 40 MFCC (Mel-Frequency Cepstral Coefficient) features.

An SVM (Support Vector Machine) classifier is used for emotion classification.

## 📊 Model Performance

Accuracy: 53.82%

Precision: 53.43%

Recall: 53.82%

F1-Score: 52.14%

## 🚀 Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK