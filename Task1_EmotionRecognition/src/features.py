import librosa
import numpy as np


def extract_features_from_audio(audio, sample_rate):

    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    if len(audio) == 0:
        raise ValueError("No usable audio detected.")

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    audio = librosa.effects.preemphasis(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio
    )

    rms = librosa.feature.rms(
        y=audio
    )

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.std(spectral_contrast, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.std(zero_crossing_rate, axis=1),
        np.mean(rms, axis=1),
        np.std(rms, axis=1)
    ])

    return feature_vector


def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    return extract_features_from_audio(
        audio,
        sample_rate
    )


def add_noise(audio, noise_level=0.005):

    noise = np.random.normal(
        0,
        noise_level,
        len(audio)
    )

    return audio + noise


def change_volume(audio, factor=0.8):

    return audio * factor


def change_pitch(audio, sample_rate, steps=1):

    return librosa.effects.pitch_shift(
        y=audio,
        sr=sample_rate,
        n_steps=steps
    )