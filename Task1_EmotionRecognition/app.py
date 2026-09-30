import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import joblib
import os
import tempfile

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤"
)

st.title("🎤 Speech Emotion Recognition")
st.write("CodeAlpha Internship - Task 2")
st.success("Application is running successfully!")

model = joblib.load("models/emotion_model.pkl")

if "recordings" not in st.session_state:
    st.session_state.recordings = []

if "current_index" not in st.session_state:
    st.session_state.current_index = 0


st.subheader("🎙️ Record New Voice")

recorded_audio = st.audio_input(
    "Click to start recording"
)

if recorded_audio is not None:

    audio_bytes = recorded_audio.getvalue()

    exists = False

    for item in st.session_state.recordings:
        if item["data"] == audio_bytes:
            exists = True
            break

    if not exists:

        st.session_state.recordings.append({
            "name": f"Recording {len(st.session_state.recordings) + 1}",
            "data": audio_bytes,
            "extension": ".wav"
        })

        st.session_state.current_index = (
            len(st.session_state.recordings) - 1
        )


st.subheader("📁 Upload Old Voice")

uploaded_audio = st.file_uploader(
    "Upload an existing audio file",
    type=["wav", "mp3", "ogg", "m4a"]
)

if uploaded_audio is not None:

    audio_bytes = uploaded_audio.getvalue()

    exists = False

    for item in st.session_state.recordings:
        if item["data"] == audio_bytes:
            exists = True
            break

    if not exists:

        extension = os.path.splitext(
            uploaded_audio.name
        )[1].lower()

        st.session_state.recordings.append({
            "name": uploaded_audio.name,
            "data": audio_bytes,
            "extension": extension
        })

        st.session_state.current_index = (
            len(st.session_state.recordings) - 1
        )


if len(st.session_state.recordings) == 0:

    st.info(
        "Record a new voice or upload an old voice recording."
    )

else:

    total = len(st.session_state.recordings)

    if st.session_state.current_index >= total:
        st.session_state.current_index = total - 1

    current = st.session_state.recordings[
        st.session_state.current_index
    ]

    st.divider()

    st.subheader("🎧 Current Voice")

    st.write(current["name"])

    st.audio(current["data"])

    st.write(
        f"Voice {st.session_state.current_index + 1} of {total}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "◀ Previous",
            disabled=st.session_state.current_index == 0
        ):

            st.session_state.current_index -= 1
            st.rerun()

    with col2:

        if st.button(
            "Next ▶",
            disabled=st.session_state.current_index == total - 1
        ):

            st.session_state.current_index += 1
            st.rerun()

    with col3:

        if st.button("🗑️ Delete"):

            st.session_state.recordings.pop(
                st.session_state.current_index
            )

            if len(st.session_state.recordings) == 0:

                st.session_state.current_index = 0

            elif st.session_state.current_index >= len(
                st.session_state.recordings
            ):

                st.session_state.current_index = (
                    len(st.session_state.recordings) - 1
                )

            st.rerun()


    st.divider()

    if st.button("🤖 Predict Emotion"):

        temp_path = None

        try:

            extension = current["extension"]

            if extension == ".wav":

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".wav",
                    delete=False
                ) as file:

                    file.write(current["data"])
                    file.flush()

                    temp_path = file.name

                audio_data, sample_rate = sf.read(
                    temp_path,
                    dtype="float32"
                )

                if audio_data.ndim > 1:

                    audio_data = np.mean(
                        audio_data,
                        axis=1
                    )

            else:

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=extension,
                    delete=False
                ) as file:

                    file.write(current["data"])
                    file.flush()

                    temp_path = file.name

                audio_data, sample_rate = librosa.load(
                    temp_path,
                    sr=None,
                    mono=True
                )

            audio_data = np.asarray(
                audio_data,
                dtype=np.float32
            )

            if len(audio_data) == 0:
                st.error("Audio file is empty.")
                st.stop()

            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=40
            )

            features = np.mean(
                mfcc,
                axis=1
            ).reshape(1, -1)

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            classes = model.classes_

            st.subheader("🎯 Predicted Emotion")

            st.success(
                prediction.upper()
            )

            st.subheader(
                "📊 Emotion Prediction Percentage"
            )

            results = sorted(
                zip(classes, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            for emotion, probability in results:

                percentage = probability * 100

                st.write(
                    f"**{emotion.capitalize()}**: "
                    f"{percentage:.2f}%"
                )

                st.progress(
                    float(probability)
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)