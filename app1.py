import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
st.set_page_config(page_title="Sound Classifier", layout="centered")
st.title("🔊 Real-Time Sound Classification")
# Load model
@st.cache_resource
def load_sound_model():
    model = tf.keras.models.load_model("sound_classifier_model_cnn.h5")  # Replace with your model path
    return model

model = load_sound_model()

# Define class names (update this list with your actual labels)
CLASS_NAMES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling','engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Preprocess function
def preprocess_audio(file_path, sr=22050, duration=4, offset=0.5, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    if len(audio) < sr * duration:
        pad_width = sr * duration - len(audio)
        audio = np.pad(audio, (0, int(pad_width)))
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = mel_spect[..., np.newaxis]  # add channel dimension
    return mel_spect

# Convert mp3 to wav using pydub
def convert_mp3_to_wav(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file, format="mp3")
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        return tmp.name

# Streamlit UI

st.write("Upload a `.wav` or `.mp3` file and let the model classify the sound!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save or convert uploaded file
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        if uploaded_file.name.endswith(".mp3"):
            temp_wav_path = convert_mp3_to_wav(uploaded_file)
        else:
            tmp_wav.write(uploaded_file.read())
            temp_wav_path = tmp_wav.name

    # Preprocess
    mel_spect = preprocess_audio(temp_wav_path)
    input_tensor = np.expand_dims(mel_spect, axis=0)

    # Predict
    prediction = model.predict(input_tensor)
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))

    # Output
    st.success(f"🎧 Predicted Class: **{predicted_class}**")
    st.write(f"🧠 Confidence: `{confidence:.2%}`")

