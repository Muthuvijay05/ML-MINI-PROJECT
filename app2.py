import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
st.set_page_config(page_title="Sound Classifier", layout="centered")
st.title("🔊 Sound Classification Using CNN")
# Load your model
model = load_model("sound_classifier_model_cnn.h5")
class_labels = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]
# Load and preprocess the dog bark MP3
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, duration=4.0)
    mels = librosa.feature.melspectrogram(y=audio, sr=sr)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    mels_mean = np.mean(mels_db.T, axis=0)
    
    if len(mels_mean) < 128:
        mels_mean = np.pad(mels_mean, (0, 128 - len(mels_mean)))
    else:
        mels_mean = mels_mean[:128]
    
    return mels_mean.reshape(1, 16, 8, 1)


st.title("Upload a File")

file_path= st.file_uploader("Choose a file", type=["mp3", "wav"])

if file_path is not None:
    processed = preprocess_audio(file_path)
    prediction = model.predict(processed)
    predicted_class_id = int(np.argmax(prediction))
    predicted_class=class_labels[np.argmax(prediction)]

    print(f"Predicted Class ID: {predicted_class}")
    if st.button('analyse'):
        st.write(f"Predicted Class ID: {predicted_class}")
