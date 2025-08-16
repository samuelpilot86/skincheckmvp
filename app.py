import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Charger le modèle pré-entraîné (assurez-vous qu'il est uploadé dans le dépôt)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_lesion_model_binary.keras')

model = load_model()

# Fonction de prétraitement de l'image sans cv2
def preprocess_image(image, target_size=(224, 224)):
    try:
        # Convertir l'image PIL en RGB et redimensionner
        img = image.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        # Convertir en tableau numpy et normaliser
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        st.error(f"Erreur de prétraitement : {e}")
        return None

# Fonction de prédiction
def predict_user_image(image):
    img_array = preprocess_image(image)
    if img_array is None:
        return "Erreur : Impossible de traiter l'image."
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    threshold = 0.5  # Ajustez selon votre seuil optimal
    probability = prediction[0][0] * 100  # Index 0 pour 'mel' (à vérifier)
    if probability >= threshold * 100:
        return f"Malin - Probabilité : {probability:.2f}%. Consultez un dermatologue. (Prototype, pas garanti)"
    else:
        return f"Bénin - Probabilité : {(100 - probability):.2f}%. Consultez un dermatologue. (Prototype, pas garanti)"

# Interface Streamlit
st.title("Classificateur Naevus-Mélanomes")
st.write("Téléchargez une image de grain de beauté pour une classification expérimentale.")

uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("Analyse en cours...")
    result = predict_user_image(image)
    st.write(result)

# Avertissement
st.write("**Avertissement : Cet outil est un prototype. Consultez toujours un dermatologue pour un diagnostic officiel.**")