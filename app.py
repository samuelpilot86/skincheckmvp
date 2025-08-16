# Si jamais ce code devait être modifié, le faire sans la bibliothèque cv2, que Streamlit a régulièrement du mal à importer

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import keras.backend as K

# Définir la fonction focal_loss_fixed (identique à celle de votre code d'entraînement)
def focal_loss_fixed(gamma=1.0, alpha=0.9, class_weights=None):
    def focal_loss_fixed_internal(y_true, y_pred):
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = K.clip(pt, K.epsilon(), 1 - K.epsilon())
        loss = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        if class_weights is not None:
            weight_mask = tf.stack([y_true[:, idx] * class_weights[idx] for idx in range(2)], axis=1)
            loss = loss * weight_mask
        return K.mean(loss, axis=-1)
    return focal_loss_fixed_internal

# Charger le modèle pré-entraîné avec la fonction personnalisée
@st.cache_resource
def load_model():
    custom_objects = {'focal_loss_fixed': focal_loss_fixed(gamma=1.0, alpha=0.9)}
    return tf.keras.models.load_model('skin_lesion_model_binary.keras', custom_objects=custom_objects)

model = load_model()

# Fonction de prétraitement de l'image sans cv2
def preprocess_image(image, target_size=(224, 224)):
    try:
        img = image.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
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
    threshold = 0.5  # Ajustez selon votre seuil optimal (par ex. 0.XXX basé sur votre entraînement)
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