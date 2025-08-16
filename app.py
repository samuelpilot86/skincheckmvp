import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Charger le modèle pré-entraîné (assurez-vous qu'il est uploadé dans le dépôt ou accessible via un lien)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_lesion_model_binary.keras')

model = load_model()

# Fonction de prétraitement de l'image (tirée de votre code)
def preprocess_image(image, target_size=(224, 224)):
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if img.shape[:2] != target_size:
            img = cv2.copyMakeBorder(img, 0, max(0, target_size[0] - img.shape[0]), 0, max(0, target_size[1] - img.shape[1]),
                                    cv2.BORDER_REFLECT_101)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img / 255.0
    except Exception as e:
        st.error(f"Erreur de prétraitement : {e}")
        return None

# Fonction de prédiction (adaptée pour Streamlit)
def predict_user_image(image):
    img = preprocess_image(image)
    if img is None:
        return "Erreur : Impossible de traiter l'image."
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    threshold = 0.5  # Vous pouvez ajuster ce seuil basé sur votre seuil optimal trouvé
    probability = prediction[0][0] * 100  # Index 0 pour 'mel' (ajustez si nécessaire)
    if probability >= threshold * 100:
        return f"Malin - Probabilité : {probability:.2f}%. Consultez un dermatologue. (Prototype, pas garanti)"
    else:
        return f"Bénin - Probabilité : {(100 - probability):.2f}%. Consultez un dermatologue. (Prototype, pas garanti)"

# Interface Streamlit
st.title("Classificateur Naevus-Mélanomes")
st.write("Téléchargez une image de grain de beauté pour une classification expérimentale.")

uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("Analyse en cours...")
    result = predict_user_image(image)
    st.write(result)

# Avertissement
st.write("**Avertissement : Cet outil est un prototype. Consultez toujours un dermatologue pour un diagnostic officiel.**")