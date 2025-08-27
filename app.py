import numpy as np
from PIL import Image, ImageOps
import os
import base64
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_FORCE_CPU_ONLY"] = "1"
import tensorflow as tf
from model_utils import focal_loss_fixed, MelanomaRecall, NevusSpecificity, CombinedMetric, ThresholdOptimizer
import streamlit as st

# Envelopper tout le contenu dans un conteneur principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Fonction pour charger dynamiquement les images
def load_examples(dynamic_dir="examples"):
    exemples_complets = {"benign": [], "melanoma": []}
    base_dir = os.path.join(os.getcwd(), dynamic_dir)
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        relative_path = os.path.relpath(file_path, os.getcwd())
                        label = f"{os.path.basename(os.path.dirname(file_path))} - {file}"
                        category = "benign" if "benign" in root.lower() else "melanoma"
                        exemples_complets[category].append((label, relative_path))
    else:
        st.write(f"Le répertoire {base_dir} n'existe pas.")
    return exemples_complets

# Charger les exemples dynamiquement
exemples_complets = load_examples()

# Charger le modèle
@st.cache_resource
def load_model():
    custom_objects = {
        'focal_loss_fixed': focal_loss_fixed(gamma=1.0, alpha=0.9),
        'MelanomaRecall': MelanomaRecall,
        'NevusSpecificity': NevusSpecificity,
        'CombinedMetric': CombinedMetric,
        'ThresholdOptimizer': ThresholdOptimizer
    }
    return tf.keras.models.load_model('skin_lesion_model_binary.keras', custom_objects=custom_objects)
model = load_model()

# Fonction de prétraitement
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
        return "Erreur : Impossible de traiter l'image.", None, None
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    threshold = 0.487
    probability = prediction[0][0] * 100
    if probability >= threshold * 100:
        return "Melanoma", probability, "red"
    else:
        return "Benign", (100 - probability), "green"

# Interface Streamlit
st.set_page_config(page_title="SkinCheck", layout="centered")

# Charger le CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Contenu principal dans .main-container
# Logo et titre dans un tableau HTML
logo_path = os.path.join("images", "logo_skincheck_transparent_reduit.png")
if os.path.exists(logo_path):
    try:
        logo_data = base64.b64encode(open(logo_path, "rb").read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" alt="Logo" width="46" height="auto">'
    except Exception as e:
        st.write(f"Erreur lors du chargement de l'image : {e}")
        logo_html = ""
else:
    logo_html = ""
html = f'''
    <table class="header-table">
      <tr>
        <td class="logo-cell">{logo_html}</td>
        <td class="title-cell">
          <div class="app-title"><span class="skin">Skin</span><span class="check">Check</span></div>
          <div class="subtitle">Should I show this mole to my dermatologist?</div>
        </td>
        <td class="empty-cell"></td>
      </tr>
    </table>
'''
st.markdown(html, unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True) # Ferme le conteneur principal
