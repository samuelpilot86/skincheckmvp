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
 
# Logo et titre dans un tableau HTML - définition en amont pour pouvoir la réutiliser dans plusieurs pages
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
title_html = f'''
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


# Navigation et mode
if 'screen' not in st.session_state:
    st.session_state.screen = "Accueil"
if st.session_state.screen == "Accueil":
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Take a photo of your mole*. An artificial intelligence will try to determine if you should show it to a dermatologist.</div>', unsafe_allow_html=True)
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col_btn = st.columns([1, 1, 1])
    with col_btn[0]:
        if st.button("Take a photo", key="take_photo"):
            st.session_state.screen = "Photo"
            st.rerun()
    with col_btn[1]:
        if st.button("Browse phone photos", key="browse"):
            st.session_state.screen = "Browse"
            st.rerun()
    with col_btn[2]:
        if st.button("Select demo example", key="demo"):
            st.session_state.screen = "Examples"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-note">*to French users: a mole is a “grain de beauté”.</div>', unsafe_allow_html=True)
   
elif st.session_state.screen == "Photo":
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    st.markdown('<div class="photo-section">Take a sharp photo as close as possible</div>', unsafe_allow_html=True)
    captured_file = st.camera_input("")
    if captured_file is not None:
        image = Image.open(captured_file)
        image = ImageOps.exif_transpose(image)
        st.session_state.image = image
        st.session_state.screen = "Reframe"
        st.rerun()
elif st.session_state.screen == "Browse":
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()  # Ajouté pour revenir à l'écran "Accueil" avec un seul clic
    st.markdown('<div class="photo-section">Choose an image (JPG/PNG)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        st.session_state.image = image
        st.session_state.screen = "Reframe"
        st.rerun()

elif st.session_state.screen == "Examples":
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=False, width=46, output_format="PNG", channels="RGB", caption="")
    st.markdown('<div class="app-title"><span class="skin">Skin</span><span class="check">Check</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Should I show this mole to my dermatologist?</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Définition des chemins des images fixes (ajustement du chemin)
    base_dir = os.path.join(os.getcwd(), "examples")
    st.write(f"Répertoire de travail : {os.getcwd()}")  # Débogage
    st.write(f"Contenu de {base_dir}: {os.listdir(base_dir) if os.path.exists(base_dir) else 'Non trouvé'}")  # Débogage
    benign_images = [
        os.path.join(base_dir, "Grain de beauté 1.jpg"),
        os.path.join(base_dir, "Grain de beauté 2.jpg"),
        os.path.join(base_dir, "Grain de beauté 3.jpg")
    ]
    melanoma_images = [
        os.path.join(base_dir, "Mélanome 1.jpg"),
        os.path.join(base_dir, "Mélanome 2.jpg"),
        os.path.join(base_dir, "Mélanome 3.jpg")
    ]

    # HTML pour le tableau avec images cliquables (placeholders numérotés)
    table_html = """
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <th style="text-align: center; padding: 10px; background-color: #f0f0f0;">Benign moles</th>
                <th style="text-align: center; padding: 10px; background-color: #f0f0f0;">Melanomas</th>
            </tr>
            <tr>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{0}" alt="Benign 1" class="clickable-image" data-path="{1}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{2}" alt="Melanoma 1" class="clickable-image" data-path="{3}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{4}" alt="Benign 2" class="clickable-image" data-path="{5}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{6}" alt="Melanoma 2" class="clickable-image" data-path="{7}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{8}" alt="Benign 3" class="clickable-image" data-path="{9}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
                <td style="padding: 10px; text-align: center;">
                    <img src="data:image/jpeg;base64,{10}" alt="Melanoma 3" class="clickable-image" data-path="{11}" style="width: 150px; height: auto; cursor: pointer;">
                </td>
            </tr>
        </table>
        <script>
            document.querySelectorAll('.clickable-image').forEach(img => {
                img.addEventListener('click', function() {
                    const path = this.getAttribute('data-path');
                    const imgElement = new Image();
                    imgElement.src = this.src;
                    imgElement.onload = function() {
                        const result = { result: "Simulated", probability: 50, color: "yellow" }; // Remplacez par predict_user_image si possible
                        window.parent.postMessage({
                            type: 'updateSession',
                            screen: 'Result',
                            image: path,
                            result: result.result,
                            probability: result.probability,
                            color: result.color
                        }, '*');
                    };
                });
            });
            window.addEventListener('message', function(event) {
                if (event.data.type == 'updateSession') {
                    window.parent.postMessage({
                        type: 'updateSession',
                        screen: event.data.screen,
                        image: event.data.image,
                        result: event.data.result,
                        probability: event.data.probability,
                        color: event.data.color
                    }, '*');
                }
            });
        </script>
    """

    # Conversion des images en base64 avec vérification
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            st.write(f"Erreur : Fichier {image_path} non trouvé.")
            return ""
        except Exception as e:
            st.write(f"Erreur lors du chargement de {image_path} : {e}")
            return ""

    benign_base64 = [image_to_base64(img) for img in benign_images]
    melanoma_base64 = [image_to_base64(img) for img in melanoma_images]

    # Vérification des données avant formatage
    if len(benign_base64) != 3 or len(melanoma_base64) != 3 or any(b == "" for b in benign_base64) or any(m == "" for m in melanoma_base64):
        st.write("Erreur : Certaines images n'ont pas pu être chargées. Vérifiez les chemins.")
    else:
        # Remplissage du HTML avec les images encodées
        filled_table_html = table_html.format(
            benign_base64[0], benign_images[0],
            melanoma_base64[0], melanoma_images[0],
            benign_base64[1], benign_images[1],
            melanoma_base64[1], melanoma_images[1],
            benign_base64[2], benign_images[2],
            melanoma_base64[2], melanoma_images[2]
        )
        st.markdown(filled_table_html, unsafe_allow_html=True)

elif st.session_state.screen == "Reframe":
    if st.button("←", key="back"):
        st.session_state.screen = "Photo"
        st.rerun()  
    if 'image' in st.session_state:
        image = st.session_state.image
        st.image(image, caption="Frame the picture so that the mole takes half the space", use_container_width=True)
        st.markdown(f'<div class="normal-text">Current size: {image.size[0]} x {image.size[1]}</div>', unsafe_allow_html=True)
        if st.button("Reframe", key="reframe"):
            st.warning("Veuillez recadrer manuellement l'image pour que la lésion occupe environ la moitié de l'espace.")
        if st.button("Analyze", key="analyze"):
            with st.spinner("Analysis in progress..."):
                result, prob, color = predict_user_image(image)
            st.session_state.screen = "Result"
            st.session_state.result = (result, prob, color)
            st.rerun() 
elif st.session_state.screen == "Result":
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun() 
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=False, width=46, output_format="PNG", channels="RGB", caption="")
    st.markdown('<div class="app-title"><span class="skin">Skin</span><span class="check">Check</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Should I show this mole to my dermatologist?</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if 'result' in st.session_state:
        result, prob, color = st.session_state.result
        st.image(st.session_state.image, caption="Analysis result:", use_container_width=True)
        st.markdown(f'<div class="normal-text">This should be a {result} mole. Yet, if it is asymmetrical, has an irregular border, several colors, a diameter >6mm and/or has evolved recently, show it to a dermatologist.</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">{result}</div>', unsafe_allow_html=True)
        st.markdown('<div class="normal-text">New analysis:</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            col_btn = st.columns([1, 1, 1])
            with col_btn[0]:
                if st.button("Take a photo"):
                    st.session_state.screen = "Photo"
                    st.rerun() 
            with col_btn[1]:
                if st.button("Browse phone photos"):
                    st.session_state.screen = "Browse"
                    st.rerun() 
            with col_btn[2]:
                if st.button("Select demo example"):
                    st.session_state.screen = "Examples"
                    st.rerun()  
            st.markdown('</div>', unsafe_allow_html=True)
