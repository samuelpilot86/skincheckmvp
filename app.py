import numpy as np
from PIL import Image, ImageOps  # Importation explicite
import os
import base64
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_FORCE_CPU_ONLY"] = "1"
import tensorflow as tf
import streamlit as st
from model_utils import focal_loss_fixed, MelanomaRecall, NevusSpecificity, CombinedMetric, ThresholdOptimizer, preprocess_image, predict_user_image
from st_clickable_images import clickable_images
from streamlit_cropper import st_cropper

# Charger le modèle
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "skin_lesion_model_binary.keras")
    if not os.path.exists(model_path):
        st.markdown(f'<div class="normal-text">Erreur : Le fichier modèle {model_path} n\'existe pas.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="normal-text">Répertoire de travail actuel : {os.getcwd()}</div>', unsafe_allow_html=True)
        return None
    try:
        custom_objects = {
            'focal_loss_fixed': focal_loss_fixed(gamma=1.0, alpha=0.9),
            'MelanomaRecall': MelanomaRecall,
            'NevusSpecificity': NevusSpecificity,
            'CombinedMetric': CombinedMetric,
            'ThresholdOptimizer': ThresholdOptimizer
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement du modèle : {e}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="normal-text">Répertoire de travail actuel : {os.getcwd()}</div>', unsafe_allow_html=True)
        return None

model = load_model()

# Interface Streamlit
st.set_page_config(page_title="SkinCheck", layout="centered")

# Charger le CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Création du logo et titre dans une variable HTML
logo_path = os.path.join("images", "logo_skincheck_transparent_reduit.png")
if os.path.exists(logo_path):
    try:
        logo_data = base64.b64encode(open(logo_path, "rb").read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" alt="Logo" width="46" height="auto">'
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement du logo : {e}</div>', unsafe_allow_html=True)
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

# Création des instructions de recadrage
reframed_mole_path = os.path.join("images", "FramedMole.jpg")
if os.path.exists(reframed_mole_path):
    try:
        reframed_mole_data = base64.b64encode(open(reframed_mole_path, "rb").read()).decode()
        reframed_mole_html = f'<img src="data:image/jpg;base64,{reframed_mole_data}" alt="" width="150" height="auto">'
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement de l\'image d\'exemple : {e}</div>', unsafe_allow_html=True)
        reframed_mole_html = ""
else:
    reframed_mole_html = ""

reframe_instructions_html = f'''
    <table class="instructions-table">
      <tr>
        <td><div class="normal-text">Move the frame to crop the picture so that the mole takes about half the space.</div></td>
        <td>{reframed_mole_html}</td>
      </tr>
    </table>
'''

# Création de l'avertissement "prototype non validée médicalement"
warning_html = f'''
    <table class="instructions-table">
      <tr>
        <td><div class="warning">⚠</div></td>
        <td><div class="warning-text">This prototype has not been validated by any medical authority. If you have any doubts, consult your dermatologist.</div></td>
      </tr>
    </table>
'''

# Navigation et mode
if 'screen' not in st.session_state:
    st.session_state.screen = "Accueil"

if st.session_state.screen == "Accueil":
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Take a photo of your mole* or choose an existing file. An artificial intelligence will try to determine if you should show it to a dermatologist.</div>', unsafe_allow_html=True)
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col_btn = st.columns([1, 1])
    with col_btn[0]:
        if st.button("Select/take photo", key="photo"):
            st.session_state.screen = "Photo"
            st.rerun()
    with col_btn[1]:
        if st.button("Select demo example", key="demo"):
            st.session_state.screen = "Examples"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(warning_html, unsafe_allow_html=True)
    st.markdown('<div class="bottom-note">*to French users: a mole is a “grain de beauté”.</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Photo":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    st.markdown('<div class="normal-text">Click Browse files to select or take a photo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Photos must be zoomed in while perfectly sharp.</div>', unsafe_allow_html=True)
   
    # File uploader stylisé
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="file_uploader")
   
    # Script JavaScript pour modifier le texte du bouton
    st.markdown("""
    <script>
        console.log('JavaScript: Script loaded at ' + new Date().toISOString());

        function updateUploaderButton() {
            console.log('JavaScript: Running updateUploaderButton at ' + new Date().toISOString());
            let uploaderButton = document.querySelector('[data-testid="stBaseButton-secondary"]');
            if (uploaderButton) {
                console.log('JavaScript: Button found, setting text to "Select/take photo"');
                uploaderButton.innerText = 'Select/take photo';
            } else {
                console.log('JavaScript: Button not found with selector [data-testid="stBaseButton-secondary"]');
            }
        }

        // Run immediately
        updateUploaderButton();

        // Fallback with delay
        setTimeout(updateUploaderButton, 500);

        // Fallback for Streamlit re-renders
        console.log('JavaScript: Starting MutationObserver');
        const observer = new MutationObserver(updateUploaderButton);
        const uploaderContainer = document.querySelector('[data-testid="stFileUploader"]');
        if (uploaderContainer) {
            console.log('JavaScript: File uploader container found');
            observer.observe(uploaderContainer, { childList: true, subtree: true });
        } else {
            console.log('JavaScript: File uploader container not found');
        }
    </script>
    """, unsafe_allow_html=True)
   
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if not isinstance(image, Image.Image):
                st.markdown('<div class="normal-text">Erreur : L\'image téléchargée est invalide.</div>', unsafe_allow_html=True)
            else:
                image = ImageOps.exif_transexo_transpose(image)
                st.session_state.image = image
                st.session_state.screen = "Reframe"
                st.rerun()
        except Exception as e:
            st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de l\'image : {e}</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Examples":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    st.markdown('<div class="normal-text">Click one of these 6 photos to analyze it.</div>', unsafe_allow_html=True)
    base_dir = os.path.join(os.getcwd(), "examples")
    benign_images = [os.path.join(base_dir, "benignmole1.jpg"), os.path.join(base_dir, "benignmole2.jpg"), os.path.join(base_dir, "benignmole3.jpg")]
    melanoma_images = [os.path.join(base_dir, "melanoma1.jpg"), os.path.join(base_dir, "melanoma2.jpg"), os.path.join(base_dir, "melanoma3.jpg")]
    
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            st.markdown(f'<div class="normal-text">Erreur : Fichier {image_path} non trouvé.</div>', unsafe_allow_html=True)
            return None
        except Exception as e:
            st.markdown(f'<div class="normal-text">Erreur lors du chargement de {image_path} : {e}</div>', unsafe_allow_html=True)
            return None
    
    benign_base64 = [image_to_base64(img) for img in benign_images if image_to_base64(img) is not None]
    melanoma_base64 = [image_to_base64(img) for img in melanoma_images if image_to_base64(img) is not None]
    
    if len(benign_base64) != 3 or len(melanoma_base64) != 3:
        st.markdown('<div class="normal-text">Erreur : Certaines images n\'ont pas pu être encodées en base64.</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="column-title">Benign moles:</div>', unsafe_allow_html=True)
            clicked_benign = clickable_images(
                [f"data:image/jpeg;base64,{b}" for b in benign_base64],
                titles=["", "", ""],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "background-color": "#F5F5F5"},
                img_style={"margin": "5px", "cursor": "pointer", "max-width": "150px", "height": "auto", "background-color": "#F5F5F5"}
            )
            if clicked_benign is not None and clicked_benign >= 0:
                img_path = benign_images[clicked_benign]
                try:
                    image = Image.open(img_path)
                    if not isinstance(image, Image.Image):
                        st.markdown(f'<div class="normal-text">Erreur : L\'image {img_path} est invalide.</div>', unsafe_allow_html=True)
                    else:
                        with st.spinner("Analysis in progress..."):
                            result, prob, color = predict_user_image(image, model)
                        st.session_state.screen = "Result"
                        st.session_state.image = img_path
                        st.session_state.result = (result, prob, color)
                        st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de {img_path} : {e}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="column-title">Melanomas:</div>', unsafe_allow_html=True)
            clicked_melanoma = clickable_images(
                [f"data:image/jpeg;base64,{m}" for m in melanoma_base64],
                titles=["", "", ""],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "background-color": "#F5F5F5"},
                img_style={"margin": "5px", "cursor": "pointer", "max-width": "150px", "height": "auto", "background-color": "#F5F5F5"}
            )
            if clicked_melanoma is not None and clicked_melanoma >= 0:
                img_path = melanoma_images[clicked_melanoma]
                try:
                    image = Image.open(img_path)
                    if not isinstance(image, Image.Image):
                        st.markdown(f'<div class="normal-text">Erreur : L\'image {img_path} est invalide.</div>', unsafe_allow_html=True)
                    else:
                        with st.spinner("Analysis in progress..."):
                            result, prob, color = predict_user_image(image, model)
                        st.session_state.screen = "Result"
                        st.session_state.image = img_path
                        st.session_state.result = (result, prob, color)
                        st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de {img_path} : {e}</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Reframe":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Photo"
        st.rerun()
    if 'image' in st.session_state:
        image = st.session_state.image
        width, height = image.size
        if width > 390:
            new_width = 390
            new_height = int(height * (new_width / width))
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            image_resized = image
        width, height = image_resized.size
        aspect_ratio = (3, 4) if height > width else (4, 3)
        
        st.markdown(reframe_instructions_html, unsafe_allow_html=True)
        
        cropped_image = st_cropper(image_resized, realtime_update=False, box_color='#4A90E2', aspect_ratio=aspect_ratio)
        
        if st.button("Analyze", key="analyze"):
            if cropped_image is not None and isinstance(cropped_image, Image.Image):
                with st.spinner("Analysis in progress..."):
                    result, prob, color = predict_user_image(cropped_image, model)
                st.session_state.screen = "Result"
                st.session_state.image = cropped_image
                st.session_state.result = (result, prob, color)
                st.rerun()
            else:
                st.markdown('<div class="normal-text">Erreur : L\'image recadrée n\'est pas valide. Veuillez valider le recadrage.</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Result":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    if 'result' in st.session_state:
        result, prob, color = st.session_state.result
        st.image(st.session_state.image, caption="", use_container_width=True)
        if result == "probably benign mole":
            st.markdown(f'<div style="background-color: #7ED321; color: white; padding: 10px; border-radius: 5px; text-align: center;">Result: {result}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="normal-text">This should be a benign mole.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="normal-text">Yet, if it is asymmetrical, has an irregular border, several colors, a diameter >6mm and/or has evolved recently, show it to a dermatologist.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #FFA500; color: white; padding: 10px; border-radius: 5px; text-align: center;">Result: {result}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="normal-text">This could be a melanoma, meaning a cluster of cancerous cells.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="normal-text">No need to worry, melanomas are highly treatable if detected early. Show it to a dermatologist in 2-4 weeks.</div>', unsafe_allow_html=True)
        st.markdown(warning_html, unsafe_allow_html=True)
        st.markdown('<div class="normal-text">New analysis:</div>', unsafe_allow_html=True)
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col_btn = st.columns([1, 1])
        with col_btn[0]:
            if st.button("Select/take photo", key="photo"):
                st.session_state.screen = "Photo"
                st.rerun()
        with col_btn[1]:
            if st.button("Select demo example", key="demo"):
                st.session_state.screen = "Examples"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
