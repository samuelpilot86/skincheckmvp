import numpy as np
from PIL import Image, ImageOps
import os
import base64
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["OMP_NUM_THREADS"] = "8"    
os.environ["TF_FORCE_CPU_ONLY"] = "1"   
import tensorflow as tf  
import streamlit as st 
from model_utils import focal_loss_fixed, MelanomaRecall, NevusSpecificity, CombinedMetric, ThresholdOptimizer, load_model, preprocess_image, predict_user_image
from st_clickable_images import clickable_images
from streamlit_cropper import st_cropper

# Charger le modèle 
model = load_model()

# Interface Streamlit
st.set_page_config(page_title="SkinCheck", layout="centered")

# Charger le CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Création du logo et titre dans une variable HTML - définition en amont pour pouvoir la réutiliser dans plusieurs pages
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

# Création des instructions de recadrage
reframed_mole_path = os.path.join("images", "FramedMole.jpg")
if os.path.exists(reframed_mole_path):
    try:
        reframed_mole_data = base64.b64encode(open(reframed_mole_path, "rb").read()).decode()
        reframed_mole_html = f'<img src="data:image/jpg;base64,{reframed_mole_data}" alt="" width="150" height="auto">'
    except Exception as e:
        st.write(f"Erreur lors du chargement de l'image : {e}")
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
    st.markdown('<div class="bottom-note">*to French users: a mole is a “grain de beauté”.</div>', unsafe_allow_html=True)
  
elif st.session_state.screen == "Photo":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    st.markdown('<div class="normal-text">Click \'Browse files\' to select a photo or take one (phone only).</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Ensure the photo is perfectly sharp and as zoomed in as possible.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png"], key="file_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        st.session_state.image = image
        st.session_state.screen = "Reframe"
        st.rerun()

elif st.session_state.screen == "Examples":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
    
    # Définition des chemins des images fixes
    base_dir = os.path.join(os.getcwd(), "examples")
    benign_images = [
        os.path.join(base_dir, "benignmole1.jpg"),
        os.path.join(base_dir, "benignmole2.jpg"),
        os.path.join(base_dir, "benignmole3.jpg")
    ]
    melanoma_images = [
        os.path.join(base_dir, "melanoma1.jpg"),
        os.path.join(base_dir, "melanoma2.jpg"),
        os.path.join(base_dir, "melanoma3.jpg")
    ]

    # Conversion des images en base64
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            st.write(f"Erreur : Fichier {image_path} non trouvé.")
            return None
        except Exception as e:
            st.write(f"Erreur lors du chargement de {image_path} : {e}")
            return None

    benign_base64 = [image_to_base64(img) for img in benign_images if image_to_base64(img) is not None]
    melanoma_base64 = [image_to_base64(img) for img in melanoma_images if image_to_base64(img) is not None]

    # Vérification des données
    if len(benign_base64) != 3 or len(melanoma_base64) != 3:
        st.write("Erreur : Certaines images n'ont pas pu être encodées en base64.")
    else:
        # Affichage des en-têtes et images cliquables avec fond blanc cassé
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="column-title">Benign moles</div>', unsafe_allow_html=True)
            clicked_benign = clickable_images(
                [f"data:image/jpeg;base64,{b}" for b in benign_base64],
                titles=["", "", ""],  # Vides pour éviter les légendes
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "background-color": "#F5F5F5"},
                img_style={"margin": "5px", "cursor": "pointer", "max-width": "150px", "height": "auto", "background-color": "#F5F5F5"}
            )
            if clicked_benign is not None and clicked_benign >= 0:
                img_path = benign_images[clicked_benign]
                image = Image.open(img_path)
                with st.spinner("Analysis in progress..."):
                    result, prob, color = predict_user_image(image)
                st.session_state.screen = "Result"
                st.session_state.image = img_path
                st.session_state.result = (result, prob, color)
                st.rerun()
        
        with col2:
            st.markdown('<div class="column-title">Melanomas</div>', unsafe_allow_html=True)
            clicked_melanoma = clickable_images(
                [f"data:image/jpeg;base64,{m}" for m in melanoma_base64],
                titles=["", "", ""],  # Vides pour éviter les légendes
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "background-color": "#F5F5F5"},
                img_style={"margin": "5px", "cursor": "pointer", "max-width": "150px", "height": "auto", "background-color": "#F5F5F5"}
            )
            if clicked_melanoma is not None and clicked_melanoma >= 0:
                img_path = melanoma_images[clicked_melanoma]
                image = Image.open(img_path)
                with st.spinner("Analysis in progress..."):
                    result, prob, color = predict_user_image(image)
                st.session_state.screen = "Result"
                st.session_state.image = img_path
                st.session_state.result = (result, prob, color)
                st.rerun()
               
elif st.session_state.screen == "Reframe":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Photo"
        st.rerun()
    if 'image' in st.session_state:
        image = st.session_state.image
        # Redimensionner l'image pour une largeur maximale de 390px tout en conservant les proportions
        width, height = image.size
        if width > 390:
            new_width = 390
            new_height = int(height * (new_width / width))
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            image_resized = image

        # Déterminer l'aspect ratio selon l'orientation de l'image redimensionnée
        width, height = image_resized.size
        if height > width:  # Portrait
            aspect_ratio = (3, 4)  # 4 hauteur pour 3 largeur
        else:  # Paysage
            aspect_ratio = (4, 3)  # 3 hauteur pour 4 largeur
        
        st.markdown(reframe_instructions_html, unsafe_allow_html=True)
        #st.markdown(f'<div class="normal-text">Move the frame to crop the picture so that the mole takes about half the space.</div>', unsafe_allow_html=True)
        #st.markdown(f'<div class="normal-text"> </div>', unsafe_allow_html=True)
        # Stocker les coordonnées du recadrage (st_cropper retourne l'image recadrée uniquement après validation)
        cropped_image = st_cropper(image_resized, realtime_update=False, box_color='#4A90E2', aspect_ratio=aspect_ratio)
        
        if st.button("Analyze", key="analyze"):
            with st.spinner("Analysis in progress..."):
                result, prob, color = predict_user_image(cropped_image)
            st.session_state.screen = "Result"
            st.session_state.image = cropped_image
            st.session_state.result = (result, prob, color)
            st.rerun()

elif st.session_state.screen == "Result":
    st.markdown(title_html, unsafe_allow_html=True)
    if st.button("←", key="back"):
        st.session_state.screen = "Accueil"
        st.rerun()
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
