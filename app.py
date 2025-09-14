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
from model_utils import focal_loss_fixed, MelanomaRecall, NevusSpecificity, CombinedMetric, ThresholdOptimizer, preprocess_image, predict_user_image
from st_clickable_images import clickable_images
from streamlit_cropper import st_cropper
from cryptography.fernet import Fernet

# Fonction pour afficher le bouton de retour
def display_back_button():
    with st.container():
        st.markdown(
            f"""
            <style>
            /* Style spécifique pour le bouton de retour basé sur la clé dynamique */
            [data-testid="stBaseButton-secondary"][data-key="back_to_previous_{st.session_state.screen}"] {{
                background-color: #333333 !important; /* Fond gris foncé */
                border: none !important;
                padding: 10px 20px !important; /* Padding cohérent avec les autres boutons */
                cursor: pointer !important;
                font-size: 18px !important; /* Taille cohérente avec les autres boutons */
                color: #808080 !important; /* Gris pour le texte/flèche */
                border-radius: 5px !important;
                width: 100% !important; /* Remplir la colonne */
                text-align: center !important;
                box-sizing: border-box !important;
            }}
            [data-testid="stBaseButton-secondary"][data-key="back_to_previous_{st.session_state.screen}"]:hover {{
                background-color: #222222 !important; /* Gris plus foncé au survol */
                color: #606060 !important; /* Texte plus foncé au survol */
            }}
            /* Assurer que les autres boutons restent bleus */
            [data-testid="stBaseButton-secondary"]:not([data-key="back_to_previous_{st.session_state.screen}"]) {{
                background-color: #4A90E2 !important;
                color: #F5F5F5 !important;
            }}
            [data-testid="stBaseButton-secondary"]:not([data-key="back_to_previous_{st.session_state.screen}"]):hover {{
                background-color: #3A7AC2 !important;
            }}
            /* Masquer le texte par défaut et utiliser une icône SVG */
            [data-testid="stBaseButton-secondary"][data-key="back_to_previous_{st.session_state.screen}"] div[data-testid="stMarkdownContainer"] p {{
                display: none !important;
            }}
            [data-testid="stBaseButton-secondary"][data-key="back_to_previous_{st.session_state.screen}"]::before {{
                content: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTE1LjQxIDE2LjU5TDEwLjgzIDEybDQuNTgtNC41OUwxNCA2bC02IDYgNiA2eiIgZmlsbD0iIzgwODA4MCIvPjwvc3ZnPg==');
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 24px;
                height: 24px;
            }}
            [data-testid="stBaseButton-secondary"][data-key="back_to_previous_{st.session_state.screen}"]:hover::before {{
                content: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTE1LjQxIDE2LjU5TDEwLjgzIDEybDQuNTgtNC41OUwxNCA2lC02IDYgNiA2eiIgZmlsbD0iIzYwNjA2MCIvPjwvc3ZnPg==');
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        # Déterminer l'écran précédent
        if 'screen_history' not in st.session_state:
            st.session_state.screen_history = []
        if st.session_state.screen not in st.session_state.screen_history:
            st.session_state.screen_history.append(st.session_state.screen)
        
        if len(st.session_state.screen_history) > 1:
            previous_screen = st.session_state.screen_history[-2]  # Deuxième élément depuis la fin
            # Forcer le retour à "Accueil" depuis "Examples"
            if st.session_state.screen == "Examples":
                previous_screen = "Accueil"
        else:
            previous_screen = "Accueil"  # Par défaut, revenir à Accueil si aucun écran précédent
        
        # Placer le bouton dans une colonne pour suivre le layout naturel
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("", key=f"back_to_previous_{st.session_state.screen}", help="Retour"):
                if previous_screen in st.session_state.screen_history or previous_screen == "Accueil":
                    st.session_state.screen = previous_screen
                    if st.session_state.screen != "Accueil":  # Ne pas pop si on revient à Accueil
                        st.session_state.screen_history.pop()  # Supprimer l'écran actuel de l'historique
                    st.rerun()

# Helper functions for encryption/decryption
def encrypt_image(image):
    key = Fernet.generate_key()
    f = Fernet(key)
    img_bytes = image.tobytes()
    encrypted = f.encrypt(img_bytes)
    return encrypted, key, image.mode, image.size

def decrypt_image(encrypted_data, key, mode, size):
    f = Fernet(key)
    decrypted_bytes = f.decrypt(encrypted_data)
    return Image.frombytes(mode, size, decrypted_bytes)

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
        logo_html = f'<img src="data:image/png;base64,{logo_data}" alt="SkinCheck Logo" style="width: 46px; height: auto; display: inline-block; vertical-align: middle;">'
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement du logo : {e}</div>', unsafe_allow_html=True)
        logo_html = ""
else:
    logo_html = ""

title_html = f'''
<div class="header-container">
    <table class="header-table">
        <tr>
            <td class="logo-cell">{logo_html}</td>
            <td class="title-cell"><div class="app-title"><span class="skin">Skin</span><span class="check">Check</span></div></td>
            <td class="empty-cell"></td>
        </tr>
    </table>
    <div class="subtitle">Should I show this mole to my dermatologist?</div>
</div>
'''

# Création des instructions de recadrage
reframed_mole_path = os.path.join("images", "FramedMole.jpg")
if os.path.exists(reframed_mole_path):
    try:
        reframed_mole_data = base64.b64encode(open(reframed_mole_path, "rb").read()).decode()
        reframed_mole_html = f'<img src="data:image/jpeg;base64,{reframed_mole_data}" style="width: 100px !important; height: auto; display: block; margin: 10px auto;">'
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement de l\'image d\'exemple : {e}</div>', unsafe_allow_html=True)
        reframed_mole_html = ""
else:
    reframed_mole_html = ""

reframe_instructions_html = f'''
<table class="instructions-table">
    <tr>
        <td>
            <div class="normal-text">Move the frame to crop the picture so that the mole takes about half the space.</div>
        </td>
        <td style="padding-left: 10px">
            {reframed_mole_html}
        </td>
    </tr>
</table>
'''

# Création de l'avertissement "prototype non validée médicalement"
warning_html = f'''
<table class="instructions-table">
    <tr>
        <td><span class="warning">⚠</span></td>
        <td><div class="warning-text">This prototype has not been validated by any medical authority. If you have any doubts, consult your dermatologist.</div></td>
    </tr>
</table>
'''

# Navigation et mode
if 'screen' not in st.session_state:
    st.session_state.screen = "Accueil"
    if 'screen_history' not in st.session_state:
        st.session_state.screen_history = []
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    if 'last_crop_box' not in st.session_state:
        st.session_state.last_crop_box = None

if st.session_state.screen == "Accueil":
    # Réinitialiser l'historique à "Accueil" pour nettoyer les résidus
    st.session_state.screen_history = ["Accueil"]
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Submit a photograph of a concerning mole* for AI to assess the need for a dermatologist consultation.</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">The image must be sharply focused and captured at close range**.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:5px;"></div>', unsafe_allow_html=True)  # Espacement réduit pour positionner plus haut
    # Solution CSS pour styliser les boutons et aligner verticalement
    st.markdown("""
    <style>
    /* Conteneur pour aligner les boutons verticalement et centrer, positionné plus haut */
    .button-container-accueil {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        max-width: 225px; /* Ajusté pour inclure le décalage de 25px */
        margin: 0 auto;
        margin-top: 0; /* Supprime la marge par défaut pour le rapprocher du haut */
    }
    /* Styliser le bouton de st.file_uploader */
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::before {
        content: "Take/select photo";
        visibility: visible;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #4A90E2;
        color: #F5F5F5;
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 0 auto;
        max-width: 200px;
    }
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover::before {
        background-color: #3A7AC2;
    }
    /* Réduire l'espace des instructions en diminuant leur taille de police */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important; /* Déjà masqué, mais ajout de font-size pour minimiser */
        font-size: 0; /* Réduit la taille de police à 0 pour minimiser l'espace */
    }
    /* Réduire la hauteur de la section pour minimiser l'espace */
    [data-testid="stFileUploaderDropzone"] {
        background-color: transparent;
        min-height: 0 !important; /* Supprime la hauteur minimale par défaut */
        height: auto; /* Suit la taille du bouton */
    }
    /* Styliser et décaler le bouton Select demo example à droite de 25px */
    .stButton {
        margin-left: 23px; /* Décalage de 25px à droite */
    }
    .stButton > button {
        background-color: #4A90E2;
        color: #F5F5F5;
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 18px !important;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 0 auto;
        max-width: 200px;
        display: block;
    }
    .stButton > button:hover {
        background-color: #3A7AC2;
    }
    /* Réduire le padding-top du block-container pour rapprocher les boutons du haut */
    .block-container {
        padding-top: 10px !important; /* Réduit de 27px à 10px */
    }
    /* Ajuster la marge du header-container pour rapprocher les boutons */
    .header-container {
        margin-top: 10px !important; /* Réduit de 20px à 10px */
    }
    </style>
    """, unsafe_allow_html=True)
    # Conteneur pour les boutons
    with st.container():
        st.markdown('<div class="button-container-accueil">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"], key="file_uploader_accueil", label_visibility="hidden")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                if not isinstance(image, Image.Image):
                    st.markdown('<div class="normal-text">Erreur : L\'image téléchargée est invalide.</div>', unsafe_allow_html=True)
                else:
                    # Tourner l'image en paysage si elle est en portrait
                    width, height = image.size
                    if height > width:
                        image = image.rotate(90, expand=True)
                    image = ImageOps.exif_transpose(image)  # Ajuster l'orientation EXIF
                    # Encrypt the image before storing
                    encrypted_data, key, mode, size = encrypt_image(image)
                    st.session_state.encrypted_original = (encrypted_data, key, mode, size)
                    st.session_state.last_image = (encrypted_data, key, mode, size)  # Sauvegarder la dernière image
                    st.session_state.screen = "Reframe"
                    st.session_state.screen_history.append("Reframe")
                    st.session_state.last_crop_box = None  # Réinitialiser le cadrage
                    st.rerun()
            except Exception as e:
                st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de l\'image : {e}</div>', unsafe_allow_html=True)
    
        if st.button("Select demo example", key="demo"):
            st.session_state.screen = "Examples"
            st.session_state.screen_history.append("Examples")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    st.markdown(warning_html, unsafe_allow_html=True)
    st.markdown('<div class="bottom-note">*For French users: a mole is a “grain de beauté”.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-note">**This requires zooming lenses (iPhone Pro 11+, Samsung Galaxy S Ultra, Google Pixel Pro…)</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Examples":
    display_back_button()
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Click one of these 6 photos to analyze it.</div>', unsafe_allow_html=True)
    base_dir = os.path.join(os.getcwd(), "examples")
    # Définir les listes d'images
    benign_images = [os.path.join(base_dir, "benignmole1.jpg"), os.path.join(base_dir, "benignmole2.jpg"), os.path.join(base_dir, "benignmole3.jpg")]
    melanoma_images = [os.path.join(base_dir, "melanoma1.jpg"), os.path.join(base_dir, "melanoma2.jpg"), os.path.join(base_dir, "melanoma3.jpg")]
 
    # Vérifier si les listes sont définies
    if not benign_images or not melanoma_images:
        st.markdown('<div class="normal-text">Erreur : Les listes d\'images de démonstration ne sont pas définies.</div>', unsafe_allow_html=True)
    else:
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
                            # Tourner l'image en paysage si elle est en portrait
                            width, height = image.size
                            if height > width:
                                image = image.rotate(90, expand=True)
                            image = ImageOps.exif_transpose(image)  # Ajuster l'orientation EXIF
                            with st.spinner("Analysis in progress..."):
                                result, prob, color = predict_user_image(image, model)
                            st.session_state.screen = "Result"
                            st.session_state.screen_history.append("Result")
                            st.session_state.result = (result, prob, color)
                            # For demos, no encryption needed
                            st.session_state.cropped_image = image  # Pas de recadrage pour les démos
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
                            # Tourner l'image en paysage si elle est en portrait
                            width, height = image.size
                            if height > width:
                                image = image.rotate(90, expand=True)
                            image = ImageOps.exif_transpose(image)  # Ajuster l'orientation EXIF
                            with st.spinner("Analysis in progress..."):
                                result, prob, color = predict_user_image(image, model)
                            st.session_state.screen = "Result"
                            st.session_state.screen_history.append("Result")
                            st.session_state.result = (result, prob, color)
                            # For demos, no encryption needed
                            st.session_state.cropped_image = image  # Pas de recadrage pour les démos
                            st.rerun()
                    except Exception as e:
                        st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de {img_path} : {e}</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Reframe":
    display_back_button()
    st.markdown(title_html, unsafe_allow_html=True)
    if 'last_image' in st.session_state and st.session_state.last_image:
        encrypted_data, key, mode, size = st.session_state.last_image
        original_image = decrypt_image(encrypted_data, key, mode, size)
        original_width, original_height = size
    elif 'encrypted_original' in st.session_state:
        encrypted_data, key, mode, size = st.session_state.encrypted_original
        original_image = decrypt_image(encrypted_data, key, mode, size)
        original_width, original_height = size
    else:
        st.markdown('<div class="normal-text">Aucune image disponible pour recadrage.</div>', unsafe_allow_html=True)

    if 'original_image' in locals():  # Vérifier si une image a été chargée
        # Tourner l'image en paysage si elle est en portrait (already handled before encryption, but re-check)
        if original_height > original_width:
            original_image = original_image.rotate(90, expand=True)
            original_width, original_height = original_image.size

        # Déterminer les nouvelles dimensions avec une hauteur maximale de 320px et une largeur maximale de 390px
        if original_width > 390 or original_height > 320:
            width_ratio = 390 / original_width
            height_ratio = 320 / original_height
            resize_ratio = min(width_ratio, height_ratio)
            new_width = int(original_width * resize_ratio)
            new_height = int(original_height * resize_ratio)
            image_resized = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            image_resized = original_image
            new_width, new_height = original_width, original_height

        # Forcer l'aspect ratio en paysage (4:3)
        aspect_ratio = (4, 3)

        st.markdown(reframe_instructions_html, unsafe_allow_html=True)
        crop_box = st_cropper(image_resized, realtime_update=True, box_color='#4A90E2', aspect_ratio=aspect_ratio, return_type="box")
        if st.button("Analyze", key="analyze"):
            if crop_box:
                scale_x = original_width / new_width
                scale_y = original_height / new_height
                left = round(crop_box['left'] * scale_x)
                top = round(crop_box['top'] * scale_y)
                width = round(crop_box['width'] * scale_x)
                height = round(crop_box['height'] * scale_y)
                if width < 224 or height < 224:
                    st.markdown('<div class="normal-text">Erreur : Le cadre de recadrage doit faire au moins 224 pixels en largeur et en hauteur. Veuillez recadrer une zone plus grande.</div>', unsafe_allow_html=True)
                else:
                    cropped_image = original_image.crop((left, top, left + width, top + height))
                    if isinstance(cropped_image, Image.Image):
                        with st.spinner("Analysis in progress..."):
                            result, prob, color = predict_user_image(cropped_image, model)
                        # Encrypt cropped image for storage
                        encrypted_cropped, cropped_key, cropped_mode, cropped_size = encrypt_image(cropped_image)
                        st.session_state.encrypted_cropped = (encrypted_cropped, cropped_key, cropped_mode, cropped_size)
                        st.session_state.screen = "Result"
                        st.session_state.screen_history.append("Result")
                        st.session_state.result = (result, prob, color)
                        st.session_state.last_crop_box = crop_box  # Sauvegarder le dernier cadrage
                        # Delete original after cropping
                        st.session_state.pop('encrypted_original', None)
                        st.rerun()
                    else:
                        st.markdown('<div class="normal-text">Erreur : L\'image recadrée n\'est pas valide. Veuillez valider le recadrage.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="normal-text">Erreur : Veuillez sélectionner une zone de recadrage.</div>', unsafe_allow_html=True)

elif st.session_state.screen == "Result":
    display_back_button()
    st.markdown(title_html, unsafe_allow_html=True)
    if 'result' in st.session_state:
        result, prob, color = st.session_state.result
        if 'encrypted_cropped' in st.session_state:
            encrypted_cropped, cropped_key, cropped_mode, cropped_size = st.session_state.encrypted_cropped
            cropped_image = decrypt_image(encrypted_cropped, cropped_key, cropped_mode, cropped_size)
        elif 'cropped_image' in st.session_state:  # For demos
            cropped_image = st.session_state.cropped_image
        else:
            cropped_image = None
        if cropped_image:
            # Tourner l'image en paysage si elle est en portrait avant affichage
            width, height = cropped_image.size
            if height > width:
                cropped_image = cropped_image.rotate(90, expand=True)
            st.image(cropped_image, caption="", use_container_width=True)
        # Présenter les résultats dans une boîte stylisée
        if result == "probably benign mole":
            st.markdown(f'<div class="result-box benin">Result: {result}</div>', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">This is probably a benign mole.</div>', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">Yet, if it is Asymmetrical, has an irregular Border, several Colors, a Diameter >6mm and/or has Evolved recently (ABCDE criteria), show it to a dermatologist.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box warning">Result: {result}</div>', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">This could be a melanoma, meaning a cluster of cancerous cells.</div>', unsafe_allow_html=True)
            st.markdown('<div class="normal-text">Melanomas are highly treatable if detected early. Show it to a dermatologist in 2-4 weeks.</div>', unsafe_allow_html=True)
        st.markdown(warning_html, unsafe_allow_html=True)
        st.markdown('<div class="normal-text">New analysis:</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
  
        # Solution CSS pour styliser les boutons et aligner verticalement
        st.markdown("""
        <style>
        /* Conteneur pour aligner les boutons verticalement et centrer, positionné plus haut */
        .button-container-result {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            max-width: 225px; /* Ajusté pour inclure le décalage de 25px */
            margin: 0 auto;
            margin-top: 0; /* Supprime la marge par défaut pour le rapprocher du haut */
        }
        /* Styliser le bouton de st.file_uploader sans modifier le texte principal */
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::before {
            content: "Take/select photo";
            visibility: visible;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #4A90E2;
            color: #F5F5F5;
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 auto;
            max-width: 200px;
        }
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover::before {
            background-color: #3A7AC2;
        }
        /* Réduire l'espace des instructions en diminuant leur taille de police */
        [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important; /* Déjà masqué */
            font-size: 0; /* Réduit la taille de police à 0 pour minimiser l'espace */
        }
        /* Réduire la hauteur de la section pour minimiser l'espace */
        [data-testid="stFileUploaderDropzone"] {
            background-color: transparent;
            min-height: 0 !important; /* Supprime la hauteur minimale par défaut */
            height: auto; /* Suit la taille du bouton */
        }
        /* Styliser et décaler le bouton Select demo example à droite de 25px */
        .stButton {
            margin-left: 23px; /* Décalage de 25px à droite */
        }
        .stButton > button {
            background-color: #4A90E2;
            color: #F5F5F5;
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
            font-size: 18px !important;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 auto;
            max-width: 200px;
            display: block;
        }
        .stButton > button:hover {
            background-color: #3A7AC2;
        }
        /* Réduire le padding-top du block-container pour rapprocher les boutons du haut */
        .block-container {
            padding-top: 10px !important; /* Réduit de 27px à 10px */
        }
        /* Ajuster la marge du header-container pour rapprocher les boutons */
        .header-container {
            margin-top: 10px !important; /* Réduit de 20px à 10px */
        }
        </style>
        """, unsafe_allow_html=True)
  
        # Conteneur pour les boutons
        with st.container():
            st.markdown('<div class="button-container-result">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"], key="file_uploader_result", label_visibility="hidden")
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    if not isinstance(image, Image.Image):
                        st.markdown('<div class="normal-text">Erreur : L\'image téléchargée est invalide.</div>', unsafe_allow_html=True)
                    else:
                        # Tourner l'image en paysage si elle est en portrait
                        width, height = image.size
                        if height > width:
                            image = image.rotate(90, expand=True)
                        image = ImageOps.exif_transpose(image)  # Ajuster l'orientation EXIF
                        # Encrypt before storing
                        encrypted_data, key, mode, size = encrypt_image(image)
                        st.session_state.encrypted_original = (encrypted_data, key, mode, size)
                        st.session_state.last_image = (encrypted_data, key, mode, size)  # Sauvegarder la dernière image
                        st.session_state.screen = "Reframe"
                        st.session_state.screen_history.append("Reframe")
                        st.session_state.last_crop_box = None  # Réinitialiser le cadrage
                        st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="normal-text">Erreur lors de l\'ouverture de l\'image : {e}</div>', unsafe_allow_html=True)
      
            if st.button("Select demo example", key="demo"):
                st.session_state.screen = "Examples"
                st.session_state.screen_history.append("Examples")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
  
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
       
        # Delete encrypted data after display and results (for compliance)
        st.session_state.pop('encrypted_cropped', None)
        st.session_state.pop('encrypted_original', None)
        st.session_state.pop('cropped_image', None)  # For demos
    else:
        st.markdown('<div class="normal-text">Erreur : Aucune image ou résultat disponible pour l\'affichage.</div>', unsafe_allow_html=True)
