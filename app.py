# Si jamais ce code devait être modifié, le faire sans la bibliothèque cv2, que Streamlit a régulièrement du mal à importer
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import keras.backend as K

# Fonction pour charger dynamiquement les images depuis le répertoire "examples" et tous ses sous-répertoires
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

# Fonction focal_loss_fixed
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

# Classe MelanomaRecall avec from_config amélioré
class MelanomaRecall(tf.keras.metrics.Metric):
    def __init__(self, melanoma_index, name='melanoma_recall', **kwargs):
        super(MelanomaRecall, self).__init__(name=name, **kwargs)
        self.melanoma_index = melanoma_index
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
        true_melanoma = tf.equal(y_true, self.melanoma_index)
        pred_melanoma = tf.equal(y_pred, self.melanoma_index)
        true_pos = tf.reduce_sum(tf.cast(true_melanoma & pred_melanoma, tf.float32))
        possible_pos = tf.reduce_sum(tf.cast(true_melanoma, tf.float32))
        self.true_positives.assign_add(true_pos)
        self.possible_positives.assign_add(possible_pos)

    def result(self):
        return self.true_positives / (self.possible_positives + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0.)
        self.possible_positives.assign(0.)

    @classmethod
    def from_config(cls, config):
        melanoma_index = config.get('melanoma_index', 0)
        filtered_config = {k: v for k, v in config.items() if k not in ['melanoma_index']}
        return cls(melanoma_index=melanoma_index, **filtered_config)

# Classe NevusSpecificity avec from_config amélioré
class NevusSpecificity(tf.keras.metrics.Metric):
    def __init__(self, nevus_index, name='nevus_specificity', **kwargs):
        super(NevusSpecificity, self).__init__(name=name, **kwargs)
        self.nevus_index = nevus_index
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.possible_negatives = self.add_weight(name='pn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
        true_nevus = tf.equal(y_true, self.nevus_index)
        pred_nevus = tf.equal(y_pred, self.nevus_index)
        true_neg = tf.reduce_sum(tf.cast(true_nevus & pred_nevus, tf.float32))
        possible_neg = tf.reduce_sum(tf.cast(true_nevus, tf.float32))
        self.true_negatives.assign_add(true_neg)
        self.possible_negatives.assign_add(possible_neg)

    def result(self):
        return self.true_negatives / (self.possible_negatives + K.epsilon())

    def reset_states(self):
        self.true_negatives.assign(0.)
        self.possible_negatives.assign(0.)

    @classmethod
    def from_config(cls, config):
        nevus_index = config.get('nevus_index', 1)
        filtered_config = {k: v for k, v in config.items() if k not in ['nevus_index']}
        return cls(nevus_index=nevus_index, **filtered_config)

# Classe CombinedMetric avec from_config corrigé
class CombinedMetric(tf.keras.metrics.Metric):
    def __init__(self, melanoma_recall, nevus_specificity, name='combined_metric', alpha=0.55, **kwargs):
        super(CombinedMetric, self).__init__(name=name, **kwargs)
        self.melanoma_recall = melanoma_recall
        self.nevus_specificity = nevus_specificity
        self.alpha = alpha
        self.combined_value = self.add_weight(name='combined_value', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.melanoma_recall.update_state(y_true, y_pred, sample_weight)
        self.nevus_specificity.update_state(y_true, y_pred, sample_weight)
        recall_value = self.melanoma_recall.result()
        specificity_value = self.nevus_specificity.result()
        combined = self.alpha * recall_value + (1 - self.alpha) * specificity_value
        self.combined_value.assign(combined)

    def result(self):
        return self.combined_value

    def reset_states(self):
        self.melanoma_recall.reset_states()
        self.nevus_specificity.reset_states()
        self.combined_value.assign(0.)

    @classmethod
    def from_config(cls, config):
        melanoma_recall = MelanomaRecall.from_config({'melanoma_index': 0, 'name': 'melanoma_recall'})
        nevus_specificity = NevusSpecificity.from_config({'nevus_index': 1, 'name': 'nevus_specificity'})
        filtered_config = {k: v for k, v in config.items() if k not in ['melanoma_recall_config', 'nevus_specificity_config']}
        return cls(melanoma_recall=melanoma_recall, nevus_specificity=nevus_specificity, **filtered_config)

# Classe ThresholdOptimizer (inclus pour compatibilité)
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, class_to_idx, target_recall=0.85, target_specificity=0.70):
        super(ThresholdOptimizer, self).__init__()
        self.val_data = validation_data
        self.class_to_idx = class_to_idx
        self.target_recall = target_recall
        self.target_specificity = target_specificity
        self.best_threshold = 0.5
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Charger le modèle avec tous les objets personnalisés
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

# Fonction de prédiction avec seuil fixe
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

st.markdown("<h1 style='text-align: center; color: #00aaff;'>SkinCheck</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Should I show this mole to my dermatologist?</h3>", unsafe_allow_html=True)

# Avertissement
st.markdown("<div style='background-color: #ff4500; color: white; padding: 10px; border-radius: 5px; text-align: center;'>"
            "<strong>Warning:</strong> This app is a prototype and has not been validated by any medical authority. Its results should not be trusted. If you have any doubts, consult your dermatologist.</div>", 
            unsafe_allow_html=True)

# Navigation et mode
if 'screen' not in st.session_state:
    st.session_state.screen = "Accueil"

if st.button("←", key="back"):
    st.session_state.screen = "Accueil"

if st.session_state.screen == "Accueil":
    st.write("### Take a photo of your mole*. An artificial intelligence will try to determine if you should show it to a dermatologist.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Take a photo", key="take_photo"):
            st.session_state.screen = "Photo"
        if st.button("Browse phone photos", key="browse"):
            st.session_state.screen = "Browse"
        if st.button("Select demo example", key="demo"):
            st.session_state.screen = "Examples"

elif st.session_state.screen == "Photo":
    captured_file = st.camera_input("Take a sharp photo as close as possible")
    if captured_file is not None:
        image = Image.open(captured_file)
        image = ImageOps.exif_transpose(image)
        st.session_state.image = image
        st.session_state.screen = "Reframe"

elif st.session_state.screen == "Browse":
    uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        st.session_state.image = image
        st.session_state.screen = "Reframe"

elif st.session_state.screen == "Examples":
    st.write("### Choose one of the following examples:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Benign moles:**")
        for label, path in exemples_complets.get("benign", []):
            if st.button(label, key=f"benign_{label}"):
                st.session_state.image = Image.open(path)
                st.session_state.screen = "Reframe"
    with col2:
        st.write("**Melanomas:**")
        for label, path in exemples_complets.get("melanoma", []):
            if st.button(label, key=f"melanoma_{label}"):
                st.session_state.image = Image.open(path)
                st.session_state.screen = "Reframe"

elif st.session_state.screen == "Reframe":
    if 'image' in st.session_state:
        image = st.session_state.image
        st.image(image, caption="Frame the picture so that the mole takes half the space", use_column_width=True)
        st.write(f"Current size: {image.size[0]} x {image.size[1]}")
        if st.button("Reframe", key="reframe"):
            # Simuler un ajustement (limitation sans cv2)
            st.warning("Veuillez recadrer manuellement l'image pour que la lésion occupe environ la moitié de l'espace.")
        if st.button("Analyze", key="analyze"):
            with st.spinner("Analysis in progress..."):
                result, prob, color = predict_user_image(image)
            st.session_state.screen = "Result"
            st.session_state.result = (result, prob, color)

elif st.session_state.screen == "Result":
    if 'result' in st.session_state:
        result, prob, color = st.session_state.result
        st.image(st.session_state.image, caption="Analysis result:", use_column_width=True)
        st.write(f"This should be a {result} mole. Yet, if it is asymmetrical, has an irregular border, several colors, a diameter >6mm and/or has evolved recently, show it to a dermatologist.")
        st.markdown(f"<div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;'>{result}</div>", unsafe_allow_html=True)
        st.write("New analysis:")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Take a photo"):
                st.session_state.screen = "Photo"
            if st.button("Browse phone photos"):
                st.session_state.screen = "Browse"
            if st.button("Select demo example"):
                st.session_state.screen = "Examples"

# Instructions
st.markdown("### Instructions")
st.write("""
- **Accueil**: Take a photo, browse phone photos, or select a demo example.
- **Reframe**: Frame the mole to take half the space.
- **Result**: View the analysis and decide next steps.
""")

# Avertissement final
st.markdown("---")
st.error("**Avertissement : Cet outil est un prototype. Consultez toujours un dermatologue pour un diagnostic officiel.**")
