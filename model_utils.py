import tensorflow as tf
import keras.backend as K
import streamlit as st
import numpy as np
from PIL import Image  # Ajout de l'importation explicite

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

# Classe MelanomaRecall
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

# Classe NevusSpecificity
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

# Classe CombinedMetric
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

# Classe ThresholdOptimizer
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

# Charger le modèle avec chemin vérifié
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "skin_lesion_model_binary.keras")
    st.markdown(f'<div class="normal-text">Tentative de chargement du modèle depuis : {model_path}</div>', unsafe_allow_html=True)
    try:
        custom_objects = {
            'focal_loss_fixed': focal_loss_fixed(gamma=1.0, alpha=0.9),
            'MelanomaRecall': MelanomaRecall,
            'NevusSpecificity': NevusSpecificity,
            'CombinedMetric': CombinedMetric,
            'ThresholdOptimizer': ThresholdOptimizer
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.markdown('<div class="normal-text">Modèle chargé avec succès.</div>', unsafe_allow_html=True)
        return model
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur lors du chargement du modèle : {e}</div>', unsafe_allow_html=True)
        return None

# Fonction de prétraitement
def preprocess_image(image, target_size=(224, 224)):
    try:
        st.markdown('<div class="normal-text">Débogage : Conversion de l\'image en RGB...</div>', unsafe_allow_html=True)
        img = image.convert('RGB')
        st.markdown('<div class="normal-text">Débogage : Redimensionnement de l\'image...</div>', unsafe_allow_html=True)
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        st.markdown('<div class="normal-text">Débogage : Conversion en tableau numpy...</div>', unsafe_allow_html=True)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        st.markdown(f'<div class="normal-text">Erreur de prétraitement : {e}</div>', unsafe_allow_html=True)
        return None

# Fonction de prédiction
def predict_user_image(image, model):
    if model is None:
        st.markdown('<div class="normal-text">Erreur : Le modèle n\'a pas été chargé correctement.</div>', unsafe_allow_html=True)
        return "Erreur : Impossible de traiter l'image.", None, None
    
    st.markdown('<div class="normal-text">Débogage : Préparation de l\'image pour la prédiction...</div>', unsafe_allow_html=True)
    img_array = preprocess_image(image)
    if img_array is None:
        st.markdown('<div class="normal-text">Erreur : L\'image n\'a pas pu être prétraitée. Vérifiez le format ou la validité de l\'image.</div>', unsafe_allow_html=True)
        return "Erreur : Impossible de traiter l'image.", None, None
