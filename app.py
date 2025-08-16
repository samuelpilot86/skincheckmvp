# Si jamais ce code devait être modifié, le faire sans la bibliothèque cv2, que Streamlit a régulièrement du mal à importer

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import keras.backend as K

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

# Classe ThresholdOptimizer (utilisée comme callback, mais incluse pour compatibilité)
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
        pass  # Simplifié, car non utilisé ici

    def on_train_end(self, logs=None):
        pass  # Simplifié, car non utilisé ici

# Charger le modèle avec tous les objets personnalisés
@st.cache_resource
def load_model():
    melanoma_index = 0  # 'mel' (à ajuster si nécessaire)
    nevus_index = 1     # 'nv' (à ajuster si nécessaire)
    custom_objects = {
        'focal_loss_fixed': focal_loss_fixed(gamma=1.0, alpha=0.9),
        'MelanomaRecall': MelanomaRecall(melanoma_index),
        'NevusSpecificity': NevusSpecificity(nevus_index),
        'CombinedMetric': CombinedMetric(MelanomaRecall(melanoma_index), NevusSpecificity(nevus_index))
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

# Fonction de prédiction
def predict_user_image(image):
    img_array = preprocess_image(image)
    if img_array is None:
        return "Erreur : Impossible de traiter l'image."
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    threshold = 0.487  # Ajustez selon votre seuil optimal (par ex. basé sur threshold_optimizer.best_threshold)
    probability = prediction[0][0] 