import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import traceback

# --- 0. THE COMPATIBILITY BRIDGE (FIXES KERAS 3 -> 2 ERRORS) ---
import keras.models
import keras.layers
import keras.saving

# A. Redirect missing Keras 3 paths ('keras.src') to Keras 2 locations
sys.modules["keras.src"] = keras
sys.modules["keras.src.models"] = keras.models
sys.modules["keras.src.models.functional"] = keras.models
sys.modules["keras.src.layers"] = keras.layers
sys.modules["keras.src.saving"] = keras.saving

# B. THE UNIVERSAL LAYER PATCHER
# This intercepts configurations for ALL layers to fix DTypePolicy and batch_shape
original_layer_from_config = keras.layers.Layer.from_config

@classmethod
def patched_layer_from_config(cls, config):
    # Fix 1: Convert complex DTypePolicy back to a simple string (fixes Conv2D error)
    if "dtype" in config and isinstance(config["dtype"], dict):
        # Extract 'float32' from the Keras 3 DTypePolicy dictionary
        if "config" in config["dtype"] and "name" in config["dtype"]["config"]:
            config["dtype"] = config["dtype"]["config"]["name"]
        else:
            config["dtype"] = "float32"

    # Fix 2: Rename 'batch_shape' to 'input_shape' (fixes InputLayer error)
    if "batch_shape" in config:
        config["input_shape"] = config.pop("batch_shape")
        
    return original_layer_from_config(config)

# Apply the patch to the base Layer class
keras.layers.Layer.from_config = patched_layer_from_config

# --- 1. ENVIRONMENT CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 2. GLOBAL CUSTOM FUNCTIONS ---
@keras.utils.register_keras_serializable(package="Custom")
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

@keras.utils.register_keras_serializable(package="Custom")
def capsule_length(vectors):
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

# --- 3. CUSTOM CAPSULE LAYER ---
@keras.utils.register_keras_serializable(package="Custom")
class CapsuleLayer(keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )

    def call(self, inputs):
        inputs_hat = tf.einsum('bji,jkio->bkjo', inputs, self.W)
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsule, self.input_num_capsule])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s_j = tf.einsum('bkl,bkld->bkd', c, inputs_hat)
            v_j = squash(s_j)
            if i < self.routings - 1:
                agreement = tf.einsum('bkd,bkld->bkl', v_j, inputs_hat)
                b += agreement
        return v_j

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        })
        return config

# --- 4. PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        # 1. Image Preprocessing
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        # 2. Setup paths and Custom Objects
        custom_map = {
            'CapsuleLayer': CapsuleLayer, 
            'squash': squash, 
            'capsule_length': capsule_length
        }
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        binary_path = os.path.join(base_path, "binary_classifier_dataset_model.h5")
        severity_path = os.path.join(base_path, "severity_classifier_dataset_model.h5")

        # 3. Load Binary Model
        m_binary = keras.models.load_model(binary_path, custom_objects=custom_map, compile=False)
        preds_bin = m_binary.predict(img_array, verbose=0)[0]
        idx_bin = np.argmax(preds_bin)
        is_diseased = (idx_bin == 0) 
        conf_bin = float(preds_bin[idx_bin])
        
        del m_binary
        keras.backend.clear_session()

        if not is_diseased:
            return "Healthy", None, conf_bin

        # 4. Load Severity Model
        m_severity = keras.models.load_model(severity_path, custom_objects=custom_map, compile=False)
        preds_sev = m_severity.predict(img_array, verbose=0)[0]
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(preds_sev)]
        conf_sev = float(preds_sev[np.argmax(preds_sev)])

        del m_severity
        keras.backend.clear_session()

        return ("Bacterial Disease", res_stage, conf_sev)

    except Exception as e:
        print(f"❌ Prediction Engine Failed: {e}")
        traceback.print_exc()
        return "PredictionError", None, 0.0