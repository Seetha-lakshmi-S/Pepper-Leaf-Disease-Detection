import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import traceback

# --- 1. COMPATIBILITY BRIDGE (KERAS 3 TO 2) ---
import keras.models
import keras.layers
import keras.saving

# Redirect Keras 3 internal paths to Keras 2 locations
sys.modules["keras.src"] = keras
sys.modules["keras.src.models"] = keras.models
sys.modules["keras.src.layers"] = keras.layers
sys.modules["keras.src.saving"] = keras.saving

# 🔥 THE PATCHER - Fixes Conv2D and cleans Keras 3 metadata
original_layer_from_config = keras.layers.Layer.from_config

@classmethod
def patched_layer_from_config(cls, config):
    # 1. DType Fix (converts dict to string)
    if "dtype" in config and isinstance(config["dtype"], dict):
        config["dtype"] = config["dtype"].get("config", {}).get("name", "float32")
    
    # 2. CONV2D FIX - Keras 3 saves 'filters', but the shim expects 'num_filters'
    layer_name = config.get('name', '').lower()
    if 'conv2d' in layer_name and 'filters' in config:
        config['num_filters'] = config.pop('filters')
    
    # 3. Clean Keras 3 exclusive metadata
    for key in ["sparse", "ragged", "registered_name", "module"]:
        config.pop(key, None)
    
    # 4. Input Shape Fix
    if "batch_shape" in config:
        config["input_shape"] = config.pop("batch_shape")
    
    return original_layer_from_config(config)

keras.layers.Layer.from_config = patched_layer_from_config

# --- 2. ENVIRONMENT & CUSTOM LAYERS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@keras.utils.register_keras_serializable(package="Custom")
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

@keras.utils.register_keras_serializable(package="Custom")
def capsule_length(vectors):
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

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
            initializer='glorot_uniform', name='W')

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
        config.update({'num_capsule': self.num_capsule, 'dim_capsule': self.dim_capsule, 'routings': self.routings})
        return config

# --- 3. PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        custom_map = {'CapsuleLayer': CapsuleLayer, 'squash': squash, 'capsule_length': capsule_length}
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Load Binary Model
        m_bin = keras.models.load_model(os.path.join(base_path, "binary_classifier_dataset_model.h5"), 
                                        custom_objects=custom_map, compile=False)
        p_bin = m_bin.predict(img_array, verbose=0)[0]
        is_diseased = (np.argmax(p_bin) == 0)
        conf_bin = float(p_bin[np.argmax(p_bin)])
        del m_bin; keras.backend.clear_session()

        if not is_diseased:
            return "Healthy", None, conf_bin

        # Load Severity Model
        m_sev = keras.models.load_model(os.path.join(base_path, "severity_classifier_dataset_model.h5"), 
                                        custom_objects=custom_map, compile=False)
        p_sev = m_sev.predict(img_array, verbose=0)[0]
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(p_sev)]
        conf_sev = float(p_sev[np.argmax(p_sev)])
        del m_sev; keras.backend.clear_session()

        return ("Bacterial Disease", res_stage, conf_sev)
    except Exception as e:
        traceback.print_exc()
        return "PredictionError", None, 0.0