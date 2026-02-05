import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import traceback

# --- COMPATIBILITY BRIDGE (ENHANCED) ---
import keras.models
import keras.layers
import keras.saving

sys.modules["keras.src"] = keras
sys.modules["keras.src.models"] = keras.models
sys.modules["keras.src.layers"] = keras.layers
sys.modules["keras.src.saving"] = keras.saving

# 🔥 FIXED LAYER PATCHER - CATCHES ALL Conv2D
original_layer_from_config = keras.layers.Layer.from_config

@classmethod
def patched_layer_from_config(cls, config):
    # 1. DType fix
    if "dtype" in config and isinstance(config["dtype"], dict):
        config["dtype"] = config["dtype"].get("config", {}).get("name", "float32")
    
    # 🔥 2. CONV2D FIX - Check BOTH class_name AND layer name
    layer_name = config.get('name', '').lower()
    class_name = config.get('class_name', '').lower()
    
    # Fix Conv2D regardless of class_name presence
    if ('conv2d' in layer_name or 'conv2d' in class_name) and 'filters' in config:
        print(f"🔧 Fixing Conv2D: {config.get('name')} filters={config['filters']}")
        config['num_filters'] = config.pop('filters')
    
    # 3. Remove unwanted keys
    unwanted_keys = ["sparse", "ragged", "registered_name", "module"]
    for key in unwanted_keys:
        config.pop(key, None)
    
    # 4. Batch shape fix
    if "batch_shape" in config:
        config["input_shape"] = config.pop("batch_shape")
    
    return original_layer_from_config(config)

keras.layers.Layer.from_config = patched_layer_from_config

# --- ENVIRONMENT ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- CUSTOM FUNCTIONS ---
@keras.utils.register_keras_serializable(package="Custom")
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

@keras.utils.register_keras_serializable(package="Custom")
def capsule_length(vectors):
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

# --- CAPSULE LAYER ---
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

# --- PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        print(f"🔍 Loading image: {image_path}")
        
        # Image preprocessing
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        custom_map = {
            'CapsuleLayer': CapsuleLayer, 
            'squash': squash, 
            'capsule_length': capsule_length
        }
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        binary_path = os.path.join(base_path, "binary_classifier_dataset_model.h5")
        severity_path = os.path.join(base_path, "severity_classifier_dataset_model.h5")
        
        print(f"📂 Model paths - Binary: {binary_path}, Severity: {severity_path}")

        # BINARY MODEL
        print("🤖 Loading binary model...")
        m_binary = keras.models.load_model(binary_path, custom_objects=custom_map, compile=False)
        preds_bin = m_binary.predict(img_array, verbose=0)[0]
        idx_bin = np.argmax(preds_bin)
        is_diseased = (idx_bin == 0) 
        conf_bin = float(preds_bin[idx_bin])
        
        del m_binary
        keras.backend.clear_session()
        print(f"✅ Binary: {'Diseased' if is_diseased else 'Healthy'} ({conf_bin:.2%})")

        if not is_diseased:
            return "Healthy", None, conf_bin

        # SEVERITY MODEL
        print("🤖 Loading severity model...")
        m_severity = keras.models.load_model(severity_path, custom_objects=custom_map, compile=False)
        preds_sev = m_severity.predict(img_array, verbose=0)[0]
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(preds_sev)]
        conf_sev = float(preds_sev[np.argmax(preds_sev)])

        del m_severity
        keras.backend.clear_session()
        print(f"✅ Severity: {res_stage} ({conf_sev:.2%})")

        return ("Bacterial Disease", res_stage, conf_sev)

    except Exception as e:
        print(f"❌ Prediction Engine Failed: {e}")
        traceback.print_exc()
        return "PredictionError", None, 0.0
