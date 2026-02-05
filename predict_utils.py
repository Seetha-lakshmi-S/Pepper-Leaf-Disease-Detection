import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import traceback

# --- 1. COMPATIBILITY BRIDGE (RECURSION SAFE) ---
import keras.models
import keras.layers
import keras.saving

# Redirect internal Keras 3 paths to Keras 2
sys.modules["keras.src"] = keras
sys.modules["keras.src.models"] = keras.models
sys.modules["keras.src.layers"] = keras.layers
sys.modules["keras.src.saving"] = keras.saving

original_layer_from_config = keras.layers.Layer.from_config

@classmethod
def patched_layer_from_config(cls, config):
    """
    Patches the Keras Layer loader to handle Keras 3 metadata and 
    redirect Conv2D configs to the correct class loader.
    """
    # A. DType Fix: Convert dictionary dtypes to strings
    if "dtype" in config and isinstance(config["dtype"], dict):
        config["dtype"] = config["dtype"].get("config", {}).get("name", "float32")
    
    # B. Clean Keras 3 exclusive metadata that Keras 2 doesn't recognize
    for key in ["sparse", "ragged", "registered_name", "module"]:
        config.pop(key, None)
    
    # C. Input Shape Fix: Map Keras 3 batch_shape to Keras 2 input_shape
    if "batch_shape" in config:
        config["input_shape"] = config.pop("batch_shape")

    # 🔥 D. RECURSION-SAFE CONV2D ROUTING
    # We only intercept if the class being called is the BASE Layer class.
    # This prevents the "RecursionError" when Conv2D calls its own from_config.
    layer_name = config.get('name', '').lower()
    class_name = config.get('class_name', '').lower()

    if cls is keras.layers.Layer and ('conv2d' in layer_name or 'conv2d' in class_name):
        # Standardize parameter naming
        if 'num_filters' in config:
            config['filters'] = config.pop('num_filters')
        elif 'filters' in config:
            # Ensure consistency if the model expects one specifically
            config['filters'] = config['filters']
            
        # Manually route to Conv2D class loader
        return keras.layers.Conv2D.from_config(config)
    
    # Use the original method for all other layers or when cls is already a subclass
    return original_layer_from_config(config)

# Apply the global patch
keras.layers.Layer.from_config = patched_layer_from_config

# --- 2. CUSTOM LAYERS ---
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
        config.update({
            'num_capsule': self.num_capsule, 
            'dim_capsule': self.dim_capsule, 
            'routings': self.routings
        })
        return config

# --- 3. PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        # Preprocessing
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        custom_map = {
            'CapsuleLayer': CapsuleLayer, 
            'squash': squash, 
            'capsule_length': capsule_length
        }
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Load and Predict Binary Model
        binary_model_path = os.path.join(base_path, "binary_classifier_dataset_model.h5")
        m_bin = keras.models.load_model(binary_model_path, custom_objects=custom_map, compile=False)
        p_bin = m_bin.predict(img_array, verbose=0)[0]
        
        is_diseased = (np.argmax(p_bin) == 0)
        conf_bin = float(p_bin[np.argmax(p_bin)])
        
        # Cleanup memory immediately
        del m_bin
        keras.backend.clear_session()

        if not is_diseased:
            return "Healthy", None, conf_bin

        # Load and Predict Severity Model
        severity_model_path = os.path.join(base_path, "severity_classifier_dataset_model.h5")
        m_sev = keras.models.load_model(severity_model_path, custom_objects=custom_map, compile=False)
        p_sev = m_sev.predict(img_array, verbose=0)[0]
        
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(p_sev)]
        conf_sev = float(p_sev[np.argmax(p_sev)])
        
        del m_sev
        keras.backend.clear_session()

        return ("Bacterial Disease", res_stage, conf_sev)

    except Exception as e:
        print("❌ Prediction Engine Failed:")
        traceback.print_exc()
        return "PredictionError", None, 0.0