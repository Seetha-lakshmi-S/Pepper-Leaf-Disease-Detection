import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import traceback

# --- 1. ENVIRONMENT & MEMORY OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU to save memory on Render

# --- 2. CUSTOM CAPSULE MATH ---
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

# --- 3. THE NUCLEAR LOADER (ARCHITECTURAL REBUILD) ---
def safe_load_model(model_path, num_classes):
    """
    Manual reconstruction of the Capsule Network. 
    Matches standard CapsNet training for 128x128 images.
    """
    inputs = keras.Input(shape=(128, 128, 3))
    
    # Block 1: Feature Extraction
    x = keras.layers.Conv2D(128, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    # Block 2: Higher Level Features
    x = keras.layers.Conv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    # Block 3: Primary Capsules (The Interface)
    # Most CapsNets use a larger kernel here to define the 'parts' of the image
    x = keras.layers.Conv2D(8 * 32, kernel_size=9, strides=2, padding='valid')(x)
    x = keras.layers.Reshape(target_shape=(-1, 8))(x)
    primary_caps = keras.layers.Lambda(squash)(x)
    
    # Block 4: Digit Capsules (The Classification)
    digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3)(primary_caps)
    outputs = keras.layers.Lambda(capsule_length)(digit_caps)
    
    model = keras.Model(inputs, outputs)
    
    # Load weights BYPASSING the corrupted .h5 config dictionary
    model.load_weights(model_path)
    return model

# --- 4. PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        # Image Loading
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # --- 1. BINARY PREDICTION ---
        binary_path = os.path.join(base_path, "binary_classifier_dataset_model.h5")
        m_bin = safe_load_model(binary_path, num_classes=2)
        p_bin = m_bin.predict(img_array, verbose=0)[0]
        
        is_diseased = (np.argmax(p_bin) == 0) # Assumes Class 0 = Diseased
        conf_bin = float(p_bin[np.argmax(p_bin)])
        
        # Clear memory between models
        del m_bin
        keras.backend.clear_session()

        if not is_diseased:
            return "Healthy", None, conf_bin

        # --- 2. SEVERITY PREDICTION ---
        severity_path = os.path.join(base_path, "severity_classifier_dataset_model.h5")
        m_sev = safe_load_model(severity_path, num_classes=3)
        p_sev = m_sev.predict(img_array, verbose=0)[0]
        
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(p_sev)]
        conf_sev = float(p_sev[np.argmax(p_sev)])
        
        del m_sev
        keras.backend.clear_session()

        return ("Bacterial Disease", res_stage, conf_sev)

    except Exception as e:
        print(f"❌ CRITICAL FAILURE in Prediction Engine: {e}")
        traceback.print_exc()
        return "PredictionError", None, 0.0