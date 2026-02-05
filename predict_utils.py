import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import traceback
import gc

# --- 1. MEMORY & ENVIRONMENT OPTIMIZATION ---
# Force CPU and disable heavy logging to save memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Limit TensorFlow's memory footprint by restricting threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --- 2. CUSTOM CAPSULE LAYERS ---
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

# --- 3. ARCHITECTURE BUILDER (BYPASSES CONFIG ERRORS) ---
def build_capsule_net(num_classes):
    """
    Manually reconstructs the architecture. 
    Ensure these parameters match your training script exactly.
    """
    inputs = keras.Input(shape=(128, 128, 3))
    
    # Feature Extraction
    x = keras.layers.Conv2D(128, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x = keras.layers.Conv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    # Primary Capsules (Kernel 9, Stride 2 is standard for CapsNet)
    x = keras.layers.Conv2D(8 * 32, kernel_size=9, strides=2, padding='valid')(x)
    x = keras.layers.Reshape(target_shape=(-1, 8))(x)
    primary_caps = keras.layers.Lambda(squash)(x)
    
    # Digit Capsules
    digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3)(primary_caps)
    outputs = keras.layers.Lambda(capsule_length)(digit_caps)
    
    return keras.Model(inputs, outputs)

# --- 4. PREDICTION ENGINE ---
def run_prediction(image_path):
    try:
        # Preprocess Image
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), 0)

        base_path = os.path.dirname(os.path.abspath(__file__))

        # --- PHASE 1: BINARY CLASSIFICATION ---
        model = build_capsule_net(num_classes=2)
        model.load_weights(os.path.join(base_path, "binary_classifier_dataset_model.h5"))
        
        preds = model.predict(img_array, verbose=0)[0]
        # Class 0=Diseased, 1=Healthy (Verify this index with your training!)
        is_diseased = (np.argmax(preds) == 0)
        confidence = float(np.max(preds))

        # 🔥 AGGRESSIVE MEMORY CLEANUP
        del model
        keras.backend.clear_session()
        gc.collect()

        if not is_diseased:
            return "Healthy", None, confidence

        # --- PHASE 2: SEVERITY CLASSIFICATION ---
        model = build_capsule_net(num_classes=3)
        model.load_weights(os.path.join(base_path, "severity_classifier_dataset_model.h5"))
        
        preds = model.predict(img_array, verbose=0)[0]
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        res_stage = stages[np.argmax(preds)]
        res_conf = float(np.max(preds))

        # FINAL CLEANUP
        del model
        keras.backend.clear_session()
        gc.collect()

        return ("Bacterial Disease", res_stage, res_conf)

    except Exception as e:
        print(f"❌ Prediction Engine Error: {e}")
        traceback.print_exc()
        # Ensure session is cleared even on fail
        keras.backend.clear_session()
        gc.collect()
        return "PredictionError", None, 0.0