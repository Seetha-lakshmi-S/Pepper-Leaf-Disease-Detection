import os
import gc
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
import keras

# --- 1. RENDER & MEMORY OPTIMIZATION ---
# Reduce TensorFlow logging and force CPU usage to stay within RAM limits
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'

# --- 2. CUSTOM CAPSULE COMPONENTS ---
# These must match your training logic exactly to decode the .keras files
@keras.utils.register_keras_serializable(package="Project")
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

@keras.utils.register_keras_serializable(package="Project")
def get_capsule_length(vectors):
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

@keras.utils.register_keras_serializable(package="Project")
class CapsuleLayer(keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
            initializer="glorot_uniform", name="W"
        )

    def call(self, inputs):
        inputs_hat = tf.einsum("bji,jkio->bkjo", inputs, self.W)
        b = tf.zeros([tf.shape(inputs)[0], self.num_capsule, self.input_num_capsule])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s_j = tf.einsum("bki,bkio->bko", c, inputs_hat)
            v_j = squash(s_j)
            if i < self.routings - 1:
                b += tf.einsum("bko,bkio->bki", v_j, inputs_hat)
        return v_j

    def get_config(self):
        config = super().get_config()
        config.update({"num_capsule": self.num_capsule, "dim_capsule": self.dim_capsule, "routings": self.routings})
        return config

# --- 3. LAZY MODEL LOADING (RAM Efficiency) ---
custom_map = {
    "Project>CapsuleLayer": CapsuleLayer,
    "Project>squash": squash,
    "Project>get_capsule_length": get_capsule_length,
    "margin_loss": lambda y_true, y_pred: y_pred
}

MODEL_BIN = None
MODEL_SEV = None

def get_binary_model():
    global MODEL_BIN
    if MODEL_BIN is None:
        print("Initializing Binary Model...")
        MODEL_BIN = keras.models.load_model("binary_classifier_dataset_model.keras", 
                                          custom_objects=custom_map, compile=False)
    return MODEL_BIN

def get_severity_model():
    global MODEL_SEV
    if MODEL_SEV is None:
        print("Initializing Severity Model...")
        MODEL_SEV = keras.models.load_model("severity_classifier_dataset_model.keras", 
                                          custom_objects=custom_map, compile=False)
    return MODEL_SEV

# --- 4. PREDICTION LOGIC ---
def run_prediction(image_path):
    """
    Main entry point for prediction. 
    Uses Stage 1 (Binary) then Stage 2 (Severity) if diseased.
    """
    try:
        # 1. IMAGE PREPROCESSING
        img = Image.open(image_path).convert('RGB')
        
        # Resize immediately to 128x128 to save RAM
        img = img.resize((128, 128))
        
        # Match training filters exactly
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        # 2. STAGE 1: BINARY (Healthy vs Diseased)
        m_bin = get_binary_model()
        preds_bin = m_bin.predict(img_array, verbose=0)[0]
        bin_idx = np.argmax(preds_bin)
        bin_conf = float(preds_bin[bin_idx])

        # Logic check: Assuming index 1 is 'Healthy'
        if bin_idx == 1:
            return "Healthy", "Healthy", bin_conf

        # 3. STAGE 2: SEVERITY (Early, Mid, Advanced)
        m_sev = get_severity_model()
        preds_sev = m_sev.predict(img_array, verbose=0)[0]
        
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        sev_idx = np.argmax(preds_sev)
        
        res_stage = stages[sev_idx]
        res_conf = float(preds_sev[sev_idx])

        return "Bacterial Disease", res_stage, res_conf

    except Exception as e:
        print(f"Prediction logic error: {e}")
        return "Error", str(e)[:100], 0.0
    
    finally:
        # 4. FORCE MEMORY CLEANUP
        if 'img' in locals(): del img
        if 'img_array' in locals(): del img_array
        gc.collect()