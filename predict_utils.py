import os
import gc
import numpy as np
import urllib.request
from PIL import Image, ImageOps, ImageFilter

# --- RENDER & MEMORY OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'

# --- 0. AUTO-DOWNLOAD CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "binary_classifier_dataset_model.keras")
SEVERITY_MODEL_PATH = os.path.join(BASE_DIR, "severity_classifier_dataset_model.keras")

def ensure_models_exist():
    """Checks if models exist locally; if not, downloads from GitHub Release Assets."""
    # These URLs point to your manual uploads in the 'Releases' tab
    model_configs = {
        BINARY_MODEL_PATH: "https://github.com/Seetha-lakshmi-S/Pepper-Leaf-Disease-Detection/releases/download/v2.0/binary_classifier_dataset_model.keras",
        SEVERITY_MODEL_PATH: "https://github.com/Seetha-lakshmi-S/Pepper-Leaf-Disease-Detection/releases/download/v2.0/severity_classifier_dataset_model.keras"
    }

    for path, url in model_configs.items():
        # Logic: If file doesn't exist OR it's a small LFS pointer (under 100KB), download real file
        if not os.path.exists(path) or os.path.getsize(path) < 102400:
            print(f"Model {os.path.basename(path)} missing or LFS pointer detected. Downloading real file...")
            try:
                # Bypass SSL if needed for Render environment
                urllib.request.urlretrieve(url, path)
                print(f"Successfully downloaded {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
            except Exception as e:
                print(f"Download failed for {os.path.basename(path)}: {e}")

def run_prediction(image_path):
    """Keras 3 Compatible Prediction Logic with Release-Asset fallback."""
    import tensorflow as tf
    import keras
    from keras import layers

    # 1. First, make sure the files are actually there
    ensure_models_exist()

    # --- MATCH TRAINING SCRIPT CUSTOM COMPONENTS ---
    @keras.utils.register_keras_serializable(package="Project")
    def squash(vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
        return scale * vectors

    @keras.utils.register_keras_serializable(package="Project")
    def get_capsule_length(vectors):
        return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

    @keras.utils.register_keras_serializable(package="Project")
    class CapsuleLayer(layers.Layer):
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

    try:
        # --- 2. IMAGE PREPROCESSING ---
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = img.resize((128, 128))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        custom_map = {
            "Project>CapsuleLayer": CapsuleLayer,
            "Project>squash": squash,
            "Project>get_capsule_length": get_capsule_length,
            "margin_loss": lambda y_true, y_pred: y_pred
        }

        # --- 3. STAGE 1: BINARY ---
        m_bin = keras.models.load_model(BINARY_MODEL_PATH, 
                                        custom_objects=custom_map, compile=False)
        preds_bin = m_bin.predict(img_array, verbose=0)[0]
        bin_idx = np.argmax(preds_bin)
        bin_conf = float(preds_bin[bin_idx])

        del m_bin
        keras.backend.clear_session()
        gc.collect()

        if bin_idx == 1:
            return "Healthy", "Healthy", bin_conf

        # --- 4. STAGE 2: SEVERITY ---
        m_sev = keras.models.load_model(SEVERITY_MODEL_PATH, 
                                        custom_objects=custom_map, compile=False)
        preds_sev = m_sev.predict(img_array, verbose=0)[0]
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        sev_idx = np.argmax(preds_sev)
        
        res_stage = stages[sev_idx]
        res_conf = float(preds_sev[sev_idx])

        del m_sev
        keras.backend.clear_session()
        gc.collect()

        return "Bacterial Disease", res_stage, res_conf

    except Exception as e:
        gc.collect()
        return "Error", str(e)[:100], 0.0