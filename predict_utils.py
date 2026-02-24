import os
import gc
import numpy as np
import urllib.request
import cv2
from PIL import Image
from skimage.segmentation import chan_vese
from skimage import img_as_float

# --- RENDER & MEMORY OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "binary_classifier_dataset_model.keras")
SEVERITY_MODEL_PATH = os.path.join(BASE_DIR, "severity_classifier_dataset_model.keras")

def project_pipeline_segmentation(image_path):
    """
    Implements your Final Year Project logic:
    Module 1: Median Filtering
    Module 2: Chan-Vese Active Contour Segmentation
    """
    # 1. Load and Denoise (Module 1)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Standardize size for processing
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Median Blur (ksize=5 as per your script)
    denoised = cv2.medianBlur(img_rgb, 5)

    # 2. Prepare for Chan-Vese (Module 2)
    lab_image = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    hsv_image = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    segmentation_channel = lab_image[:, :, 1] # 'a*' channel for high contrast

    # Stage 1: Initial Mask (Green + Disease)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    lower_disease = np.array([10, 50, 20])
    upper_disease = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv_image, lower_disease, upper_disease)
    
    combined_mask = cv2.bitwise_or(green_mask, disease_mask)

    # Stage 2: Morphological Closing to fill gaps
    kernel = np.ones((7,7), np.uint8)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Stage 3: Chan-Vese ACS
    # Reduced iterations slightly (100) for faster web response
    gray_float = img_as_float(segmentation_channel)
    cv_mask_bool = chan_vese(
        gray_float, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
        max_num_iter=100, dt=0.5, init_level_set=closed_mask
    )

    # Stage 4: Final Masking
    final_mask = (cv_mask_bool.astype(np.uint8)) * 255
    segmented_output = cv2.bitwise_and(denoised, denoised, mask=final_mask)
    
    return segmented_output

def ensure_models_exist():
    model_configs = {
        BINARY_MODEL_PATH: "https://github.com/Seetha-lakshmi-S/Pepper-Leaf-Disease-Detection/releases/download/v2.0/binary_classifier_dataset_model.keras",
        SEVERITY_MODEL_PATH: "https://github.com/Seetha-lakshmi-S/Pepper-Leaf-Disease-Detection/releases/download/v2.0/severity_classifier_dataset_model.keras"
    }
    for path, url in model_configs.items():
        if not os.path.exists(path) or os.path.getsize(path) < 102400:
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Download failed: {e}")

def run_prediction(image_path):
    import tensorflow as tf
    import keras
    from keras import layers

    ensure_models_exist()

    # Capsule Network Custom Components
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
        # --- NEW PIPELINE STEP ---
        # Apply Median Filter + Chan-Vese
        processed_img = project_pipeline_segmentation(image_path)
        
        # --- FINAL FORMATTING ---
        # Resize to model input and normalize
        img_final = cv2.resize(processed_img, (128, 128))
        img_array = img_final.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        custom_map = {
            "Project>CapsuleLayer": CapsuleLayer,
            "Project>squash": squash,
            "Project>get_capsule_length": get_capsule_length,
            "margin_loss": lambda y_true, y_pred: y_pred
        }

        # --- STAGE 1: BINARY ---
        m_bin = keras.models.load_model(BINARY_MODEL_PATH, custom_objects=custom_map, compile=False)
        preds_bin = m_bin.predict(img_array, verbose=0)[0]
        bin_idx = np.argmax(preds_bin)
        bin_conf = float(preds_bin[bin_idx])

        del m_bin
        gc.collect()

        if bin_idx == 1:
            return "Healthy", "Healthy", bin_conf

        # --- STAGE 2: SEVERITY ---
        m_sev = keras.models.load_model(SEVERITY_MODEL_PATH, custom_objects=custom_map, compile=False)
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