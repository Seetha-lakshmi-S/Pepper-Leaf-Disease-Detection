import os
import gc
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# --- RENDER & MEMORY OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_prediction(image_path):
    """
    Two-stage prediction:
    1. Binary (Diseased vs Healthy)
    2. Severity (Early vs Mid vs Advanced)
    Optimized for 512MB RAM using lazy imports and manual GC.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, backend as K

    # --- 1. CUSTOM COMPONENTS (CAPSULE NETWORK LOGIC) ---
    @keras.utils.register_keras_serializable(package="Custom")
    def squash(vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

    @keras.utils.register_keras_serializable(package="Custom")
    def capsule_length(vectors):
        return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + K.epsilon())

    @keras.utils.register_keras_serializable(package="Custom")
    class CapsuleLayer(layers.Layer):
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
                s_j = tf.einsum('bki,bkio->bko', c, inputs_hat)
                v_j = squash(s_j)
                if i < self.routings - 1:
                    agreement = tf.einsum('bko,bkio->bki', v_j, inputs_hat)
                    b += agreement
            return v_j

        def get_config(self):
            config = super().get_config()
            config.update({'num_capsule': self.num_capsule, 'dim_capsule': self.dim_capsule, 'routings': self.routings})
            return config

    try:
        # --- 2. IMAGE PREPROCESSING (BACKGROUND IGNORE) ---
        img = Image.open(image_path).convert('RGB')
        
        # Boost contrast to make disease spots stand out from the green leaf
        img = ImageOps.autocontrast(img, cutoff=1)
        
        # Apply smoothing to ignore background noise/dirt while keeping lesion edges
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # Standard Resize for Model
        img = img.resize((128, 128))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        custom_map = {"CapsuleLayer": CapsuleLayer, "squash": squash, "capsule_length": capsule_length}

        # --- 3. STAGE 1: BINARY PREDICTION ---
        m_bin = keras.models.load_model("binary_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, compile=False)
        
        raw_bin = m_bin.predict(img_array, verbose=0)[0]
        # Ensure we handle logits vs probabilities correctly
        bin_probs = tf.nn.softmax(raw_bin).numpy()
        bin_idx = np.argmax(bin_probs)
        
        # Alphabetical sorting: [0: Diseased, 1: Healthy]
        is_diseased = (bin_idx == 0)
        bin_conf = float(bin_probs[bin_idx])

        # Cleanup Stage 1 from RAM immediately
        del m_bin
        K.clear_session()
        gc.collect()

        if not is_diseased:
            return "Healthy", "Healthy", bin_conf

        # --- 4. STAGE 2: SEVERITY PREDICTION ---
        m_sev = keras.models.load_model("severity_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, compile=False)
        
        raw_sev = m_sev.predict(img_array, verbose=0)[0]
        sev_probs = tf.nn.softmax(raw_sev).numpy()
        
        # Matching your JSON/Folder structure
        stages = ["Advanced Stage", "Early Stage", "Mid Stage"]
        sev_idx = np.argmax(sev_probs)
        
        res_stage = stages[sev_idx]
        res_conf = float(sev_probs[sev_idx])

        # Final Cleanup
        del m_sev
        K.clear_session()
        gc.collect()

        return "Bacterial Disease", res_stage, res_conf

    except Exception as e:
        print(f"Prediction logic error: {e}")
        gc.collect()
        return "PredictionError", str(e), 0.0