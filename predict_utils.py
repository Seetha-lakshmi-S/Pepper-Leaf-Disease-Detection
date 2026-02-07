import os
import gc
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# 1. RENDER OPTIMIZATION: Force CPU only and minimize memory fragmentation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_prediction(image_path):
    """
    Two-stage prediction: 
    1. Binary (Diseased vs Healthy)
    2. Severity (Early vs Mid vs Advanced)
    """
    # Lazy import to save initial RAM
    import tensorflow as tf
    from tensorflow import keras
    
    # ----------------- CUSTOM COMPONENTS (MATCHING TRAINING) -----------------
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
        # ----------------- NOISE REDUCTION PIPELINE -----------------
        # This ensures the model focuses on the leaf and ignores the background
        img = Image.open(image_path).convert('RGB')
        
        # 1. Enhance Contrast: Makes lesions stand out from the green leaf
        img = ImageOps.autocontrast(img, cutoff=1)
        
        # 2. Denoise: Smooths background textures while keeping disease spot edges
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # 3. Standard Resize
        img = img.resize((128, 128))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        # Mapping for Keras 3.x
        custom_map = {
            "CapsuleLayer": CapsuleLayer, 
            "squash": squash, 
            "get_capsule_length": get_capsule_length,
            "margin_loss": lambda y_true, y_pred: y_pred 
        }

        # ----------------- STAGE 1: BINARY -----------------
        # Loading with compile=False and safe_mode=False is mandatory for Render/Keras3
        m_bin = keras.models.load_model("binary_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, compile=False, safe_mode=False)
        
        bin_probs = m_bin.predict(img_array, verbose=0)[0]
        bin_idx = np.argmax(bin_probs)
        
        # Mapping based on folder alphabetical sort: [diseased, healthy]
        # Index 0 = Diseased, Index 1 = Healthy
        is_diseased = (bin_idx == 0)
        bin_conf = float(bin_probs[bin_idx])

        # Cleanup Stage 1 from RAM
        del m_bin
        gc.collect()

        if not is_diseased:
            return "Healthy", "None", bin_conf

        # ----------------- STAGE 2: SEVERITY -----------------
        m_sev = keras.models.load_model("severity_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, compile=False, safe_mode=False)
        
        sev_probs = m_sev.predict(img_array, verbose=0)[0]
        
        # Mapping based on folder alphabetical sort: [advanced, early, mid]
        stages = ["Advanced Stage", "Early Stage", "Mid Stage"]
        sev_idx = np.argmax(sev_probs)
        
        res_stage = stages[sev_idx]
        res_conf = float(sev_probs[sev_idx])

        # Final memory cleanup
        del m_sev
        keras.backend.clear_session()
        gc.collect()

        return "Diseased", res_stage, res_conf

    except Exception as e:
        gc.collect()
        return "Error", str(e), 0.0