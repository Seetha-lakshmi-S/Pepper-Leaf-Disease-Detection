import os
import gc
import numpy as np
from PIL import Image

# Force CPU-only and minimize logs before any TF logic starts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_FOR_DEVICE'] = '1'

def run_prediction(image_path):
    # 1. LAZY IMPORT: Don't load TF until we actually need to predict
    import tensorflow as tf
    from tensorflow import keras
    
    # ----------------- RE-DECLARE CUSTOM COMPONENTS -----------------
    # These must be inside the function or accessible here
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

    try:
        # 2. IMAGE PREPROCESSING
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)

        custom_map = {"CapsuleLayer": CapsuleLayer, "squash": squash, "capsule_length": capsule_length}

        # --- STAGE 1: BINARY ---
        # Note: Use compile=False to avoid loading training state (Optimizer)
        m_bin = keras.models.load_model("binary_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, 
                                        compile=False)
        
        raw_bin = m_bin.predict(img_array, verbose=0)[0]
        bin_probs = tf.nn.softmax(raw_bin).numpy()
        bin_idx = np.argmax(bin_probs)
        is_diseased = (bin_idx == 0) 
        bin_conf = float(bin_probs[bin_idx])

        # DESTROY BINARY MODEL IMMEDIATELY
        del m_bin
        keras.backend.clear_session()
        gc.collect()

        if not is_diseased:
            return "Healthy", None, bin_conf

        # --- STAGE 2: SEVERITY ---
        m_sev = keras.models.load_model("severity_classifier_dataset_model.keras", 
                                        custom_objects=custom_map, 
                                        compile=False)
        
        raw_sev = m_sev.predict(img_array, verbose=0)[0]
        sev_probs = tf.nn.softmax(raw_sev).numpy()
        
        stages = ["Early Stage", "Mid Stage", "Advanced Stage"]
        sev_idx = np.argmax(sev_probs)
        
        res_stage = stages[sev_idx]
        res_conf = float(sev_probs[sev_idx])

        # FINAL CLEANUP
        del m_sev
        keras.backend.clear_session()
        gc.collect()

        return ("Bacterial Disease", res_stage, res_conf)

    except Exception as e:
        gc.collect()
        print(f"Prediction Error: {e}")
        return "PredictionError", None, 0.0