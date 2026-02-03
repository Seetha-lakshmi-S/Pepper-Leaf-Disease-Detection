import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import traceback
import os

# --- MEMORY HYGIENE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --- ALL Custom Components ---
def capsule_length(vectors):
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1))

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

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

    def call(self, inputs, training=None):
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

def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

custom_objects = {
    'CapsuleLayer': CapsuleLayer,
    'margin_loss': margin_loss,
    'squash': squash,
    'capsule_length': capsule_length
}

# --- LAZY LOADING STRATEGY ---
# We do NOT load models at the top level anymore to save RAM during startup
BINARY_MODEL_PATH = "binary_classifier_dataset_model.keras"
SEVERITY_MODEL_PATH = "severity_classifier_dataset_model.keras"

def run_prediction(image_path):
    try:
        # Load Image
        img = Image.open(image_path).resize((128, 128))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = tf.expand_dims(img_array.astype(np.float32), 0)

        # 1. Load & Run Binary Model
        # compile=False saves significant memory
        binary_model = keras.models.load_model(BINARY_MODEL_PATH, custom_objects=custom_objects, compile=False)
        binary_preds = binary_model.predict(img_array, verbose=0)[0]
        
        binary_class_names = ['Diseased', 'Healthy']
        binary_index = np.argmax(binary_preds)
        binary_result = binary_class_names[binary_index]
        binary_conf = float(np.clip(binary_preds[binary_index], 0.0, 1.0))

        # IMPORTANT: Delete model from RAM immediately after use
        del binary_model
        keras.backend.clear_session()

        if binary_result == 'Healthy':
            return "Healthy", None, binary_conf

        # 2. Load & Run Severity Model (Only if diseased)
        severity_model = keras.models.load_model(SEVERITY_MODEL_PATH, custom_objects=custom_objects, compile=False)
        severity_preds = severity_model.predict(img_array, verbose=0)[0]
        
        severity_class_names = ["1_Early_Stage", "2_Mid_Stage", "3_Advanced_Stage"]
        severity_index = np.argmax(severity_preds)
        severity_label = severity_class_names[severity_index].split('_', 1)[1].replace('_', ' ')
        severity_conf = float(np.clip(severity_preds[severity_index], 0.0, 1.0))

        # Cleanup again
        del severity_model
        keras.backend.clear_session()

        return ("Bacterial Disease", severity_label, severity_conf)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "PredictionError", None, 0.0