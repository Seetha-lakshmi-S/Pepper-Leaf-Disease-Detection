import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# --------- ALL Custom Components (REQUIRED) --------------
@tf.function
def capsule_length(vectors):
    return tf.sqrt(
        tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon()
    )

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(
        s_squared_norm + keras.backend.epsilon()
    )
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
            shape=[self.input_num_capsule, self.num_capsule,
                   self.input_dim_capsule, self.dim_capsule],
            initializer="glorot_uniform",
            name="W",
            dtype=tf.float32
        )

    def call(self, inputs, training=None):
        inputs_hat = tf.einsum("bji,jkio->bkjo", inputs, self.W)
        b = tf.zeros(
            [tf.shape(inputs)[0], self.num_capsule, self.input_num_capsule],
            dtype=inputs.dtype
        )

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s_j = tf.einsum("bkl,bkld->bkd", c, inputs_hat)
            v_j = squash(s_j)
            if i < self.routings - 1:
                agreement = tf.einsum("bkd,bkld->bkl", v_j, inputs_hat)
                b += agreement
        return v_j

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "routings": self.routings
        })
        return config

def margin_loss(y_true, y_pred):
    L = (
        y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) +
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    )
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# --------- TWO-STAGE INFERENCE FUNCTION ------------------
def predict_two_stage(image_path, binary_model_path, severity_model_path):
    print("\n" + "=" * 65)
    print("🩺 TWO-STAGE PEPPER LEAF DISEASE DIAGNOSIS SYSTEM")
    print("=" * 65)

    custom_objects = {
        "CapsuleLayer": CapsuleLayer,
        "margin_loss": margin_loss,
        "squash": squash,
        "capsule_length": capsule_length
    }

    # ---------------- Load Models ----------------
    print("\n🔄 Loading trained models...")
    binary_model = keras.models.load_model(
        binary_model_path, custom_objects=custom_objects
    )
    severity_model = keras.models.load_model(
        severity_model_path, custom_objects=custom_objects
    )
    print("✅ Models loaded successfully")

    # ---------------- Load Image ----------------
    img = keras.utils.load_img(image_path, target_size=(128, 128))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = tf.cast(img_array, tf.float32)

    print(f"🖼 Image loaded: {os.path.basename(image_path)}")

    # Show image (demo-friendly)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Pepper Leaf Image")
    plt.show()

    # ---------------- STAGE 1 ------------------------
    print("\n" + "-" * 45)
    print("[STAGE 1] General Diagnosis (Healthy / Diseased)")
    print("-" * 45)

    binary_preds = binary_model.predict(img_array, verbose=0)[0]
    binary_classes = ["Diseased", "Healthy"] 
    binary_index = np.argmax(binary_preds)

    binary_label = binary_classes[binary_index]
    binary_confidence = float(binary_preds[binary_index])

    print(f"Prediction: {binary_label}")
    print(f"Confidence: {binary_confidence:.2%}")

    # Low-confidence warning
    if binary_confidence < 0.60:
        print("⚠️ WARNING: Low confidence prediction. Consider retaking image.")

    if binary_label == "Healthy":
        print("\n" + "=" * 65)
        print("🎉 FINAL DIAGNOSIS: HEALTHY PEPPER LEAF")
        print(f"✅ Confidence: {binary_confidence:.2%}")
        print("=" * 65)
        return

    # ---------------- STAGE 2 ------------------------
    print("\n" + "-" * 45)
    print("[STAGE 2] Disease Severity Classification")
    print("-" * 45)

    severity_classes = [
        "1_Early_Stage",
        "2_Mid_Stage",
        "3_Advanced_Stage"
    ]

    severity_preds = severity_model.predict(img_array, verbose=0)[0]
    severity_index = np.argmax(severity_preds)
    severity_label = severity_classes[severity_index]
    severity_confidence = float(severity_preds[severity_index])

    clean_label = severity_label.split("_", 1)[1].replace("_", " ")

    print("\n" + "=" * 65)
    print("🏥 FINAL DIAGNOSIS")
    print("=" * 65)
    print(f"⚠️ DISEASE SEVERITY: {clean_label.upper()}")
    print(f"📊 Severity confidence: {severity_confidence:.2%}")
    print(f"🔬 Binary confidence: {binary_confidence:.2%}")
    print("=" * 65)

    # Probability breakdown
    print("\nDetailed Severity Probabilities:")
    for i, cls in enumerate(severity_classes):
        name = cls.split("_", 1)[1].replace("_", " ")
        print(f"  {name:<15}: {severity_preds[i]:.2%}")

# ------------------- MAIN --------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Pepper Leaf Disease Diagnosis (CapsNet)"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--binary_model", required=True, help="Binary model (.keras)")
    parser.add_argument("--severity_model", required=True, help="Severity model (.keras)")
    args = parser.parse_args()

    predict_two_stage(args.image, args.binary_model, args.severity_model)

if __name__ == "__main__":
    main()