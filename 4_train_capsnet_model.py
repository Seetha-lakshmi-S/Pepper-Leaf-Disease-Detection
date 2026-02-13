import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ---------------- GPU MEMORY GROWTH ----------------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# ---------------- CAPSULE FUNCTIONS ----------------
@keras.utils.register_keras_serializable(package="Project")
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

@keras.utils.register_keras_serializable(package="Project")
def get_capsule_length(vectors):
    """Magnitude of capsule vectors"""
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
            shape=[self.input_num_capsule, self.num_capsule,
                   self.input_dim_capsule, self.dim_capsule],
            initializer="glorot_uniform",
            name="W"
        )

    def call(self, inputs):
        # [batch, input_caps, input_dim] -> [batch, num_capsule, input_caps, dim_capsule]
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

# ---------------- LOSS ----------------
def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# ---------------- MODEL BUILDING ----------------
def build_cnn_capsnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # --- CNN Backbone ---
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # --- Primary Capsules ---
    x = layers.Conv2D(16 * 32, kernel_size=9, strides=2, padding="valid", activation="relu")(x)
    x = layers.Reshape((-1, 16))(x)
    primary_caps = layers.Lambda(squash, name="primary_caps_squash")(x)

    # --- Digit Capsules ---
    digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=32, routings=3, name="digit_caps")(primary_caps)

    # --- Capsule Length as Output ---
    outputs = layers.Lambda(get_capsule_length, name="capsule_magnitude")(digit_caps)

    model = models.Model(inputs, outputs, name="CNN-CapsNet")
    return model

# ---------------- DATA PIPELINE ----------------
def preprocess_image(image, label, num_classes):
    image = tf.cast(image, tf.float32)/255.0
    label = tf.one_hot(label, num_classes)
    return image, label

def augment(image, label, num_classes):
    image, label = preprocess_image(image, label, num_classes)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

# ---------------- TRAINING ----------------
def train(args):
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    print(f"\n--- Training CNN-CapsNet: {dataset_name} ---")

    # Load dataset
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        args.input_dir, validation_split=0.3, subset="training",
        seed=123, image_size=(128,128), batch_size=args.batch_size
    )
    raw_temp_ds = tf.keras.utils.image_dataset_from_directory(
        args.input_dir, validation_split=0.3, subset="validation",
        seed=123, image_size=(128,128), batch_size=args.batch_size
    )

    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Classes found: {class_names}")

    # Split temp into val and test
    val_batches = tf.data.experimental.cardinality(raw_temp_ds)//2
    raw_val_ds = raw_temp_ds.take(val_batches)
    raw_test_ds = raw_temp_ds.skip(val_batches)

    # Prepare pipelines
    train_ds = raw_train_ds.map(lambda x,y: augment(x,y,num_classes)).shuffle(512).prefetch(tf.data.AUTOTUNE)
    val_ds = raw_val_ds.map(lambda x,y: preprocess_image(x,y,num_classes)).prefetch(tf.data.AUTOTUNE)
    test_ds = raw_test_ds.map(lambda x,y: preprocess_image(x,y,num_classes)).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_cnn_capsnet((128,128,3), num_classes)

    # Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=margin_loss,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
    )

    # Callbacks with dataset-specific names
    callbacks = [
        CSVLogger(f"{dataset_name}_training_log.csv"),
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"{dataset_name}_best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    # Train
    print("🚀 Starting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Evaluate on test
    print("\n📊 Evaluating on Independent Test Set...")
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix saved with dataset name
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.savefig(f"{dataset_name}_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\n✅ Training complete. Best model saved as '{dataset_name}_best_model.keras' and metrics saved with dataset name.")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Dataset folder path")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    train(args)