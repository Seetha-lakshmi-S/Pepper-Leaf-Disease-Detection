import os
os.environ['TF_DISABLE_XLA_DYNAMIC_COMPILER'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

tf.config.optimizer.set_jit(False)

def capsule_length(vectors):
    """Compute capsule length."""
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + keras.backend.epsilon())

def squash(vectors, axis=-1):
    """Squashing function for capsules."""
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    return scale * vectors

class CapsuleLayer(keras.layers.Layer):
    """Custom Capsule Layer."""
    def __init__(self, num_capsule, dim_capsule, routings=2, **kwargs):
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
            initializer='glorot_uniform', name='W')

    def call(self, inputs, training=None):
        inputs_hat = tf.einsum('bji,jkio->bkjo', inputs, self.W)
        b = tf.zeros([tf.shape(inputs)[0], self.num_capsule, self.input_num_capsule])
        
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s_j = tf.einsum('bkl,bkld->bkd', c, inputs_hat)
            v_j = squash(s_j)
            if i < self.routings - 1:
                agreement = tf.einsum('bkd,bkld->bkl', v_j, inputs_hat)
                b += agreement
        return v_j

def margin_loss(y_true, y_pred):
    """Margin loss for capsules."""
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def augment(image, label):
    image, label = normalize(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label

def build_model(input_shape, num_classes):
    """Build CNN-CapsNet hybrid model."""
    input_image = layers.Input(shape=input_shape)
    
    # CNN backbone
    x = layers.Conv2D(128, 3, padding='same')(input_image)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Primary capsules
    x = layers.Conv2D(filters=8*32, kernel_size=9, strides=2, padding='valid')(x)
    x = layers.Reshape(target_shape=(-1, 8))(x)
    primary_caps = layers.Lambda(squash)(x)
    
    # Digit capsules
    digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=2)(primary_caps)
    output = layers.Lambda(capsule_length)(digit_caps)
    
    return models.Model(inputs=input_image, outputs=output)

def train_and_evaluate(args):
    """Main training pipeline."""
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    print("\n--- Running Module 5: Final CNN-CapsNet Model Training ---")
    
    # ✅ Dataset-specific filenames - NO timestamps
    model_name = f"{dataset_name}_model.keras"
    cm_name = f"{dataset_name}_cm.png"
    curves_name = f"{dataset_name}_curves.png"
    csv_name = f"{dataset_name}_training_log.csv"
    log_dir = f"logs/{dataset_name}"
    
    print(f"Split ratio: {args.split_ratio} (train/val = {(1-args.split_ratio)*100:.0f}/{args.split_ratio*100:.0f})")
    
    # Load datasets with COMMAND LINE SPLIT RATIO
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        args.input_dir, validation_split=args.split_ratio, subset="training",
        seed=123, image_size=(128, 128), batch_size=args.batch_size
    )
    
    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    
    # Quick class distribution check
    all_labels = []
    for _, labels in raw_train_ds.take(20):
        all_labels.extend(labels.numpy().flatten())
    print(f"Dataset: {dataset_name}")
    print(f"Classes: {class_names}")
    print(f"Distribution: {dict(zip(class_names, np.bincount(all_labels)))}")
    
    # Validation dataset (same split ratio)
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        args.input_dir, validation_split=args.split_ratio, subset="validation",
        seed=123, image_size=(128, 128), batch_size=args.batch_size
    )
    
    # For 80/20: use full validation as test set
    # For 70/30: split validation into val(15%) + test(15%)
    if args.split_ratio == 0.2:
        # 80/20: full 20% as test (no separate val split needed during training)
        val_ds = temp_ds
        test_ds = temp_ds
    else:
        # 70/30: split 30% validation into 15% val + 15% test
        val_size = int(0.5 * len(temp_ds))
        val_ds = temp_ds.take(val_size)
        test_ds = temp_ds.skip(val_size)
    
    print(f"Batches - Train: {len(raw_train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Data pipelines
    def one_hot_labels(image, label):
        return image, tf.one_hot(label, depth=num_classes)
    
    train_ds = (raw_train_ds.map(one_hot_labels)
                .map(augment)
                .shuffle(512)
                .prefetch(tf.data.AUTOTUNE))
    
    val_ds = (val_ds.map(one_hot_labels)
              .map(normalize)
              .cache()
              .prefetch(tf.data.AUTOTUNE))
    
    test_ds = (test_ds.map(one_hot_labels)
               .map(normalize)
               .cache()
               .prefetch(tf.data.AUTOTUNE))
    
    # Build and compile model
    model = build_model((128, 128, 3), num_classes)
    optimizer = keras.optimizers.AdamW(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss=margin_loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(3, name="top3_accuracy")
        ]
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        CSVLogger(csv_name),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    # Train
    print("🚀 Starting training...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=args.epochs, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Test evaluation (silent)
    test_results = model.evaluate(test_ds, return_dict=True, verbose=0)
    print(f"\n✅ Test Results: {test_results}")
    
    # Predictions for confusion matrix
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    
    # ✅ Classification Report IN LOG (NO FILE)
    print(f"\n📈 CLASSIFICATION REPORT - {dataset_name}:")
    print("=" * 70)
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    print("=" * 70)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(cm_name, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(curves_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Training complete!")
    print(f"Saved:")
    print(f"   📁 {model_name}")
    print(f"   📊 {csv_name}")
    print(f"   🖼️  {cm_name}")
    print(f"   📈 {curves_name}")
    print(f"   📂 TensorBoard: {log_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the definitive CNN-CapsNet model.')
    parser.add_argument('--input_dir', required=True, help='Dataset directory path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--split_ratio', type=float, default=0.3, help='Validation split ratio (0.2=80/20, 0.3=70/30)')
    args = parser.parse_args()
    
    train_and_evaluate(args)