#Install required packages
# !pip install pydicom lime shap


# Import libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras  import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pydicom
import gc
import matplotlib.pyplot as plt
from google.colab import drive
# Mount Google drive
drive.mount('/content/drive')

# Configuration
config = {
    'image_size': 224,
    'batch_size': 16,  # Reduced batch size to save memory
    'epochs': 5,
    'adjust_contrast': True,
    'contrast_factor': 1.2,
    'checkpoint_path': '/content/drive/MyDrive/pneumonia_model/checkpoints/model.h5',
    'data_path': '/content/drive/MyDrive/rsna-pneumonia-detection-challenge',
    'sample_size': None  # Set to a number for testing, None for full dataset
}

# Enable memory-efficient features
print("Enabling mixed precision training...")
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# ---------- MODEL DEFINITION ----------
def build_model():
    """Build the neural network model."""
    # Use a lighter model to save memory
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(config['image_size'], config['image_size'], 3)
    )

    # Freeze base model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),  # Smaller than original
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model

# ---------- DATA LOADING WITH TF.DATA ----------
# Function to preprocess an image (used inside tf.data pipeline)
def preprocess_dicom(file_path):
    """Process a single DICOM file into the format needed for the model."""
    try:
        dicom = pydicom.dcmread(file_path.numpy().decode('utf-8'))
        image = dicom.pixel_array

        # Convert to float32
        image = image.astype('float32')

        # Normalize pixel values
        if np.min(image) != np.max(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        else:
            image = np.zeros_like(image)

        # Resize and expand dimensions for model input
        image = tf.image.resize(
            tf.expand_dims(image, -1),
            [config['image_size'], config['image_size']]
        )

        # Convert to 3 channels
        image = tf.tile(image, [1, 1, 3])

        return image
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return tf.zeros([config['image_size'], config['image_size'], 3])

# Wrap preprocessing function for tf.data
def process_path(file_path, label):
    """Wrapper function for tf.data pipeline."""
    image = tf.py_function(
        preprocess_dicom,
        [file_path],
        tf.float32
    )
    image.set_shape([config['image_size'], config['image_size'], 3])
    return image, label

# Create a dataset generator
def create_dataset(df, image_dir, batch_size=16, is_training=True):
    """Creates a TensorFlow Dataset from a DataFrame of patient IDs and labels."""
    # Create file paths and labels
    file_paths = [os.path.join(image_dir, f"{patient_id}.dcm") for patient_id in df['patientId']]
    labels = df['Target'].values

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Shuffle if training
    if is_training:
        ds = ds.shuffle(buffer_size=1000)

    # Process images
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch, cache, and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# ---------- MAIN TRAINING FUNCTION ----------
def train_pneumonia_detection():
    """Main function to train the pneumonia detection model."""
    print("Starting pneumonia detection training with memory-efficient approach")

    # Check dataset structure
    print("Verifying dataset...")
    labels_path = os.path.join(config['data_path'], 'stage_2_train_labels_original.csv')
    image_dir = os.path.join(config['data_path'], 'stage_2_train_images')

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Images directory not found at {image_dir}")

    # Load labels
    print("Loading and splitting dataset...")
    labels_df = pd.read_csv(labels_path)

    # Use sample for testing if specified
    if config['sample_size']:
        labels_df = labels_df.head(config['sample_size'])

    # Split dataset
    train_df, temp_df = train_test_split(
        labels_df, test_size=0.3, random_state=42,
        stratify=labels_df['Target']
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df['Target']
    )

    print("\nData split sizes:")
    print(f"Training set: {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    print(f"Test set: {len(test_df)}")

    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_dataset(train_df, image_dir, config['batch_size'], is_training=True)
    val_dataset = create_dataset(val_df, image_dir, config['batch_size'], is_training=False)
    test_dataset = create_dataset(test_df, image_dir, config['batch_size'], is_training=False)

    # Build model
    print("Building model...")
    model = build_model()

    # Setup callbacks
    print("Setting up training callbacks...")
    checkpoint_dir = os.path.dirname(config['checkpoint_path'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config['checkpoint_path'],
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Memory cleanup callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )

    return model, history, test_dataset

# ---------- EVALUATION FUNCTIONS ----------
def evaluate_model(model, test_dataset):
    """Evaluate the trained model."""
    print("\nEvaluating model on test set...")

    # Get predictions
    y_pred_prob = model.predict(test_dataset)

    # Extract true labels from the test dataset
    y_true = np.concatenate([y for _, y in test_dataset], axis=0)

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"Test AUC: {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# ---------- VISUALIZATION FUNCTIONS ----------
def plot_training_history(history):
    """Plot the training history metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(['Train', 'Validation'])

    # Plot loss
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(['Train', 'Validation'])

    # Plot AUC
    axes[2].plot(history.history['auc'])
    axes[2].plot(history.history['val_auc'])
    axes[2].set_title('Model AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Loss: {history.history['loss'][-1]:.4f}")
    print(f"AUC: {history.history['auc'][-1]:.4f}")

    print("\nFinal Validation Metrics:")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

import time
import random
import numpy as np
from IPython.display import display, Javascript, HTML
import threading

class ColabKeepAlive:
    """
    A class to keep Google Colab sessions active by simulating activity.
    """

    def __init__(self, interval_minutes=9.5, jitter_seconds=30, verbose=True):
        """
        Initialize the keep-alive service.

        Args:
            interval_minutes (float): Time between activity simulations in minutes
            jitter_seconds (int): Random jitter in seconds to add to the interval
            verbose (bool): Whether to print status messages
        """
        self.interval_minutes = interval_minutes
        self.jitter_seconds = jitter_seconds
        self.verbose = verbose
        self.running = False
        self.thread = None

    def _simulate_activity(self):
        """Simulates user activity by executing JavaScript to interact with the page."""
        # Simulate a scroll action
        display(Javascript('''
            function randomScroll() {
                window.scrollBy({
                    top: Math.floor(Math.random() * 100) - 50,
                    behavior: 'smooth'
                });
            }
            randomScroll();
        '''))

        # Small computation to keep Python kernel active
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 100)

        if self.verbose:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"✅ Keep-alive action performed at {current_time}")

    def _keep_alive_loop(self):
        """Main loop that periodically simulates activity."""
        while self.running:
            self._simulate_activity()

            # Calculate next interval with jitter
            jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds)
            wait_seconds = (self.interval_minutes * 60) + jitter

            if self.verbose:
                print(f"Next keep-alive action in {wait_seconds:.1f} seconds")

            # Sleep until next iteration
            time.sleep(wait_seconds)

    def start(self):
        """Start the keep-alive service in a background thread."""
        if self.running:
            print("Keep-alive service is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()

        if self.verbose:
            print("🚀 Colab keep-alive service started")
            print(f"⏱️ Activity interval: ~{self.interval_minutes} minutes (±{self.jitter_seconds} seconds)")
            display(HTML("""
            <div style="background-color: #d2f4d3; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <p style="color: #0b6623; margin: 0;">
                    <b>Keep-alive service is running</b> - Your Colab session should remain active.
                    You can continue working in other cells.
                </p>
            </div>
            """))

    def stop(self):
        """Stop the keep-alive service."""
        if not self.running:
            print("Keep-alive service is not running")
            return

        self.running = False
        self.thread.join(timeout=1.0)

        if self.verbose:
            print("🛑 Colab keep-alive service stopped")


# Example usage:
if __name__ == "__main__":
    # Create and start the keep-alive service
    # Default settings will perform an action every ~9.5 minutes (with small random variation)
    # which is frequent enough to prevent Colab's ~90 minute timeout
    keepalive = ColabKeepAlive(interval_minutes=9.5, verbose=True)
    keepalive.start()

    # To stop the service:
    # keepalive.stop()


# ---------- EXECUTION ----------
# Add this to your notebook to execute the training
if __name__ == "__main__":
    try:
        # Train the model
        model, history, test_dataset = train_pneumonia_detection()

        # Plot training history
        plot_training_history(history)

        # Evaluate model
        evaluate_model(model, test_dataset)

    except Exception as e:
        print(f"Error during training: {str(e)}")