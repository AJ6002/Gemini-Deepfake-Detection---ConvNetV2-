import os
import sys

# --- WSL GPU FIX ---
# Automatically inject the CUDA library path for WSL2 sessions
cuda_path = "/usr/lib/wsl/lib"
if os.path.exists(cuda_path):
    if "LD_LIBRARY_PATH" not in os.environ:
        os.environ["LD_LIBRARY_PATH"] = cuda_path
    elif cuda_path not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}:{os.environ['LD_LIBRARY_PATH']}"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_hub
import pywt
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "ULTRA_SWIN_DATA")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "Swin_Diff_7Ch", "convnext_diff_7ch_v2.keras")
IMG_SIZE = 256
BATCH_SIZE = 16 # Safe for 6GB VRAM
EPOCHS = 30

# Enable Mixed Precision for RTX 3050 speedup
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(f"GPU Ready: {tf.config.list_physical_devices('GPU')}")

# --- 7-CHANNEL WAVELET UTILS ---
def get_wavelet_channels(img_tensor):
    """Extracts 4 DWT sub-bands (LH, HL, HH, fused) to create 7-channel input."""
    img = img_tensor.numpy()
    channels_7 = []
    
    for i in range(img.shape[0]): # Batch loop
        sample = img[i]
        # Ch 1-3: RGB
        # Ch 4-6: DWT sub-bands
        coeffs2 = pywt.dwt2(sample[:,:,0], 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # Resize sub-bands back to 256x256
        LH = cv2.resize(LH, (IMG_SIZE, IMG_SIZE))
        HL = cv2.resize(HL, (IMG_SIZE, IMG_SIZE))
        HH = cv2.resize(HH, (IMG_SIZE, IMG_SIZE))
        
        # Ch 7: High-freq fusion
        fused = np.sqrt(LH**2 + HL**2 + HH**2)
        
        # Stack 7 channels
        combined = np.dstack([sample, LH, HL, HH, fused])
        channels_7.append(combined)
        
    return tf.convert_to_tensor(np.array(channels_7), dtype=tf.float32)

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# --- ARCHITECTURE: CONVNEXT-DIFF-7CH ---
def build_convnext_7ch():
    print("Building ConvNeXt-Diff-7Ch Architecture (Phase 4)...")
    
    # 7-Channel Input Layer
    input_7ch = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 7), name="7ch_input")
    
    # 1x1 Projection Layer: Compresses 7 channels to 3 for ImageNet weights
    # This preserves the wavelet artifacts for the backbone
    x = layers.Conv2D(3, (1, 1), padding='same', name="wavelet_projection")(input_7ch)
    
    # ConvNeXt Backbone (ImageNet Pretrained)
    # Using Small variant for best balance on 6GB VRAM
    backbone = tf.keras.applications.ConvNeXtSmall(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    
    x = backbone(x)
    
    # Specialist Head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = keras.Model(inputs=input_7ch, outputs=output)
    return model, backbone

# --- DATASET PIPELINE ---
print(f"Loading Dataset from: {DATASET_DIR}")
ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Custom 7-channel mapping (using Haar Wavelets)
import cv2
def map_7ch(img, label):
    # This is a bit slow in TF Graph, but necessary for the 7-ch fusion
    def wrapper(img_np):
        res = []
        for i in range(img_np.shape[0]):
            sample = img_np[i] / 255.0
            coeffs2 = pywt.dwt2(sample[:,:,0], 'haar')
            LL, (LH, HL, HH) = coeffs2
            LH = cv2.resize(LH, (IMG_SIZE, IMG_SIZE))
            HL = cv2.resize(HL, (IMG_SIZE, IMG_SIZE))
            HH = cv2.resize(HH, (IMG_SIZE, IMG_SIZE))
            fused = np.sqrt(LH**2 + HL**2 + HH**2)
            combined = np.dstack([sample, LH, HL, HH, fused])
            res.append(combined)
        return np.array(res).astype(np.float32)

    img_7ch = tf.py_function(wrapper, [img], tf.float32)
    img_7ch.set_shape([None, IMG_SIZE, IMG_SIZE, 7])
    return img_7ch, label

train_ds = ds.map(map_7ch).prefetch(tf.data.AUTOTUNE)

# --- TRAINING ---
model, backbone = build_convnext_7ch()

# Phase 1: Warmup (Frozen backbone)
print("Phase 1: Warming up the specialist head...")
backbone.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, epochs=3)

# Phase 2: Deep Fine-Tuning
print("Phase 2: Full Fine-tuning (Deep backbone surgery)...")
backbone.trainable = True
# Unfreeze only the top layers to prevent catastrophic forgetting
for layer in backbone.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5), # Very low learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n✅ MEGA-TRAIN COMPLETE!")
print(f"Final Model Saved to: {MODEL_SAVE_PATH}")
