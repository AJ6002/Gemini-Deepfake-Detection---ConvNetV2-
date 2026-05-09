#!/usr/bin/env python3
import os

# --- CRITICAL WSL STABILITY FLAGS (Must be at the very top) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import numpy as np
import cv2, pywt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- FIX 2: MIXED PRECISION (halves GPU memory for all weights/activations) ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- LIMIT THREADS TO PREVENT MALLOC CORRUPTION ---
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# ── CONFIG ───────────────────────────────────────────────────
MANIFEST_CSV = "mega_dataset_manifest.csv"
SAVE_DIR     = 'models/hybrid_7ch'
IMG_SIZE     = 256
BATCH_SIZE   = 4   # FIX 1: Halved from 8 → 4 for large Xception model
EPOCHS       = 25
STEPS_EPOCH  = 1000
VAL_STEPS    = 100

# ── GPU SETUP ────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {gpus[0]} | Mixed Precision: float16")

# ── PREPROCESSING ─────────────────────────────────────────────
_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def normalize_band(b, size):
    b = cv2.resize(b.astype(np.float32), (size, size))
    lo, hi = b.min(), b.max()
    return (b - lo) / (hi - lo + 1e-7)

def extract_7ch(img_rgb, size):
    img = cv2.resize(img_rgb, (size, size))
    rgb = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    LH_n = normalize_band(LH, size)
    HL_n = normalize_band(HL, size)
    HH_n = normalize_band(HH, size)
    HF   = normalize_band(np.sqrt(LH**2 + HL**2 + HH**2), size)
    return np.dstack([rgb, LH_n[...,None], HL_n[...,None], HH_n[...,None], HF[...,None]]).astype(np.float32)

def crop_face(img_rgb):
    try:
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = _cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            h, w = img_rgb.shape[:2]; side = min(h, w)
            return img_rgb[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        m = int(w * 0.2)
        return img_rgb[max(0,y-m):min(img_rgb.shape[0],y+h+m), max(0,x-m):min(img_rgb.shape[1],x+w+m)]
    except: return img_rgb

# ── DATASET BUILDER ───────────────────────────────
def _inner_loader(p_b, l):
    path = p_b.numpy().decode('utf-8')
    img  = cv2.imread(path)
    if img is None: return np.zeros((IMG_SIZE, IMG_SIZE, 7), np.float32), float(l)
    rgb  = crop_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return extract_7ch(rgb, IMG_SIZE), float(l)

def _load_wrapper(path_b, label):
    x, y = tf.py_function(_inner_loader, [path_b, label], [tf.float32, tf.float32])
    x.set_shape([IMG_SIZE, IMG_SIZE, 7])
    y.set_shape([])
    return x, y

def make_mega_dataset():
    df = pd.read_csv(MANIFEST_CSV)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    def prepare_ds(data_df, shuffle=True):
        fake_paths = data_df[data_df.label == 0].filepath.values
        real_paths = data_df[data_df.label == 1].filepath.values
        ds_fake = tf.data.Dataset.from_tensor_slices((fake_paths, [0.0]*len(fake_paths)))
        ds_real = tf.data.Dataset.from_tensor_slices((real_paths, [1.0]*len(real_paths)))
        if shuffle:
            ds_fake = ds_fake.shuffle(5000).repeat()
            ds_real = ds_real.shuffle(5000).repeat()
        ds = tf.data.Dataset.sample_from_datasets([ds_fake, ds_real], weights=[0.5, 0.5])
        ds = ds.map(_load_wrapper, num_parallel_calls=2)
        return ds.batch(BATCH_SIZE).prefetch(1)

    return prepare_ds(train_df), prepare_ds(val_df, shuffle=False), len(train_df), len(val_df)

# ── TRAINING ───────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, n_t, n_v = make_mega_dataset()
    print(f"📊 Mega-Train: {n_t:,} images | Target: Xception")

    model_path = f"{SAVE_DIR}/Xcep_7ch_Calibrated.keras"
    model = tf.keras.models.load_model(model_path, compile=False)

    # FIX 3: Freeze bottom 50% (more frozen = fewer gradients = less VRAM fragmentation)
    total = len(model.layers)
    for i, layer in enumerate(model.layers):
        layer.trainable = (i > total * 0.5) and not isinstance(layer, tf.keras.layers.BatchNormalization)

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"🔓 Trainable: {trainable}/{total} layers (top 50% only)")

    # Mixed precision requires loss scaling
    opt = tf.keras.optimizers.Adam(1e-5)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    save_path = f"{SAVE_DIR}/Xcep_7ch_Mega.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds,
              steps_per_epoch=STEPS_EPOCH, validation_steps=VAL_STEPS,
              epochs=EPOCHS, callbacks=callbacks)
    print(f"\n✅ Saved best model to: {save_path}")
