#!/usr/bin/env python3
"""
BUILD + TRAIN: MobileNetV2 7-Channel Diffusion Detector
Replaces EfficientNet in the ensemble.

Architecture:
  [256x256x7 Input]
      → [1x1 Conv2D: 7→3 channel projection]  ← learns to combine wavelet channels
      → [MobileNetV2 backbone (ImageNet weights)]  ← intact, no surgery
      → [GlobalAveragePooling2D]
      → [Dense(256, relu) + Dropout(0.4)]
      → [Dense(1, sigmoid)]  ← REAL=1, FAKE=0

Training Dataset:
  - 50k Real + 50k Fake (SD v1.5 + v2.1 diffusion images)
  - Validation: DS0.7 (hardest diffusion fakes)
"""
import os, sys, time, shutil
import numpy as np
import cv2, pywt
import tensorflow as tf
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── GPU SETUP ────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU Ready: {gpus[0]}")
else:
    print("⚠️  No GPU detected.")

# ── CONFIG ───────────────────────────────────────────────────
TRAIN_DIR  = 'data/pool_A'
VAL_DIR    = 'data/validation'
SAVE_PATH  = 'models/hybrid_7ch/MobileNet_7ch_Calibrated.keras'
IMG_SIZE   = 256
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 1e-4

# ── PREPROCESSING (identical to logic.py) ────────────────────
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_rgb):
    try:
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = _cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            h, w  = img_rgb.shape[:2]
            side  = min(h, w)
            return img_rgb[(h-side)//2:(h-side)//2+side,
                           (w-side)//2:(w-side)//2+side]
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        m = int(w * 0.2)
        return img_rgb[max(0,y-m):min(img_rgb.shape[0],y+h+m),
                       max(0,x-m):min(img_rgb.shape[1],x+w+m)]
    except:
        return img_rgb

def extract_7ch(img_rgb, size):
    img  = cv2.resize(img_rgb, (size, size))
    rgb  = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    def nr(b):
        b = cv2.resize(b.astype(np.float32), (size, size))
        lo, hi = b.min(), b.max()
        return (b - lo) / (hi - lo + 1e-7)
    LH, HL, HH = nr(LH), nr(HL), nr(HH)
    HF = nr(np.sqrt(LH**2 + HL**2 + HH**2))
    return np.dstack([rgb, LH[...,None], HL[...,None],
                      HH[...,None], HF[...,None]]).astype(np.float32)

# ── DATASET ──────────────────────────────────────────────────
def build_file_list(directory, max_per_class=None):
    files = []
    for cls, label in [('fake', 0), ('real', 1)]:
        path = os.path.join(directory, cls)
        if not os.path.exists(path):
            continue
        imgs = [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if max_per_class:
            imgs = imgs[:max_per_class]
        files += [(p, label) for p in imgs]
    np.random.shuffle(files)
    print(f"  📊 {sum(1 for f in files if f[1]==0):,} Fake / "
          f"{sum(1 for f in files if f[1]==1):,} Real")
    return files

def _load(path_b, label):
    def _inner(path_b, label):
        path = path_b.numpy().decode('utf-8')
        img  = cv2.imread(path)
        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 7), np.float32), label
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = crop_face(rgb)
        return extract_7ch(rgb, IMG_SIZE), label
    x, y = tf.py_function(_inner, [path_b, label], [tf.float32, tf.float32])
    x.set_shape([IMG_SIZE, IMG_SIZE, 7])
    y.set_shape([])
    return x, y

def make_dataset(directory, batch_size, max_per_class=None):
    files  = build_file_list(directory, max_per_class)
    paths  = [f[0] for f in files]
    labels = [float(f[1]) for f in files]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(min(len(files), 5000))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(files)

# ── BUILD MODEL ──────────────────────────────────────────────
def build_mobilenet_7ch():
    """
    Channel Projection approach — NO surgery.
    A learned 1x1 conv maps 7ch → 3ch before MobileNetV2.
    """
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 7),
                         name='hybrid_7ch_input')

    # Learnable channel projection: 7 → 3
    # This layer learns to extract the most useful combination
    # of RGB + wavelet channels for diffusion detection.
    projected = tf.keras.layers.Conv2D(
        3, kernel_size=1, padding='same',
        activation='linear', name='ch_projection',
        kernel_initializer='glorot_uniform')(inp)

    # Intact MobileNetV2 backbone (ImageNet weights fully preserved)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet')
    backbone.trainable = False  # Freeze backbone initially

    x = backbone(projected, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Deeper Head (Inspired by Kaggle success)
    x = tf.keras.layers.Dense(512, activation='relu', name='fc_512')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_512')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', name='fc_128')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout_2')(x)
    
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)

    model = tf.keras.Model(inputs=inp, outputs=out,
                           name='MobileNetV2_7ch_DiffusionDetector')

    total     = len(model.layers)
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  ✅ Model built: {total} layers total")
    print(f"  🔓 Trainable  : {trainable} (ch_projection + head)")
    print(f"  🔒 Frozen     : {total - trainable} (MobileNetV2 backbone)")

    return model

# ── TRAIN ────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("  MobileNetV2 7-CH DIFFUSION DETECTOR — BUILD + TRAIN")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Build
    print("\n[STEP 1] Building model...")
    model = build_mobilenet_7ch()

    # Datasets
    print("\n[STEP 2] Building datasets...")
    train_ds, n_train = make_dataset(TRAIN_DIR, BATCH_SIZE)
    val_ds,   n_val   = make_dataset(VAL_DIR,   BATCH_SIZE,
                                      max_per_class=15000)
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")

    # Phase 1: Train head only (backbone frozen)
    print("\n[STEP 3] Phase 1 — Training head only (5 epochs)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds,
              steps_per_epoch=625, validation_steps=80,
              epochs=5, callbacks=callbacks, verbose=1)

    # Phase 2: Unfreeze top 40% of MobileNetV2 + fine-tune
    print("\n[STEP 4] Phase 2 — Unfreezing top 40% of backbone...")
    backbone = model.get_layer('mobilenetv2_1.00_224')
    total_bb = len(backbone.layers)
    for i, layer in enumerate(backbone.layers):
        if i < int(total_bb * 0.6):
            layer.trainable = False
        else:
            layer.trainable = not isinstance(
                layer, tf.keras.layers.BatchNormalization)

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  🔓 Trainable layers: {trainable}/{len(model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR * 0.1),  # 1e-5 for fine-tuning
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds,
                        steps_per_epoch=625, validation_steps=80,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)

    best_acc = max(history.history.get('val_accuracy', [0]))
    elapsed  = timedelta(seconds=int(time.time()-t0))

    print("\n" + "="*60)
    print(f"  ✅ MobileNetV2 DONE")
    print(f"  Best val_accuracy : {best_acc*100:.2f}%")
    print(f"  Saved to          : {SAVE_PATH}")
    print(f"  Total time        : {elapsed}")
    print("="*60)
    print("\n  ➡️  Next: Add MobileNet to logic.py ensemble\n")

if __name__ == '__main__':
    main()
