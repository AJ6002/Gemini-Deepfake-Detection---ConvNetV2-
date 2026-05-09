#!/usr/bin/env python3
"""
GEMINI OVERNIGHT TRAINING — 7-Channel Wavelet Ensemble
Models: EfficientNet → ResNet50 → Xception
Est. Time: 5.5-6.5 hours on RTX 3050 6GB
"""
import os, sys, time, shutil
import numpy as np
import cv2, pywt
import tensorflow as tf
from datetime import datetime, timedelta

# ── SUPPRESS LOGS ────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── GPU SETUP ────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU Ready: {gpus[0]}")
else:
    print("⚠️  WARNING: No GPU. Training will be VERY slow.")

# ── CONFIG ───────────────────────────────────────────────────
TRAIN_DIR   = 'data/pool_A'
VAL_DIR     = 'data/validation'
BACKUP_DIR  = 'models/backup'
SAVE_DIR    = 'models/hybrid_7ch'
IMG_SIZE    = 256
BATCH_SIZE  = 16
EPOCHS      = 20
STEPS_EPOCH = 625
VAL_STEPS   = 80
LR          = 1e-4   # Default (overridden per model below)

# ── PREPROCESSING (Synced exactly with logic.py) ─────────────
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_rgb):
    try:
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = _cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            h, w  = img_rgb.shape[:2]
            side  = min(h, w)
            x, y  = (w-side)//2, (h-side)//2
            return img_rgb[y:y+side, x:x+side]
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

# ── DATASET BUILDER ──────────────────────────────────────────
def build_file_list(directory, max_per_class=None):
    files = []
    for cls, label in [('fake', 0), ('real', 1)]:
        path = os.path.join(directory, cls)
        if not os.path.exists(path):
            print(f"  ⚠️  Missing: {path}")
            continue
        imgs = [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if max_per_class:
            imgs = imgs[:max_per_class]
        files += [(p, label) for p in imgs]
    np.random.shuffle(files)
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

# ── UTILITIES ────────────────────────────────────────────────
def freeze_bn(model, unfreeze_ratio=0.7):
    """Freeze early layers, keep top `unfreeze_ratio` trainable.
    Always freezes BatchNorm to prevent training instability."""
    total        = len(model.layers)
    freeze_until = int(total * (1 - unfreeze_ratio))
    for i, layer in enumerate(model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = not isinstance(
                layer, tf.keras.layers.BatchNormalization)
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  🔓 Trainable layers: {trainable}/{total}")

def get_callbacks(save_name):
    save_path = os.path.join(SAVE_DIR, f'{save_name}_Calibrated.keras')
    return [
        tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1),
    ]

def backup(src):
    if os.path.exists(src):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        shutil.copy2(src, os.path.join(BACKUP_DIR, os.path.basename(src)))
        print(f"  📦 Backed up: {os.path.basename(src)}")

# ── TRAIN ONE MODEL ──────────────────────────────────────────
def train_model(name, model_path, save_name, train_ds, val_ds, n,
                lr=1e-4, unfreeze_ratio=0.7):
    print(f"\n{'='*60}")
    print(f"  TRAINING: {name}")
    print(f"  Dataset : {n:,} images | Epochs: {EPOCHS} | Steps: {STEPS_EPOCH}")
    print(f"  LR      : {lr}  |  Unfreezing: top {int(unfreeze_ratio*100)}%")
    print(f"{'='*60}")

    backup(model_path)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"  ✅ Loaded {name}")
    except Exception as e:
        print(f"  ❌ Could not load {name}: {e}")
        return None

    freeze_bn(model, unfreeze_ratio=unfreeze_ratio)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_EPOCH,
        validation_steps=VAL_STEPS,
        epochs=EPOCHS,
        callbacks=get_callbacks(save_name),
        verbose=1)

    elapsed  = time.time() - t0
    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n  ✅ {name} DONE | Best val_acc: {best_acc*100:.2f}%"
          f" | Time: {timedelta(seconds=int(elapsed))}")
    return best_acc

# ── MAIN ─────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("\n" + "="*60)
    print("  GEMINI OVERNIGHT TRAINING — 7-CH WAVELET ENSEMBLE")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\n[SETUP] Building datasets...")
    train_ds, n_train = make_dataset(TRAIN_DIR, BATCH_SIZE)
    val_ds,   n_val   = make_dataset(VAL_DIR,   BATCH_SIZE, max_per_class=15000)
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")

    models_to_train = [
        # ResNet50: Continue from 82.66% checkpoint → target 87%+
        ('ResNet50', f'{SAVE_DIR}/Res_7ch_Calibrated.keras', 'Res_7ch',
         1e-5, 0.7),
        # Xception: SKIP — already at 94.61%, no retraining needed
        # ('Xception', f'{SAVE_DIR}/Xcep_7ch_Calibrated.keras', 'Xcep_7ch', 1e-5, 0.5),
        # MobileNetV2: SKIP — already at 90.08%, use build_mobilenet_7ch.py for retraining
        # ('MobileNetV2', f'{SAVE_DIR}/MobileNet_7ch_Calibrated.keras', 'MobileNet_7ch', 1e-5, 0.5),
    ]

    results = {}
    for name, path, save_name, lr, unfreeze in models_to_train:
        results[name] = train_model(
            name, path, save_name, train_ds, val_ds, n_train,
            lr=lr, unfreeze_ratio=unfreeze)

    # ── FINAL SUMMARY ──
    total = timedelta(seconds=int(time.time()-t_total))
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    for name, acc in results.items():
        if acc is None:
            print(f"  {name:15s}: ❌ FAILED TO LOAD")
        else:
            tag = "✅ PASS" if acc > 0.85 else "⚠️  BELOW 85%"
            print(f"  {name:15s}: {acc*100:.2f}%  {tag}")
    print(f"\n  Total Time : {total}")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n  ➡️  Next: Update logic.py weights, then run app.py\n")

if __name__ == '__main__':
    main()
