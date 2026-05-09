import os
import tensorflow as tf
import numpy as np
import cv2
from logic import extract_7ch, crop_face

DATA_DIR = 'data/pool_A'
MODEL_PATH = 'models/hybrid_7ch/Eff_7ch_Trainable.keras'
SAVE_PATH = 'models/hybrid_7ch/Eff_7ch_Trained.keras'

def hybrid_generator(directory, batch_size=4):
    all_files = []
    classes = {'fake': 0, 'real': 1}
    for cls, label in classes.items():
        cls_path = os.path.join(directory, cls)
        if not os.path.exists(cls_path): continue
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append((os.path.join(cls_path, img_name), label))
                
    np.random.shuffle(all_files)
    i = 0
    while True:
        bx, by = [], []
        while len(bx) < batch_size:
            if i >= len(all_files): 
                i = 0
                np.random.shuffle(all_files)
            fpath, label = all_files[i]
            i += 1
            img = cv2.imread(fpath)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = crop_face(img_rgb)
                bx.append(extract_7ch(cropped, 256))
                by.append(label)
                
        yield np.array(bx, dtype=np.float32), np.array(by, dtype=np.float32)

def train_efficientnet():
    print("\n   STARTING EFFICIENTNET TRAINING (Pool A)...")
    m = tf.keras.models.load_model(MODEL_PATH)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        SAVE_PATH, 
        monitor='accuracy', save_best_only=True, verbose=1
    )
    
    gen = hybrid_generator(DATA_DIR, batch_size=4)
    m.fit(gen, steps_per_epoch=10, epochs=3, callbacks=[checkpoint])
    
    print(f"\n   EFFICIENTNET SYNC COMPLETE! Saved to {SAVE_PATH}")

if __name__ == '__main__':
    train_efficientnet()
