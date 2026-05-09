import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

# --- WSL GPU FIX ---
cuda_path = "/usr/lib/wsl/lib"
if os.path.exists(cuda_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from logic import DeepfakeDetector, extract_7ch

# --- CONFIG ---
BLIND_TEST_DIR = os.path.join(BASE_DIR, "dataset", "BLIND_TEST_SET")
REPORT_PATH = os.path.join(BASE_DIR, "blind_test_report.txt")
BATCH_SIZE = 16 

def run_fast_evaluation():
    print("=" * 60)
    print("🚀 BATCH-OPTIMIZED BLIND EVALUATION (Phase 4 V2)")
    print("=" * 60)

    models_dir = os.path.join(BASE_DIR, "models")
    detector = DeepfakeDetector(models_dir)

    fake_dir = os.path.join(BLIND_TEST_DIR, "fake")
    real_dir = os.path.join(BLIND_TEST_DIR, "real")

    # Get all file paths
    all_samples = []
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if f.lower().endswith(('.jpg', '.png')):
                all_samples.append((os.path.join(fake_dir, f), 0)) # 0 = FAKE
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.jpg', '.png')):
                all_samples.append((os.path.join(real_dir, f), 1)) # 1 = REAL

    if not all_samples:
        print("❌ Error: No samples found in BLIND_TEST_SET. Run prep_BlindTest.py first.")
        return

    print(f"Total samples to evaluate: {len(all_samples)}")
    print(f"Using Batch Size: {BATCH_SIZE}")

    TP, FP, TN, FN = 0, 0, 0, 0

    # Process in Batches
    for i in tqdm(range(0, len(all_samples), BATCH_SIZE)):
        batch_data = all_samples[i:i+BATCH_SIZE]
        batch_imgs = []
        batch_labels = []
        
        for path, label in batch_data:
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Use extract_7ch from logic.py
            img_7ch = extract_7ch(img, 256)
            batch_imgs.append(img_7ch)
            batch_labels.append(label)
        
        if not batch_imgs: continue
        
        batch_x = np.array(batch_imgs).astype('float32')
        
        # We only evaluate the "Dictator" model (ConvNeXt V2) for the fast report
        # as it's the leader of the ensemble.
        preds = detector.model_convnext.predict(batch_x, batch_size=BATCH_SIZE, verbose=0)
        
        for pred, true_label in zip(preds, batch_labels):
            p = pred[0]
            # Prob < 0.5 = FAKE (0), Prob >= 0.5 = REAL (1)
            predicted_label = 1 if p >= 0.5 else 0
            
            if true_label == 0: # Actual FAKE
                if predicted_label == 0: TP += 1
                else: FN += 1
            else: # Actual REAL
                if predicted_label == 1: TN += 1
                else: FP += 1

    # --- FINAL METRICS ---
    total = TP + FP + TN + FN
    acc = (TP + TN) / total * 100
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0

    report = f"""
============================================================
FINAL SCIENTIFIC EVALUATION: ConvNeXt-Diff-7Ch V2
============================================================
Total Images Tested : {total} (Unseen Blind Set)

--- Results ---
Accuracy  : {acc:.2f}%
Precision : {precision:.2f}% (How reliable were the 'Fake' alerts)
Recall    : {recall:.2f}% (Percentage of fakes successfully caught)

--- Confusion Matrix ---
True Positives (Caught Fakes) : {TP}
True Negatives (Correct Reals): {TN}
False Positives (False Alarms): {FP}
False Negatives (Missed Fakes): {FN}
============================================================
"""
    print(report)
    with open(REPORT_PATH, "w") as f: f.write(report)
    print(f"✅ Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_fast_evaluation()
