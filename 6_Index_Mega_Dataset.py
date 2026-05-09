#!/usr/bin/env python3
"""
Indexer v4: ULTRA FAST ⚡
Only runs face filtering on the 'Awesome-Nano-Banana' folder.
Trusts 'pool_A' and 'mega_dataset' as they are already facial.
"""
import os
import cv2
import pandas as pd
from tqdm import tqdm

BASE_DIR  = "/mnt/x/College_Docs/PBL"
SCAN_DIRS = [
    f"{BASE_DIR}/mega_dataset",
    f"{BASE_DIR}/data/pool_A",
    f"{BASE_DIR}/dataset/Awesome-Nano-Banana-images-main"
]
OUTPUT_CSV = f"{BASE_DIR}/mega_dataset_manifest.csv"
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def has_face(img_path):
    img = cv2.imread(img_path)
    if img is None: return False
    small = cv2.resize(img, (128, 128))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    return len(faces) > 0

def index_all():
    raw_data = []
    for scan_root in SCAN_DIRS:
        if not os.path.exists(scan_root):
            print(f"⏩ Skipping {scan_root}")
            continue
        
        is_nano = "awesome-nano-banana" in scan_root.lower()
        print(f"🔍 Indexing: {scan_root} {'(with face filter 🔬)' if is_nano else '(fast scan ⚡)'}")
        
        current_count = 0
        for root, dirs, files in os.walk(scan_root):
            path_lower = root.lower()
            
            # Label logic
            label = None
            if 'fake' in path_lower or 'fakes' in path_lower: label = 0
            elif 'real' in path_lower or 'reals' in path_lower: label = 1
            elif is_nano: label = None # Handle inside file loop
            else: continue

            for f in files:
                if f.lower().endswith(VALID_EXTS):
                    f_lower = f.lower()
                    file_label = label
                    
                    if is_nano:
                        if 'output' in f_lower: file_label = 0
                        elif 'input' in f_lower: file_label = 1
                        else: continue
                        
                        # ONLY FILTER NANO-BANANA IMAGES
                        if not has_face(os.path.join(root, f)):
                            continue
                    
                    raw_data.append({
                        'filepath': os.path.join(root, f),
                        'label':    file_label,
                        'source':   scan_root.split('/')[-1]
                    })
                    current_count += 1
        print(f"   ✅ Added {current_count:,} images")

    df = pd.DataFrame(raw_data)
    print(f"\n🚀 FINAL TOTAL: {len(df):,} images")
    print(df.groupby(['source','label'])['filepath'].count())
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Manifest saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    index_all()
