import os
import tarfile
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
TAR_PATH    = "/mnt/x/College_Docs/PBL/dataset/AIGI-Now.tar.gz"
EXTRACT_DIR = "/mnt/x/College_Docs/PBL/mega_dataset/AIGI_Modern"
VALID_EXTS  = ('.jpg', '.jpeg', '.png')

# Haar Cascade for face filtering
_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_face(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    # Quick downscale for speed
    small = cv2.resize(img, (128, 128))
    faces = _cascade.detectMultiScale(small, 1.1, 3)
    return len(faces) > 0

def smart_extract():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
        os.makedirs(os.path.join(EXTRACT_DIR, "fake"))
        os.makedirs(os.path.join(EXTRACT_DIR, "real"))

    print(f"📦 Opening {TAR_PATH}...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(VALID_EXTS)]
        print(f"🔍 Found {len(members):,} images. Filtering for faces...")
        
        count = 0
        for m in tqdm(members):
            # Read image directly from memory to avoid disk thrashing
            f = tar.extractfile(m)
            if f is None: continue
            content = f.read()
            
            if is_face(content):
                # LABELING LOGIC: 
                # AIGI-Now usually has specific folders or names for fakes
                # We will check if 'fake' or generator names are in the path
                path_lower = m.name.lower()
                is_fake = any(x in path_lower for x in ['nano', 'gpt', 'kling', 'minimax', 'jimeng', 'fake'])
                
                subfolder = "fake" if is_fake else "real"
                filename = os.path.basename(m.name)
                save_path = os.path.join(EXTRACT_DIR, subfolder, filename)
                
                with open(save_path, "wb") as out:
                    out.write(content)
                count += 1
                
    print(f"\n✅ Done! Extracted {count:,} faces to {EXTRACT_DIR}")

if __name__ == "__main__":
    if os.path.exists(TAR_PATH):
        smart_extract()
    else:
        print(f"❌ File not found at {TAR_PATH}")
