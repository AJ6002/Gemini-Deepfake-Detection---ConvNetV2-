import os
import cv2
import numpy as np

# --- GLOBAL CONFIG ---
IMG_SIZE = 256
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_single_image(img_path, label_dir):
    """Detects, crops with 30% padding, and saves face as 256x256 square."""
    try:
        img = cv2.imread(img_path)
        if img is None: return False
        
        if img.shape[0] < 150 or img.shape[1] < 150: return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(100, 100))
        
        if len(faces) == 0: return False
        
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        
        pad_w, pad_h = int(w * 0.3), int(h * 0.3)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img.shape[1], x + w + pad_w), min(img.shape[0], y + h + pad_h)
        
        side = max(x2 - x1, y2 - y1)
        if x1 + side > img.shape[1]: x1 = img.shape[1] - side
        if y1 + side > img.shape[0]: y1 = img.shape[0] - side
        x1, y1 = max(0, x1), max(0, y1)

        face = img[y1:y1+side, x1:x1+side]
        if face.size == 0: return False
        
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        out_name = f"{os.path.basename(img_path).split('.')[0]}_{np.random.randint(100000)}.jpg"
        cv2.imwrite(os.path.join(label_dir, out_name), face)
        return True
    except:
        return False

def scan_directory(src_dir, output_root, label, limit, max_per_folder=3):
    """Recursive scan with diversity cap."""
    label_dir = os.path.join(output_root, label)
    os.makedirs(label_dir, exist_ok=True)
    
    current_count = 0
    print(f"🔍 Scanning: {src_dir}")
    
    for root, _, files in os.walk(src_dir):
        if any(p.startswith('.') or p.startswith('__') for p in root.split(os.sep)):
            continue
            
        folder_count = 0
        for f in files:
            if current_count >= limit: return current_count
            if folder_count >= max_per_folder: break 
                
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if process_single_image(os.path.join(root, f), label_dir):
                    current_count += 1
                    folder_count += 1
                    
        if current_count % 50 == 0 and current_count > 0:
            print(f"  Processed: {current_count}/{limit}...", end='\r')
            
    return current_count
