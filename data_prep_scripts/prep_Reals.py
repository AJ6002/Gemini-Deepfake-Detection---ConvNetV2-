import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_prep

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_POOLS = [
    os.path.join(BASE_DIR, "data", "pool_A", "real"),
    os.path.join(BASE_DIR, "data", "pool_B", "real"),
    os.path.join(BASE_DIR, "data", "pool_C", "real"),
    os.path.join(BASE_DIR, "data", "images", "real")
]
OUTPUT_ROOT = os.path.join(BASE_DIR, "dataset", "ULTRA_SWIN_DATA")
TOTAL_TARGET = 30000 

def run():
    print(f"⚖️ Balancing Dataset with Real Faces...")
    
    current_real = 0
    for pool in REAL_POOLS:
        if not os.path.exists(pool): continue
        if current_real >= TOTAL_TARGET: break
        
        remaining = TOTAL_TARGET - current_real
        # Set max_per_folder high (30,000) to ensure we get all images from each pool
        count = core_prep.scan_directory(pool, OUTPUT_ROOT, "real", remaining, max_per_folder=30000)
        current_real += count
        print(f"  Pool complete. Total reals: {current_real}")

    print(f"\n✅ SUCCESS: Dataset balanced with {current_real} Real faces.")

if __name__ == "__main__":
    run()
