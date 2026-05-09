import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_prep

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIND_TEST_DIR = os.path.join(BASE_DIR, "dataset", "BLIND_TEST_SET")

# The 53k Goldmine: we used 10,000 for training, ~43,000 are still unseen.
# We skip the first 10,000 images (training window) and take the next 2,000.
FAKE_SOURCE = os.path.join(BASE_DIR, "data", "images", "fake")
FAKE_LIMIT = 2000

REAL_POOLS = [
    os.path.join(BASE_DIR, "data", "pool_A", "real"),
    os.path.join(BASE_DIR, "data", "pool_B", "real"),
    os.path.join(BASE_DIR, "data", "pool_C", "real")
]
REAL_LIMIT = 2000

def scan_blind_global(src_dir, out_dir, label, limit, global_skip=10000):
    """
    Collects images across the entire folder tree, skipping the first
    global_skip images globally (to avoid overlap with training data).
    """
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    count = 0
    skipped = 0

    for root, _, files in os.walk(src_dir):
        if any(p.startswith('.') or p.startswith('__') for p in root.split(os.sep)):
            continue

        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in img_files:
            if count >= limit:
                return count
            # Skip the first global_skip images across ALL folders
            if skipped < global_skip:
                skipped += 1
                continue
            if core_prep.process_single_image(os.path.join(root, f), label_dir):
                count += 1
                if count % 100 == 0:
                    print(f"  Progress: {count}/{limit} {label}...", end='\r')
    return count

if __name__ == "__main__":
    print("=" * 55)
    print("BLIND TEST SET BUILDER")
    print("Source: Unseen FLUX/SDXL images (post-training window)")
    print("=" * 55)

    os.makedirs(os.path.join(BLIND_TEST_DIR, "fake"), exist_ok=True)
    os.makedirs(os.path.join(BLIND_TEST_DIR, "real"), exist_ok=True)

    # 1. Blind fakes from the goldmine (skip first 10k used in training)
    print(f"\nCollecting {FAKE_LIMIT} unseen FAKE faces from the Goldmine...")
    total_fakes = scan_blind_global(FAKE_SOURCE, BLIND_TEST_DIR, "fake", FAKE_LIMIT, global_skip=10000)
    print(f"\nFakes collected: {total_fakes}")

    # 2. Blind reals (skip first 30k used in training)
    print(f"\nCollecting {REAL_LIMIT} unseen REAL faces...")
    total_reals = 0
    for pool in REAL_POOLS:
        if not os.path.exists(pool): continue
        if total_reals >= REAL_LIMIT: break
        remaining = REAL_LIMIT - total_reals
        count = scan_blind_global(pool, BLIND_TEST_DIR, "real", remaining, global_skip=30000)
        total_reals += count
        print(f"\n  Pool done. Running total reals: {total_reals}")

    print("\n" + "=" * 55)
    print("BLIND TEST SET READY")
    print(f"  Total Fakes : {total_fakes}")
    print(f"  Total Reals : {total_reals}")
    print(f"  Location    : {BLIND_TEST_DIR}")
    print("=" * 55)
    print("\nNext step: python3 13_Evaluate_Blind.py")

