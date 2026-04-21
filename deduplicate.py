import os
import imagehash
from PIL import Image
from itertools import combinations

dataset_root = "./dataset"
THRESHOLD = 5  # hamming distance for near-duplicates

print("Hashing images...")
hashes = {}  # filepath -> hash
for class_name in sorted(os.listdir(dataset_root)):
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        try:
            hashes[fpath] = imagehash.dhash(Image.open(fpath))
        except Exception:
            print(f"Could not read: {fpath}")

print(f"Hashed {len(hashes)} images.\n")

paths = list(hashes.keys())
deleted = set()
within_class_dupes = []
cross_class_dupes = []

for i, j in combinations(range(len(paths)), 2):
    p1, p2 = paths[i], paths[j]
    if p1 in deleted or p2 in deleted:
        continue
    dist = hashes[p1] - hashes[p2]
    if dist > THRESHOLD:
        continue
    c1 = os.path.basename(os.path.dirname(p1))
    c2 = os.path.basename(os.path.dirname(p2))
    if c1 == c2:
        within_class_dupes.append((p1, p2, dist))
    else:
        cross_class_dupes.append((p1, p2, dist, c1, c2))

# Auto-delete within-class duplicates (keep first)
print("=== Within-class duplicates (auto-deleted) ===")
deleted_count = 0
for p1, p2, dist in within_class_dupes:
    if p2 not in deleted:
        os.remove(p2)
        deleted.add(p2)
        deleted_count += 1
        print(f"DELETED: {p2}  (duplicate of {os.path.basename(p1)}, distance={dist})")

if deleted_count == 0:
    print("None found.")

print(f"\n=== Cross-class duplicates (review manually) — {len(cross_class_dupes)} found ===")
for p1, p2, dist, c1, c2 in cross_class_dupes:
    print(f"WARNING [{c1}] {os.path.basename(p1)}  <->  [{c2}] {os.path.basename(p2)}  (distance={dist})")

if not cross_class_dupes:
    print("None found.")

print(f"\nDone. Deleted {deleted_count} within-class duplicates.")
