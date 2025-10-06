import os, glob, shutil, random, yaml
from collections import Counter
import numpy as np

DATASET_DIR = "./dataset"
OUTPUT_DIR = DATASET_DIR
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

random.seed(SEED)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Розбиваємо датасет на test i train, оскільки у нас був тільки train на початку
images = glob.glob(os.path.join(DATASET_DIR, "images/*.jpg"))
random.shuffle(images)

n = len(images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_imgs = images[:n_train]
val_imgs = images[n_train:n_train+n_val]
test_imgs = images[n_train+n_val:]

# перевіряємо тут цілісність даних перед тим як копіювати
# (щоб для прикладу у нас не були об'єкти поза межами картинки)
def copy_split(img_list, split_name):
    for img_path in img_list:
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(DATASET_DIR, "labels", fname.replace(".jpg", ".txt"))
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, split_name, "images", fname))
        if os.path.exists(lbl_path):
            # Перевірка bbox
            valid_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = parts
                    x, y, w, h = map(float, (x, y, w, h))
                    if w <= 0 or h <= 0:
                        continue
                    if not (0 <= x-w/2 <= 1 and 0 <= y-h/2 <=1 and 0 <= x+w/2 <=1 and 0 <= y+h/2 <=1):
                        continue  # видаляємо out-of-bounds
                    valid_lines.append(f"{cls} {x} {y} {w} {h}\n")
            # Запис коректних bbox у нову папку
            with open(os.path.join(OUTPUT_DIR, split_name, "labels", fname.replace(".jpg",".txt")), "w") as f:
                f.writelines(valid_lines)

copy_split(train_imgs, "train")
copy_split(val_imgs, "val")
copy_split(test_imgs, "test")

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Нам ймовірно потрібна розбивка для class weight оскільки дані класів досить
# незбалансовані
label_files = glob.glob(os.path.join(OUTPUT_DIR, "train", "labels/*.txt"))
classes_set = set()
class_counts = Counter()

for lbl in label_files:
    with open(lbl) as f:
        for line in f:
            cls = int(line.strip().split()[0])
            classes_set.add(cls)
            class_counts[cls] += 1

classes_list = sorted(list(classes_set))
print("Classes detected:", classes_list)

class_names = [f"class{i}" for i in classes_list]

total = sum(class_counts.values())
class_weights = [total / (len(classes_list) * class_counts.get(i,1)) for i in classes_list]
print("Class weights:", class_weights)

# створюємо dataset.yaml оскільки він буде нам потрібний згодом
data_yaml = {
    'path': OUTPUT_DIR,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(classes_list),
    'names': class_names
}

with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("dataset.yaml created!")
