
import os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#cfg
DATASET_DIR = "./dataset"     # root
OUTPUT_DIR = "./eda_results"  # out
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

os.makedirs(OUTPUT_DIR, exist_ok=True)




def parse_yolo_file(path, split=None):
    rows = []
    img_name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            rows.append({
                "split": split,
                "label_path": path,
                "image_name": img_name,
                "class": cls,
                "x": x, "y": y, "w": w, "h": h
            })
    return rows

def load_annotations(dataset_dir):
    annotations = []
    label_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(label_dir):
        raise RuntimeError(f"dir not found: {label_dir}")
    for lf in glob.glob(os.path.join(label_dir, "*.txt")):
        annotations.extend(parse_yolo_file(lf, split="all"))
    return pd.DataFrame(annotations)

df = load_annotations(DATASET_DIR)
if df.empty:
    raise RuntimeError("no lables")




df["area"] = df["w"] * df["h"]
df["aspect"] = df["w"] / df["h"].replace(0, np.nan)
df["x_min"] = df["x"] - df["w"]/2
df["y_min"] = df["y"] - df["h"]/2
df["x_max"] = df["x"] + df["w"]/2
df["y_max"] = df["y"] + df["h"]/2
df["oob"] = ((df[["x_min","y_min","x_max","y_max"]] < 0).any(axis=1) |
             (df[["x_min","y_min","x_max","y_max"]] > 1).any(axis=1))

report = {
    "total_annotations": int(len(df)),
    "unique_images": int(df["image_name"].nunique()),
    "per_class_counts": df["class"].value_counts().sort_index().to_dict(),
    "per_split_counts": df["split"].value_counts().to_dict(),
    "area_stats": df["area"].describe().to_dict(),
    "oob_count": int(df["oob"].sum())
}
with open(os.path.join(OUTPUT_DIR, "eda_report.json"), "w") as f:
    json.dump(report, f, indent=2)

#visual
def save_plot(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

#class dist
fig, ax = plt.subplots(figsize=(8,5))
counts = df["class"].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values)
ax.set_title("Розподіл анотацій за класами")
ax.set_xlabel("class id"); ax.set_ylabel("count")
save_plot(fig, "class_counts.png")

#area hist
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(df["area"], bins=40)
ax.set_title("Гістограма площі (w*h)")
ax.set_xlabel("area (normalized)"); ax.set_ylabel("frequency")
save_plot(fig, "area_hist.png")

#scatter w h
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(df["w"], df["h"], alpha=0.4)
ax.set_title("Ширина vs Висота")
ax.set_xlabel("w (normalized)"); ax.set_ylabel("h (normalized)")
save_plot(fig, "w_vs_h.png")

#box centr
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(df["x"], df["y"], alpha=0.3)
ax.set_title("Центри боксів")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel("x_center"); ax.set_ylabel("y_center")
save_plot(fig, "centers.png")

#boxplot
fig, ax = plt.subplots(figsize=(10,6))
grouped = [g["area"].values for _, g in df.groupby("class")]
labels = [str(c) for c in df["class"].unique()]
ax.boxplot(grouped, labels=labels)
ax.set_title("Розподіл площі боксів по класах")
ax.set_xlabel("class id"); ax.set_ylabel("area")
save_plot(fig, "area_boxplot_per_class.png")

print("\nEDA eda finished:", OUTPUT_DIR)
