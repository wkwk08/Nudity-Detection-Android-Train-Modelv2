import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from PIL import ImageFile
import csv

# Import paths from config.py
from config import OUTPUT_DIR, LOGS_DIR

# === FIX: allow truncated/corrupt JPEGs to load ===
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === CONFIG ===
BASE_DIR = OUTPUT_DIR   # balanced_output path from .env
OBJECTIVES = ["Objective_1", "Objective_2", "Objective_3", "Objective_4"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === BASELINE (Optimized Emon RGB/HSV thresholds) ===
def emon_baseline(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (r > 95) & (g > 40) & (b > 20) &
        (r > g) & (r > b) & (np.abs(r - g) > 15) &
        (h >= 0) & (h <= 50) &
        (s / 255.0 >= 0.23) & (s / 255.0 <= 0.78)
    )
    ratio = np.sum(mask) / mask.size
    return 1 if ratio > 0.38 else 0

# === EXPORT CONFUSION MATRIX ===
def export_confusion_matrix(y_true, y_pred, objective_name, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in class_names],
                             columns=[f"Pred_{c}" for c in class_names])
    cm_path = os.path.join(LOGS_DIR, f"{objective_name}_confusion.csv")
    cm_df.to_csv(cm_path, encoding="utf-8", index=True)
    print(f"📁 Saved confusion matrix for {objective_name} → {cm_path}")

# === ML MODEL EVALUATION ===
def evaluate_model(obj):
    print(f"\n📊 Evaluating {obj}...\n")
    DATA_DIR = os.path.join(BASE_DIR, obj)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dataset size for {obj}: {len(test_dataset)} images")

    num_classes = len(test_dataset.classes)
    model = resnet18(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(f"{obj}_model.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred_ml, y_pred_baseline = [], [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Evaluating {obj}")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred_ml.extend(preds.cpu().numpy())

            batch_start = batch_idx * test_loader.batch_size
            batch_end = batch_start + len(labels)
            batch_paths = [test_loader.dataset.samples[i][0] for i in range(batch_start, batch_end)]
            for img_path in batch_paths:
                baseline_pred = emon_baseline(img_path)
                y_pred_baseline.append(baseline_pred)

    # ML metrics
    acc_ml = accuracy_score(y_true, y_pred_ml)
    prec_ml = precision_score(y_true, y_pred_ml, average="macro", zero_division=0)
    rec_ml = recall_score(y_true, y_pred_ml, average="macro", zero_division=0)
    f1_ml = f1_score(y_true, y_pred_ml, average="macro", zero_division=0)

    # Baseline metrics
    acc_base = accuracy_score(y_true, y_pred_baseline)
    prec_base = precision_score(y_true, y_pred_baseline, average="macro", zero_division=0)
    rec_base = recall_score(y_true, y_pred_baseline, average="macro", zero_division=0)
    f1_base = f1_score(y_true, y_pred_baseline, average="macro", zero_division=0)

    print(f"ML Model ({obj}): Acc={acc_ml:.4f}, Prec={prec_ml:.4f}, Rec={rec_ml:.4f}, F1={f1_ml:.4f}")
    print(f"Baseline (Emon): Acc={acc_base:.4f}, Prec={prec_base:.4f}, Rec={rec_base:.4f}, F1={f1_base:.4f}")

    # Export confusion matrix
    export_confusion_matrix(y_true, y_pred_ml, obj, test_dataset.classes)

    return {
        "DatasetSize": len(test_dataset),
        "ML": (acc_ml, prec_ml, rec_ml, f1_ml),
        "Baseline": (acc_base, prec_base, rec_base, f1_base)
    }

# === MAIN LOOP ===
results = {}
for obj in OBJECTIVES:
    results[obj] = evaluate_model(obj)

# === FINAL COMPARISON TABLE + CSV EXPORT ===
print("\n✅ Final Comparison Table:")
print("Objective | DatasetSize | ML_Acc | ML_Prec | ML_Rec | ML_F1 || Base_Acc | Base_Prec | Base_Rec | Base_F1")

csv_rows = [["Objective","DatasetSize","ML_Acc","ML_Prec","ML_Rec","ML_F1","Base_Acc","Base_Prec","Base_Rec","Base_F1"]]

for obj, metrics in results.items():
    ml, base = metrics["ML"], metrics["Baseline"]
    dataset_size = metrics["DatasetSize"]
    print(f"{obj:10} | {dataset_size:11} | {ml[0]:.3f} | {ml[1]:.3f} | {ml[2]:.3f} | {ml[3]:.3f} || "
          f"{base[0]:.3f} | {base[1]:.3f} | {base[2]:.3f} | {base[3]:.3f}")
    csv_rows.append([
        obj, dataset_size,
        round(ml[0],3), round(ml[1],3), round(ml[2],3), round(ml[3],3),
        round(base[0],3), round(base[1],3), round(base[2],3), round(base[3],3)
    ])

eval_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
with open(eval_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"\n📁 Results saved to {eval_path}")