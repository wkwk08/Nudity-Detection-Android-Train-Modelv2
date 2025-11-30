import csv
import os
from pathlib import Path
from collections import defaultdict

# Import paths from config.py
from config import OUTPUT_DIR, LOGS_DIR

def count_images(path: Path):
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"])

def validate_and_export():
    summary = []
    totals = defaultdict(int)
    grand_total = 0

    for objective in OUTPUT_DIR.iterdir():
        if not objective.is_dir():
            continue
        for split in objective.iterdir():
            if not split.is_dir():
                continue
            for dataset in split.iterdir():
                if not dataset.is_dir():
                    continue
                for cls in dataset.iterdir():
                    if not cls.is_dir():
                        continue
                    count = count_images(cls)
                    summary.append({
                        "Objective": objective.name,
                        "Dataset": dataset.name,
                        "Split": split.name,
                        "Class": cls.name,
                        "Count": count
                    })
                    totals[objective.name] += count
                    grand_total += count

    # Print summary
    for row in summary:
        print(f"{row['Objective']} | {row['Dataset']} | {row['Split']} | {row['Class']} → {row['Count']} files")

    # Print totals per objective
    print("\n=== Totals per Objective ===")
    for obj, total in totals.items():
        print(f"{obj}: {total} files")
    print(f"Grand Total: {grand_total} files")

    # Export detailed summary into LOGS_DIR
    csv_path = os.path.join(LOGS_DIR, "validation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Objective", "Dataset", "Split", "Class", "Count"])
        writer.writeheader()
        writer.writerows(summary)

    # Export totals into LOGS_DIR
    totals_path = os.path.join(LOGS_DIR, "validation_totals.csv")
    with open(totals_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Objective", "TotalCount"])
        for obj, total in totals.items():
            writer.writerow([obj, total])
        writer.writerow(["GrandTotal", grand_total])

    print(f"\n✅ Validation summary exported to: {csv_path}")
    print(f"✅ Totals per objective + grand total exported to: {totals_path}")

if __name__ == "__main__":
    validate_and_export()