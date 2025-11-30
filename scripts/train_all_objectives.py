import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from PIL import ImageFile

# Import paths from config.py
from config import OUTPUT_DIR, MODELS_DIR, LOGS_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

OBJECTIVES = ["Objective_1", "Objective_2", "Objective_3", "Objective_4"]

# Image transformations
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for obj in OBJECTIVES:
    print(f"\n🚀 Training {obj}...\n")
    DATA_DIR = os.path.join(OUTPUT_DIR, obj)

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform["train"])
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform["val"])
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform["test"])

    print(f"📊 {obj} dataset sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model (ResNet18 transfer learning)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop with logging
    log_file = os.path.join(LOGS_DIR, f"{obj}_training_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "ValAcc", "TestAcc"])

        for epoch in range(EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_acc = correct / total
            avg_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total

            # Test evaluation
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
            test_acc = test_correct / test_total

            print(f"{obj} Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            writer.writerow([epoch+1, avg_loss, train_acc, val_acc, test_acc])

    # Save model checkpoint
    model_path = os.path.join(MODELS_DIR, f"{obj}_model.pth")
    torch.save(model.state_dict(), model_path)

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = os.path.join(MODELS_DIR, f"{obj}_model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

    print(f"✅ {obj} model trained on {len(train_dataset)} images, validated on {len(val_dataset)}, tested on {len(test_dataset)}. "
          f"Metrics logged to {log_file}, and exported to {model_path} / {onnx_path}")