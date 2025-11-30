import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config import MODELS_DIR  # import paths from config.py

OBJECTIVES = ["Objective_1", "Objective_2", "Objective_3", "Objective_4"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for obj in OBJECTIVES:
    print(f"\n🚀 Re‑exporting {obj} to ONNX (opset 13)...")

    # Load checkpoint from MODELS_DIR
    ckpt_path = os.path.join(MODELS_DIR, f"{obj}_model.pth")
    state_dict = torch.load(ckpt_path, map_location=device)

    # Auto-detect number of classes from checkpoint
    num_classes = state_dict["fc.weight"].shape[0]

    # Build model with correct output size
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Export to ONNX inside MODELS_DIR
    onnx_path = os.path.join(MODELS_DIR, f"{obj}_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
        do_constant_folding=True
    )

    print(f"✅ {obj} exported to {onnx_path} (opset 13)")