import os
import onnx
import tensorflow as tf
import numpy as np
from onnx_tf.backend import prepare
from config import MODELS_DIR  # import from config.py

# Define per-objective class labels (adjust to match training)
CLASS_LABELS = {
    "Objective_1_model": ["safe", "nude"],          # 2 classes
    "Objective_2_model": ["safe", "nude", "sexual"],# 3 classes
    "Objective_3_model": ["safe", "other"],         # 2 classes
    "Objective_4_model": ["safe", "nude", "sexual", "other"] # 4 classes
}

# Automatically find all .onnx files in MODELS_DIR
onnx_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")]

if not onnx_files:
    print("❌ No ONNX files found in", MODELS_DIR)
else:
    for onnx_file in onnx_files:
        model_name = os.path.splitext(onnx_file)[0]
        onnx_path = os.path.join(MODELS_DIR, onnx_file)
        tf_dir = os.path.join(MODELS_DIR, f"{model_name}_tf_model")
        tflite_path = os.path.join(MODELS_DIR, f"{model_name}.tflite")

        print(f"\n🚀 Converting {model_name}...")

        try:
            # Step 1: Load ONNX model
            onnx_model = onnx.load(onnx_path)

            # Step 2: Convert ONNX → TensorFlow SavedModel
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_dir)

            # Step 3: Convert TensorFlow → TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
            tflite_model = converter.convert()

            # Save TFLite model
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            print(f"✅ {model_name} converted to {tflite_path}")

            # Step 4: Dummy inference check
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            dummy_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            print(f"🔍 {model_name} dummy inference output shape: {output_data.shape}")

            if model_name in CLASS_LABELS:
                labels = CLASS_LABELS[model_name]
                if output_data.ndim == 2 and output_data.shape[1] == len(labels):
                    predicted_idx = int(np.argmax(output_data))
                    print(f"📌 Predicted class: {labels[predicted_idx]}")

        except Exception as e:
            print(f"❌ Conversion failed for {model_name}: {e}")