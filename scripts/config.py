from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))
DATASETS_DIR = Path(os.getenv("DATASETS_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
MODELS_DIR = Path(os.getenv("MODELS_DIR"))
LOGS_DIR = Path(os.getenv("LOGS_DIR"))

# Debug: review loaded values
print("BASE_DIR:", BASE_DIR)
print("DATASETS_DIR:", DATASETS_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("LOGS_DIR:", LOGS_DIR)