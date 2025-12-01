"""
step1_setup.py

Initial setup script for local fight/violence detection project.

- Assumes project root: D:/violence-detection
- Expects dataset at:  D:/violence-detection/dataset/...
- Creates workspace folders:
    workspace/
      clips/
      detections/
      tracks/
      models/
      inference_outputs/
"""

from pathlib import Path
import os


def print_tree(base_path: Path, indent: str = "", max_items: int = 5):
    """
    Recursively print folder structure:
    - Shows first 'max_items' files/folders in each directory
    - Prints "... +X more" if more items exist
    """
    try:
        items = sorted(os.listdir(base_path))
    except PermissionError:
        print(indent + "❌ [Permission Denied]")
        return
    except FileNotFoundError:
        print(indent + "❌ [Not Found]")
        return

    for i, item in enumerate(items[:max_items]):
        path = base_path / item
        if path.is_dir():
            print(f"{indent}📂 {item}/")
            print_tree(path, indent + "   ", max_items=max_items)
        else:
            print(f"{indent}📄 {item}")
    if len(items) > max_items:
        print(f"{indent}... +{len(items) - max_items} more")


def main():
    # ====== CHANGE THIS if your folder is somewhere else ======
    ROOT = Path("D:/violence-detection").resolve()
    # ==========================================================

    print(f"Project root: {ROOT}")

    if not ROOT.exists():
        raise SystemExit(f"❌ Root folder does not exist: {ROOT}")

    # Dataset root (copied from Colab structure)
    DATASET_ROOT = ROOT / "dataset"
    TRAIN_DIR = DATASET_ROOT / "train"
    VAL_DIR = DATASET_ROOT / "val"

    print("\nChecking dataset structure...")
    print(f"Dataset root exists: {DATASET_ROOT.exists()}")
    print(f" - train/ exists: {TRAIN_DIR.exists()}")
    print(f" - val/ exists:   {VAL_DIR.exists()}")

    # YOLO weights – look for files like yolo11*.pt in ROOT
    yolo_candidates = list(ROOT.glob("yolo11*.pt"))
    if yolo_candidates:
        print("\nFound YOLO weights:")
        for w in yolo_candidates:
            print("  -", w)
    else:
        print("\n⚠ No YOLO weights found in project root. "
              "Place yolo11x.pt or similar in:", ROOT)

    # Workspace folder (similar to /workspace in Colab)
    WORKSPACE = ROOT / "workspace"
    CLIPS_DIR = WORKSPACE / "clips"
    DETECTIONS_DIR = WORKSPACE / "detections"
    TRACKS_DIR = WORKSPACE / "tracks"
    MODELS_DIR = WORKSPACE / "models"
    INFER_OUT_DIR = WORKSPACE / "inference_outputs"

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INFER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nWorkspace folders created under:", WORKSPACE)
    for p in [
        CLIPS_DIR,
        DETECTIONS_DIR,
        TRACKS_DIR,
        MODELS_DIR,
        INFER_OUT_DIR,
    ]:
        print("  -", p)

    # Print a small view of dataset structure for sanity
    if DATASET_ROOT.exists():
        print("\n📂 Dataset tree (first few items):")
        print_tree(DATASET_ROOT, max_items=5)
    else:
        print("\n⚠ dataset/ folder not found under project root. "
              "Expected at:", DATASET_ROOT)

    print("\n✅ Step 1 setup done. You can now run step2_yolo_detection.py")


if __name__ == "__main__":
    main()
