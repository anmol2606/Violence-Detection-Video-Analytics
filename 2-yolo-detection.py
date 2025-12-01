"""
step2_yolo_detection.py

Run YOLO person detection on all videos in:

    D:/violence-detection/dataset/train/*/*.avi
    D:/violence-detection/dataset/val/*/*.avi

and save per-video detection JSONs to:

    D:/violence-detection/workspace/detections/<split>_<basename>.json

Each JSON is a list of:
    {
        "frame_idx": int,
        "boxes": [
            [x1, y1, x2, y2, conf, cls], ...
        ]
    }

You can safely re-run this script: it will SKIP videos that already
have a detection JSON file.
"""

from pathlib import Path
import json
import time

import cv2
import numpy as np
from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "Ultralytics is not installed. Run:\n"
        "  pip install ultralytics\n"
        "and then re-run this script."
    ) from e


def extract_box_data(b):
    """
    Robustly extract [x1, y1, x2, y2, conf, cls] from a Ultralytics box object.
    """
    # xyxy
    try:
        arr = getattr(b, "xyxy", None)
        if arr is None:
            arr = b[0].xyxy if isinstance(b, (list, tuple)) else None
        xy = np.array(arr.cpu()) if hasattr(arr, "cpu") else np.array(arr)
        xy = xy.reshape(-1)[:4].astype(float).tolist()
        x1, y1, x2, y2 = [float(v) for v in xy]
    except Exception:
        try:
            xy = b.xyxy.cpu().numpy().reshape(-1).tolist()
            x1, y1, x2, y2 = [float(v) for v in xy[:4]]
        except Exception:
            x1 = y1 = x2 = y2 = 0.0

    # conf
    try:
        conf_arr = getattr(b, "conf", None)
        if conf_arr is not None and hasattr(conf_arr, "cpu"):
            conf = float(conf_arr.cpu().numpy().reshape(-1)[0])
        else:
            conf = float(conf_arr)
    except Exception:
        conf = 0.0

    # cls
    try:
        cls_arr = getattr(b, "cls", None)
        if cls_arr is not None and hasattr(cls_arr, "cpu"):
            cls_id = int(cls_arr.cpu().numpy().reshape(-1)[0])
        else:
            cls_id = int(cls_arr)
    except Exception:
        cls_id = -1

    return x1, y1, x2, y2, conf, cls_id


def main():
    # ====== CHANGE THIS if your project is elsewhere ======
    ROOT = Path("D:/violence-detection").resolve()
    # ======================================================

    DATASET_ROOT = ROOT / "dataset"
    WORKSPACE = ROOT / "workspace"
    DETECTIONS_DIR = WORKSPACE / "detections"
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Project root:", ROOT)
    print("Dataset root:", DATASET_ROOT)
    print("Detections dir:", DETECTIONS_DIR)

    if not DATASET_ROOT.exists():
        raise SystemExit(f"❌ dataset/ folder not found at: {DATASET_ROOT}")

    # Find YOLO weights
    yolo_candidates = list(ROOT.glob("yolo11*.pt"))
    if not yolo_candidates:
        raise SystemExit(
            f"❌ No YOLO weights (yolo11*.pt) found in {ROOT}.\n"
            f"Place yolo11x.pt or similar in the project root."
        )
    YOLO_WEIGHTS = yolo_candidates[0]
    print("Using YOLO weights:", YOLO_WEIGHTS)

    # YOLO params
    FRAME_SKIP = 1        # process every FRAME_SKIP-th frame (1 = every frame)
    CONF_THRESH = 0.25    # detection confidence
    TARGET_CLASS = 0      # COCO person = 0

    print("\nLoading YOLO model...")
    model = YOLO(str(YOLO_WEIGHTS))

    # Gather videos
    video_paths = []
    for split in ["train", "val"]:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for v in sorted(cls_dir.iterdir()):
                if v.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    video_paths.append((split, cls_dir.name, v))

    print(f"\nFound {len(video_paths)} videos to process (train+val).")

    # Process videos with resume support
    for split, cls_name, vpath in tqdm(video_paths, desc="Videos"):
        base = vpath.stem  # e.g. Train-Fight-001
        out_json = DETECTIONS_DIR / f"{split}_{base}.json"

        # resume: skip if already processed
        if out_json.exists():
            # print(f"Skipping {vpath} (already has {out_json.name})")
            continue

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"⚠ Could not open video: {vpath}")
            continue

        detections = []
        frame_idx = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue

                # Run model on this frame
                try:
                    results = model.predict(
                        source=frame,
                        conf=CONF_THRESH,
                        verbose=False
                    )
                except Exception as e:
                    print(f"Model error on frame {frame_idx} of {vpath}: {e}")
                    results = []

                boxes_out = []
                if results:
                    r = results[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for b in r.boxes:
                            x1, y1, x2, y2, conf, cls_id = extract_box_data(b)

                            # filter person class & conf
                            if TARGET_CLASS is not None and cls_id != TARGET_CLASS:
                                continue
                            if conf < CONF_THRESH:
                                continue

                            boxes_out.append(
                                [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_id)]
                            )

                detections.append({"frame_idx": frame_idx, "boxes": boxes_out})
                frame_idx += 1

        finally:
            cap.release()

        elapsed = time.time() - t0
        with open(out_json, "w") as f:
            json.dump(detections, f)
        print(f"Saved {out_json.name} ({len(detections)} frames)  time {elapsed:.1f}s")

    print("\n✅ YOLO detection pass complete.")
    print("JSON files saved in:", DETECTIONS_DIR)


if __name__ == "__main__":
    main()
