"""
step3_tracking.py

Build simple IOU-based person tracks from YOLO detection JSONs.

Input:
    D:/violence-detection/workspace/detections/<split>_<video>.json

Each detection JSON contains:
    [
      {
        "frame_idx": int,
        "boxes": [
          [x1, y1, x2, y2, conf, cls], ...
        ]
      },
      ...
    ]

Output:
    D:/violence-detection/workspace/tracks/<same_name>.json

Each track JSON looks like:
    {
      "video": "<split>_<video>",
      "tracks": [
        {
          "id": int,
          "frames": [f0, f1, ...],
          "boxes": [
            [x1, y1, x2, y2, conf, cls], ...
          ]
        },
        ...
      ]
    }

You can safely re-run this script: existing track JSONs will be skipped.
"""

from pathlib import Path
import json
from tqdm.auto import tqdm


def iou(boxA, boxB):
    """
    Compute IoU between two [x1,y1,x2,y2,...] boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_tracks_for_video(dets, iou_thresh=0.3, max_age=20):
    """
    Build tracks for one video.

    dets: list of dicts:
        {
          "frame_idx": int,
          "boxes": [ [x1,y1,x2,y2,conf,cls], ... ]
        }

    Returns:
        list of track dicts:
        {
          "id": int,
          "frames": [...],
          "boxes": [...],
        }
    """
    tracks = []      # each track: {"id", "frames", "boxes", "age"}
    next_id = 0

    for frame_info in dets:
        frame_idx = frame_info["frame_idx"]
        boxes = frame_info.get("boxes", [])

        # increase age for all existing tracks
        for t in tracks:
            t["age"] += 1

        # assign detections to tracks
        for box in boxes:
            best_iou = 0.0
            best_track = None

            for t in tracks:
                last_box = t["boxes"][-1]
                i = iou(box, last_box)
                if i > best_iou:
                    best_iou = i
                    best_track = t

            if best_iou > iou_thresh and best_track is not None:
                # append to existing track
                best_track["boxes"].append(box)
                best_track["frames"].append(frame_idx)
                best_track["age"] = 0
            else:
                # create a new track
                tracks.append({
                    "id": next_id,
                    "frames": [frame_idx],
                    "boxes": [box],
                    "age": 0,
                })
                next_id += 1

        # drop old tracks
        tracks = [t for t in tracks if t["age"] <= max_age]

    # strip "age" before saving
    for t in tracks:
        t.pop("age", None)

    return tracks


def main():
    # ====== CHANGE THIS if your project is elsewhere ======
    ROOT = Path("D:/violence-detection").resolve()
    # ======================================================

    WORKSPACE = ROOT / "workspace"
    DETECTIONS_DIR = WORKSPACE / "detections"
    TRACKS_DIR = WORKSPACE / "tracks"
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    print("Project root:", ROOT)
    print("Detections dir:", DETECTIONS_DIR)
    print("Tracks dir:", TRACKS_DIR)

    if not DETECTIONS_DIR.exists():
        raise SystemExit(f"❌ Detections folder not found at: {DETECTIONS_DIR}\n"
                         f"Run step2_yolo_detection.py first.")

    det_files = sorted(DETECTIONS_DIR.glob("*.json"))
    if not det_files:
        raise SystemExit(f"❌ No detection JSON files found in: {DETECTIONS_DIR}")

    print(f"Found {len(det_files)} detection JSON files.")

    IOU_THRESH = 0.3
    MAX_AGE = 20

    for det_path in tqdm(det_files, desc="Building tracks"):
        out_path = TRACKS_DIR / det_path.name

        # Resume: skip if already processed
        if out_path.exists():
            # print(f"Skipping {det_path.name} (tracks already exist)")
            continue

        with open(det_path, "r") as f:
            dets = json.load(f)

        # Build tracks
        tracks = build_tracks_for_video(dets, iou_thresh=IOU_THRESH, max_age=MAX_AGE)

        track_data = {
            "video": det_path.stem,  # e.g. "train_Train-Fight-001"
            "tracks": tracks,
        }

        with open(out_path, "w") as f:
            json.dump(track_data, f)

    print("\n✅ Tracking complete.")
    print("Track JSONs saved to:", TRACKS_DIR)


if __name__ == "__main__":
    main()
