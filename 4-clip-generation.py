"""
step4_clip_generation.py

Generate per-person video clips from IOU-based tracks.

Input:
    D:/violence-detection/workspace/tracks/*.json
    D:/violence-detection/dataset/train|val/Fight|NonFight/*.avi (or .mp4, etc.)

Each track JSON (from step3_tracking.py) looks like:
    {
      "video": "train_Train-Fight-001",
      "tracks": [
        {
          "id": int,
          "frames": [f0, f1, ...],
          "boxes": [
            [x1,y1,x2,y2,conf,cls], ...
          ]
        },
        ...
      ]
    }

Output:
    D:/violence-detection/workspace/clips/<video>_trackXXXX_clipYYYY.npy
    D:/violence-detection/workspace/clips/labels.csv

labels.csv columns:
    clip_filename,label,video_basename,track_id,first_frame_idx,last_frame_idx

Label:
    Fight    -> 1
    NonFight -> 0

You can safely re-run this script:
    - It will SKIP individual clips that already exist.
    - It will append new rows to labels.csv.
"""

from pathlib import Path
import json
import cv2
import numpy as np
from tqdm.auto import tqdm

# -------------- CONFIG --------------
ROOT = Path("D:/violence-detection").resolve()
DATASET_ROOT = ROOT / "dataset"
WORKSPACE = ROOT / "workspace"
TRACKS_DIR = WORKSPACE / "tracks"
CLIPS_DIR = WORKSPACE / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

CLIP_LEN = 16
STRIDE = 8
IMG_SIZE = 112

LABELS_CSV = CLIPS_DIR / "labels.csv"
# ------------------------------------


def find_video_path(video_basename: str, split_hint: str = None) -> str | None:
    """
    Find the original video file given its basename, e.g. "Train-Fight-001".

    We search under dataset/train and dataset/val, inside Fight/NonFight folders.
    """
    splits_to_try = []
    if split_hint in ("train", "val"):
        splits_to_try = [split_hint]
    else:
        splits_to_try = ["train", "val"]

    exts = [".avi", ".mp4", ".mov", ".mkv"]

    for split in splits_to_try:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue

        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for ext in exts:
                cand = cls_dir / (video_basename + ext)
                if cand.exists():
                    return str(cand)

    return None


def video_label_from_name(video_basename: str) -> int:
    """
    Decide label from video_basename string:
        contains "Fight"    -> 1
        contains "NonFight" -> 0
    """
    name = video_basename.lower()
    if "nonfight" in name:
        return 0
    if "fight" in name:
        return 1
    # default / unexpected
    return -1


def ensure_labels_header():
    """
    Ensure labels.csv exists and has header.
    """
    if not LABELS_CSV.exists():
        with open(LABELS_CSV, "w", encoding="utf-8") as f:
            f.write("clip_filename,label,video_basename,track_id,first_frame_idx,last_frame_idx\n")


def main():
    print("Project root:", ROOT)
    print("Tracks dir:", TRACKS_DIR)
    print("Clips dir:", CLIPS_DIR)

    if not TRACKS_DIR.exists():
        raise SystemExit(f"❌ Tracks folder not found at: {TRACKS_DIR}\n"
                         f"Run step3_tracking.py first.")

    ensure_labels_header()

    track_files = sorted(TRACKS_DIR.glob("*.json"))
    if not track_files:
        raise SystemExit(f"❌ No track JSON files found in: {TRACKS_DIR}")

    print(f"Found {len(track_files)} track JSON files.")

    total_clips = 0
    skipped_short = 0

    for tf in tqdm(track_files, desc="Generating clips"):
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data["video"] is like: "train_Train-Fight-001"
        video_key = data.get("video", tf.stem)
        if "_" in video_key:
            split_token, video_basename = video_key.split("_", 1)
            split_token = split_token.lower()  # "train" or "val"
        else:
            video_basename = video_key
            # infer from name if possible
            if video_basename.startswith("Train"):
                split_token = "train"
            elif video_basename.startswith("Val"):
                split_token = "val"
            else:
                split_token = None

        vpath = find_video_path(video_basename, split_hint=split_token)
        if not vpath:
            print(f"⚠ Video not found for {video_basename} (from {tf.name})")
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"⚠ Cannot open video: {vpath}")
            continue

        video_label = video_label_from_name(video_basename)
        if video_label == -1:
            print(f"⚠ Could not infer label from video name: {video_basename}, skipping.")
            cap.release()
            continue

        for track in data.get("tracks", []):
            tid = track.get("id", 0)
            frames = track.get("frames", [])
            boxes = track.get("boxes", [])

            if len(frames) < CLIP_LEN:
                skipped_short += 1
                continue

            # sort by frame index to ensure correct order
            zipped = sorted(zip(frames, boxes), key=lambda x: x[0])
            frames_sorted = [z[0] for z in zipped]
            boxes_sorted = [z[1] for z in zipped]

            # Sliding window over track frames
            for start_idx in range(0, len(frames_sorted) - CLIP_LEN + 1, STRIDE):
                # this "start_idx" is index in the arrays, not the actual frame number
                window_frames = frames_sorted[start_idx:start_idx + CLIP_LEN]
                window_boxes = boxes_sorted[start_idx:start_idx + CLIP_LEN]

                # determine clip file name
                clip_name = f"{video_basename}_track{tid:04d}_clip{start_idx:04d}.npy"
                out_path = CLIPS_DIR / clip_name

                # SKIP if already exists (resume-safe)
                if out_path.exists():
                    continue

                clip_frames = []
                ok = True

                for fi, box in zip(window_frames, window_boxes):
                    # box = [x1,y1,x2,y2,conf,cls] (from step3)
                    x1, y1, x2, y2 = map(int, box[:4])

                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        ok = False
                        break

                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        ok = False
                        break

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        ok = False
                        break

                    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    clip_frames.append(crop)

                if not ok or len(clip_frames) != CLIP_LEN:
                    continue

                arr = np.stack(clip_frames, axis=0)  # (T,H,W,3)
                np.save(out_path, arr)

                # Write labels row
                first_f = window_frames[0]
                last_f = window_frames[-1]
                with open(LABELS_CSV, "a", encoding="utf-8") as f:
                    f.write(f"{clip_name},{video_label},{video_basename},{tid},{first_f},{last_f}\n")

                total_clips += 1

        cap.release()

    print("\n=== SUMMARY ===")
    print("New clips created:", total_clips)
    print("Skipped short tracks (len < CLIP_LEN):", skipped_short)
    print("Clips dir:", CLIPS_DIR)
    print("Labels file:", LABELS_CSV)


if __name__ == "__main__":
    main()
