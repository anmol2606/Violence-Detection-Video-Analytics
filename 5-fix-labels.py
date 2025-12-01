"""
fix_labels.py

Fix labels in labels.csv based on video_basename:

    * video_basename containing "NonFight" -> label = 0
    * video_basename containing "Fight"    -> label = 1

Also:
    * Backs up original labels.csv as labels_backup_before_fix.csv
    * Prints train/val class counts.

Expects:
    D:/violence-detection/workspace/clips/labels.csv
"""

from pathlib import Path
import pandas as pd
import shutil


def infer_label_from_name(vname: str) -> int:
    """
    Infer label from video_basename string.
        "NonFight" -> 0
        "Fight"    -> 1
        otherwise  -> -1
    """
    name = str(vname).lower()
    if "nonfight" in name:
        return 0
    if "fight" in name:
        return 1
    return -1


def main():
    # ====== CHANGE THIS if your project is elsewhere ======
    ROOT = Path("D:/violence-detection").resolve()
    # ======================================================

    CLIPS_DIR = ROOT / "workspace" / "clips"
    LABELS = CLIPS_DIR / "labels.csv"
    BACKUP = CLIPS_DIR / "labels_backup_before_fix.csv"

    print("Clips dir:", CLIPS_DIR)
    print("Labels path:", LABELS)
    print("Exists:", LABELS.exists())

    if not LABELS.exists():
        raise SystemExit(f"❌ labels.csv not found at {LABELS}\n"
                         f"Run step4_clip_generation.py first.")

    # 1) Backup original once
    if not BACKUP.exists():
        shutil.copy(LABELS, BACKUP)
        print("✅ Backup saved to:", BACKUP)
    else:
        print("ℹ Backup already exists at:", BACKUP)

    # 2) Load and fix labels
    df = pd.read_csv(LABELS)

    if "video_basename" not in df.columns:
        raise SystemExit("❌ labels.csv missing 'video_basename' column.")

    df["label"] = df["video_basename"].apply(infer_label_from_name)
    before_rows = len(df)
    df = df[df["label"] != -1].reset_index(drop=True)
    removed = before_rows - len(df)

    # 3) Save back
    df.to_csv(LABELS, index=False)
    print("\n✅ Fixed labels saved to labels.csv")
    if removed > 0:
        print(f"Removed {removed} rows with unknown label.")

    # 4) Add split column for summary
    def split_from_name(vname: str) -> str:
        name = str(vname)
        if name.startswith("Train"):
            return "train"
        if name.startswith("Val"):
            return "val"
        return "unknown"

    df["split"] = df["video_basename"].apply(split_from_name)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print("\nTrain label counts:")
    print(train_df["label"].value_counts())
    print("\nVal label counts:")
    print(val_df["label"].value_counts())


if __name__ == "__main__":
    main()
