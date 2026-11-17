import os
import shutil
import tempfile
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
OUTPUT_DIR = ROOT / "data" / "raw"
# These live next to the dataset but are not part of it.
LOCAL_FILES = {"injuries.txt", "depth-charts.txt"}


def load_env(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. Add KAGGLE_API_TOKEN there.")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def copy_dataset(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in LOCAL_FILES:
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


load_env(ENV_FILE)

# kagglehub refuses a non-empty output_dir, and force_download deletes
# everything in it. Download to a temp folder, then replace only the
# dataset files so the local injury report and depth charts stay put.
with tempfile.TemporaryDirectory() as tmp:
    downloaded = Path(
        kagglehub.dataset_download(
            "eoinamoore/historical-nba-data-and-player-box-scores",
            output_dir=tmp,
        )
    )
    copy_dataset(downloaded, OUTPUT_DIR)

print("Path to dataset files:", OUTPUT_DIR)
