import os
import shutil
import tempfile
from pathlib import Path

import kagglehub
from kagglehub.clients import build_kaggle_client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglesdk.datasets.types.dataset_api_service import (
    ApiGetDatasetRequest,
    ApiListDatasetFilesRequest,
)

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
OUTPUT_DIR = ROOT / "data" / "raw"
DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
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
        if item.name in LOCAL_FILES or item.name == ".complete":
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _download_files_individually(dataset: str, dest: Path) -> Path:
    owner_slug, dataset_slug = dataset.split("/", 1)
    with build_kaggle_client() as client:
        info_req = ApiGetDatasetRequest()
        info_req.owner_slug = owner_slug
        info_req.dataset_slug = dataset_slug
        version = client.datasets.dataset_api_client.get_dataset(info_req).current_version_number

        file_names: list[str] = []
        page_token = ""
        while True:
            list_req = ApiListDatasetFilesRequest()
            list_req.owner_slug = owner_slug
            list_req.dataset_slug = dataset_slug
            list_req.dataset_version_number = version
            list_req.page_size = 200
            if page_token:
                list_req.page_token = page_token
            resp = client.datasets.dataset_api_client.list_dataset_files(list_req)
            for item in resp.dataset_files or []:
                if item and item.name:
                    file_names.append(item.name)
            page_token = resp.next_page_token
            if not page_token:
                break

    versioned = f"{dataset}/versions/{version}"
    for name in file_names:
        print(f"Downloading {name} (v{version})...", flush=True)
        kagglehub.dataset_download(versioned, path=name, output_dir=str(dest))
    return dest


load_env(ENV_FILE)

# kagglehub refuses a non-empty output_dir, and force_download deletes
# everything in it. Download to a temp folder, then replace only the
# dataset files so the local injury report and depth charts stay put.
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    try:
        downloaded = Path(kagglehub.dataset_download(DATASET, output_dir=str(tmp_path)))
    except KaggleApiHTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
        print("Dataset archive returned 404; falling back to per-file download...", flush=True)
        files_dir = tmp_path / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        downloaded = _download_files_individually(DATASET, files_dir)
    copy_dataset(downloaded, OUTPUT_DIR)

print("Path to dataset files:", OUTPUT_DIR)
