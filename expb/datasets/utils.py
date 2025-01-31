import zipfile
from glob import glob
import shutil
import urllib.request
from pathlib import Path
import json

from ._utils import _rectify_paths, _get_dataset_split_dirs
from .labeling_platform import LabelingPlatform, LabelingPlatformOption


def fix_data_folder_structure(dest_path: Path):
    for file_path in glob(str(dest_path / "train" / "*")):
        shutil.move(file_path, dest_path)
    (dest_path / "train").rmdir()


def _extract_zip(src_path: Path, dest_path: Path) -> None:
    zf = zipfile.ZipFile(src_path)

    if zf.testzip() is not None:
        raise zipfile.BadZipFile("The zip file is corrupt.")

    else:
        print(f"Extracting zip file to {dest_path}.")
        zf.extractall(dest_path)
        zf.close()


def extract_zip_dataset(
    src_path: str | Path,
    dest_path: str | Path,
    overwrite: bool,
    labeling_platform: LabelingPlatformOption = LabelingPlatform.ROBOFLOW,
):
    
    src_path, dest_path = _rectify_paths(src_path, dest_path)

    if (overwrite) or (any(dest_path.iterdir()) is False):
        print("Removing old data.")
        shutil.rmtree(dest_path)

        dest_path.mkdir(parents=True)

        _extract_zip(src_path, dest_path)

        # Roboflow will include a split director for datasets with one split, so we remove it.
        if (
            labeling_platform == LabelingPlatform.ROBOFLOW
            and len(_get_dataset_split_dirs(dest_path)) == 1
        ):
            print("Fixing directory structure.")
            fix_data_folder_structure(dest_path=dest_path)

        print("Extraction Complete.")

    else:
        print(
            "No extraction occuring. The destination path must be empty or overwrite set to True."
        )


# TODO: add assert stmt about dataset_type, to enforce specific string values
def download_dataset(
    url: str,
    dest_path: Path,
    zip: bool,
    overwrite: bool = False,
    labeling_platform: LabelingPlatformOption = LabelingPlatform.ROBOFLOW,
):
    if (
        (overwrite) or (not dest_path.exists()) or (not any(dest_path.iterdir()))
    ):  # if overwrite or directory doesnt exist or directory is empty then...
        url = url.strip()
        print(f"Downloading dataset form {url}...")
        filepath, _ = urllib.request.urlretrieve(url)

        if dest_path.exists():
            print("Removing old data.")
            shutil.rmtree(dest_path)

        dest_path.mkdir(parents=True)

        if zip:
            _extract_zip(filepath, dest_path)

        if (
            labeling_platform == LabelingPlatform.ROBOFLOW
            and (dest_path / "train").exists()
        ):
            print("Fixing directory structure.")
            fix_data_folder_structure(dest_path=dest_path)

        print("Download Complete!")

    else:
        print(
            "No download occuring. The destination path must be empty or overwrite set to True."
        )



