from glob import glob
import shutil
import urllib.request
from pathlib import Path

from ._utils import _rectify_paths, _get_dataset_split_dirs, _extract_zip
from .labeling_platform import LabelingPlatform, LabelingPlatformOption


def fix_data_folder_structure(dest_path: Path) -> None:
    for file_path in glob(str(dest_path / "train" / "*")):
        shutil.move(file_path, dest_path)
    (dest_path / "train").rmdir()


def extract_zip_dataset(
    src_path: str | Path,
    dest_path: str | Path,
    overwrite: bool,
    labeling_platform: LabelingPlatformOption = LabelingPlatform.ROBOFLOW,
) -> None:
    src_path, dest_path = _rectify_paths(src_path, dest_path)

    assert src_path.exists(), f"The zip file at: {src_path} does not exist!"

    if dest_path.exists():
        assert overwrite or any(dest_path.iterdir()), (
            "No extraction occurring. If it exists, the destination path must be empty or overwrite set to True."
        )  # NOTE: This debatably uses Exceptions for control flow

        print("Removing old data.")
        shutil.rmtree(dest_path)
        dest_path.mkdir(parents=True, exist_ok=True)

    _extract_zip(src_path, dest_path)

    # Roboflow will include a split director for datasets with one split, so we remove it.
    if (
        labeling_platform == LabelingPlatform.ROBOFLOW
        and len(_get_dataset_split_dirs(dest_path)) == 1
    ):
        print("Fixing directory structure.")
        fix_data_folder_structure(dest_path=dest_path)

    print("Extraction Complete.")


# TODO: add assert stmt about dataset_type, to enforce specific string values
def download_dataset(
    url: str,
    dest_path: Path,
    is_zip: bool,
    overwrite: bool = False,
    labeling_platform: LabelingPlatformOption = LabelingPlatform.ROBOFLOW,
) -> None:
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

        if is_zip:
            _extract_zip(filepath, dest_path)

        if labeling_platform == LabelingPlatform.ROBOFLOW and (dest_path / "train").exists():
            print("Fixing directory structure.")
            fix_data_folder_structure(dest_path=dest_path)

        print("Download Complete!")

    else:
        print("No download occuring. The destination path must be empty or overwrite set to True.")
