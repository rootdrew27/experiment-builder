from pathlib import Path
import json
from typing import Any
import zipfile

def _rectify_paths(*args: Path | str) -> tuple[Path, ...]:
    output = []
    for arg in args:
        if isinstance(arg, str):
            arg = Path(arg)
        output.append(arg)

    else:
        return tuple(output)


def _rectify_dir_paths(*args: Path | str) -> tuple[Path, ...]:
    paths = _rectify_paths(*args)
    for path in paths:
        assert path.is_dir(), f"The path: {path} must be a directory!"
    return paths


def _get_dataset_split_dirs(dataset_root_dir: Path) -> list[str]:
    return [path.stem for path in dataset_root_dir.iterdir() if path.is_dir()]


def _load_annotation_data(path_to_data: Path | str) -> Any:
    """Load the json data for a given path to a json file.

    Args:
        path_to_data (Path): The path to a json file containing mask data (from roboflow).

    Returns:
        dict|list: The json, in it related python form.
    """
    with open(path_to_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _extract_zip(src_path: Path | str, dest_path: Path) -> None:
    zf = zipfile.ZipFile(src_path)

    if zf.testzip() is not None:
        raise zipfile.BadZipFile("The zip file is corrupt.")

    else:
        print(f"Extracting zip file to {dest_path}.")
        zf.extractall(dest_path)
        zf.close()