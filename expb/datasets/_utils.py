from pathlib import Path
import json


def _rectify_paths(*args: Path) -> Path | tuple[Path]:
    output = []
    for arg in args:
        if isinstance(arg, str):
            arg = Path(arg)
        output.append(arg)

    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)


def _rectify_dir_paths(*args) -> Path | tuple[Path]:
    output = []
    for arg in args:
        arg = _rectify_paths(arg)
        assert arg.is_dir(), f"The path: {arg} must be a directory!"
        output.append(arg)

    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)

def _get_dataset_split_dirs(dataset_root_dir:Path):
    return [path.stem for path in dataset_root_dir.iterdir() if path.is_dir()]

def _load_annotation_data(path_to_data):
    """Load the json data for a given path to a json file.

    Args:
        path_to_data (Path): The path to a json file containing mask data (from roboflow).

    Returns:
        dict|list: The json, in it related python form.
    """
    return json.load(open(path_to_data, "r", encoding="utf-8"))
