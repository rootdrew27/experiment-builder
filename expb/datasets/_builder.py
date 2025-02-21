from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

from ._helpers import _create_new_mmap_from_ndarray
from ._utils import _get_dataset_split_dirs, _load_annotation_data, _rectify_dir_paths
from .dataset import Dataset
from .labeling_platform import LabelingPlatform, LabelingPlatformOption
from .metadata import SegmMetadata

VALID_FORMATS = [
    "coco",
]
VALID_TASKS = [
    "segm",
]
# VALID_SOURCES = ['roboflow']


# NOTE: Roboflow exhibits a strange and automatically creates a superclass. It is effectively ignored via the following function.
# Revisions will need to be made when roboflow changes this behavior.
def _build_categoryname2id_coco__roboflow(annot_categories: list[dict]) -> tuple[dict, dict]:
    name2id = {"background": 0}
    old2new = {}

    c_id = 1
    for category in annot_categories:
        name, id, super_category = (
            category["name"],
            category["id"],
            category["supercategory"],
        )

        assert name != "background", (
            "A background category is assigned automatically, please remove or change the name of the background category in your dataset."  # TODO: implement annotation utilies to handle changing class names
        )
        # If the category does NOT have a super category, then it IS a super category.
        if super_category == "none":
            continue

        else:
            name2id[name] = c_id
            old2new[id] = c_id

        c_id += 1

    return name2id, old2new


def _get_tags_coco(annot_img: dict, all_tags: list) -> list:
    extra = annot_img.get("extra", False)
    if extra:
        tags: list = extra.get("user_tags", [])
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)
        return tags
    else:
        return []


def _build_name2info_coco(annot_data: dict, oldid2newid: dict) -> tuple[dict, list]:
    name2info = {}
    all_tags: list[str] = []

    for annot_img in annot_data["images"]:
        fname = annot_img["file_name"]
        height, width = annot_img["height"], annot_img["width"]
        tags = _get_tags_coco(annot_img, all_tags)

        name2info[fname] = {
            "width": width,
            "height": height,
            "tags": tags,
            "categories": [],
            "segmentations": [],
        }

    values = list(name2info.values())

    for annot_mask in annot_data["annotations"]:
        img_id = annot_mask["image_id"]

        category_id = oldid2newid[annot_mask["category_id"]]
        values[img_id]["categories"].append(category_id)

        points = annot_mask["segmentation"][0]
        point_pairs = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        values[img_id]["segmentations"].append(point_pairs)

    return name2info, all_tags


def _build_metadata__coco_segm(
    annot_path: Path, labeling_platform: LabelingPlatformOption
) -> SegmMetadata:
    annot_data = _load_annotation_data(annot_path)

    if labeling_platform == LabelingPlatform.ROBOFLOW:
        categoryname2id, oldid2newid = _build_categoryname2id_coco__roboflow(
            annot_data["categories"]
        )
    # else:
    #     category_name2id = None
    #     raise ValueError(
    #         "The source argument must be specified as roboflow until others are implemented."
    #     )

    fname2info, all_tags = _build_name2info_coco(annot_data, oldid2newid)

    return SegmMetadata(
        path=annot_path,
        fname2info=fname2info,
        categoryname2id=categoryname2id,
        all_tags=all_tags,
    )


def _load_imgs_as_mmap(data_path: Path) -> np.memmap:
    imgs = []
    for file_path in data_path.iterdir():
        if file_path.suffix in [".png", ".jpg", ".jpeg"]:
            imgs.append(cv2.imread(str(file_path)))
    data = np.stack(imgs)
    mm = _create_new_mmap_from_ndarray(data)
    return mm


def _builder__coco_segm(
    data_path: Path,
    labeling_platform: LabelingPlatformOption,
    name: str,
) -> Dataset:
    annot_path = data_path / "_annotations.coco.json"  # use default filename when source="roboflow"

    metadata = _build_metadata__coco_segm(
        annot_path=annot_path, labeling_platform=labeling_platform
    )

    data = _load_imgs_as_mmap(data_path)

    return Dataset(data, data_path, metadata, name=name)


def _builder__voc_bbox(
    data_path: Path, labeling_platform: LabelingPlatformOption, name: str
) -> Dataset:
    raise NotImplementedError()


_format2builder: Dict[str, Dict[str, Callable[[Path, LabelingPlatformOption, str], Dataset]]] = {
    "coco": {"segm": _builder__coco_segm},
    "voc": {"bbox": _builder__voc_bbox},
}


def build_dataset(
    data_dir: str | Path,
    format: str,
    task: str,
    labeling_platform: LabelingPlatformOption = LabelingPlatform.ROBOFLOW,
    is_split: bool = False,
) -> Dataset | list[Dataset]:
    """Build a dataset of the specified format. The dataset's name attribute will be set to the root directory name.

    Args:
        data_path (str | Path): A path to the root directory of a dataset.
        format (str): The format that the annotations are in.
        labeling_platform (LabelingPlatformOption, optional): The labeling platform used to create the annotations, if known. Defaults to LabelingPlatform.ROBOFLOW.
        is_split (bool, optional): Whether or not the root directory contains sub-directories with splits of the dataset (one or more splits is valid). The dataset names will be the root director name + the split directory name. Defaults to False.

    Returns:
        Dataset|tuple[Dataset]: A dataset, or datasets if is_split=True.
    """
    data_dir_path = _rectify_dir_paths(data_dir)[0]

    assert any(data_dir_path.iterdir()) is True, (
        f"The data_path: {data_dir_path} is empty! It must contain images and annotations file."
    )

    assert labeling_platform == LabelingPlatform.ROBOFLOW, (
        "The source parameter only supports ROBOFLOW as of now."
    )

    assert format in VALID_FORMATS, (
        f"The format ({format}) is not one of the valid formats: {VALID_FORMATS}"
    )

    assert task in VALID_TASKS, f"The task ({task}) is not one of the valid tasks: {VALID_TASKS}"

    builder = _format2builder.get(format, {}).get(task, None)

    assert builder is not None, (
        f"The task type ({task}) is not valid with the format ({format}). Please choose from {list(_format2builder[format].keys())}"
    )

    if is_split:
        assert (num_split_dirs := len(_get_dataset_split_dirs(data_dir_path))), (
            f"The parameter is_split was set to True but only {num_split_dirs} splits were identified."
        )

        split_paths = [sub_path for sub_path in data_dir_path.iterdir() if sub_path.is_dir()]
        return [
            builder(
                split_path,
                labeling_platform,
                data_dir_path.stem + " " + split_path.stem,
            )
            for split_path in split_paths
        ]

    else:
        return builder(
            data_dir_path,
            labeling_platform,
            data_dir_path.stem,
        )
