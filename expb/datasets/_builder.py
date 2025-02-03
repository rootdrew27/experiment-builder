from pathlib import Path

from .dataset import Dataset
from .metadata import CocoMetadata
from ._utils import _rectify_dir_paths, _load_annotation_data, _get_dataset_split_dirs
from .labeling_platform import LabelingPlatform, LabelingPlatformOption

VALID_FORMATS = ["coco-seg"]
VALID_SOURCE = ["roboflow"]


# NOTE: Roboflow exhibits a strange and automatically creates a superclass. It is effectively ignored via the following function.
# Revisions will need to be made when roboflow changes this behavior.
def _build_category_name2id_coco__roboflow(annot_categories: list[dict]) -> dict:
    name2id_temp = {}

    for category in annot_categories:
        name, id, super_category = (
            category["name"],
            category["id"],
            category["supercategory"],
        )

        # If the category does NOT have a super category, then it IS a super category.
        if super_category == "none":
            name2id_temp[name + "__super"] = id

        else:
            name2id_temp[name] = id

    return name2id_temp


def _get_tags_coco(annot_img: dict, all_tags: list) -> list:
    extra = annot_img.get("extra", False)
    if extra:
        tags:list = annot_img.get("user_tags", [])
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)
        return tags
    else:
        return []


def _build_name2info_coco(annot_data: dict) -> tuple[dict, list]:
    name2info = {}
    all_tags:list[str] = []

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
        
        category_id = annot_mask["category_id"]
        values[img_id]["categories"].append(category_id)

        points = annot_mask["segmentation"][0]
        point_pairs = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]
        values[img_id]["segmentations"].append(point_pairs)

    return name2info, all_tags


def _build_metadata__coco(
    annot_path: Path, labeling_platform: LabelingPlatformOption
) -> CocoMetadata:
    annot_data = _load_annotation_data(annot_path)

    if labeling_platform == LabelingPlatform.ROBOFLOW:
        category_name2id = _build_category_name2id_coco__roboflow(
            annot_data["categories"]
        )
    # else:
    #     category_name2id = None
    #     raise ValueError(
    #         "The source argument must be specified as roboflow until others are implemented."
    #     )

    fname2info, all_tags = _build_name2info_coco(annot_data)

    return CocoMetadata(
        path=annot_path,
        fname2info=fname2info,
        category_name2id=category_name2id,
        all_tags=all_tags,
    )


def _builder__coco(
    data_path: Path,
    labeling_platform: LabelingPlatformOption,
    name: str,
) -> Dataset:
    annot_path = (
        data_path / "_annotations.coco.json"
    )  # use default filename when source="roboflow"

    metadata = _build_metadata__coco(
        annot_path=annot_path, labeling_platform=labeling_platform
    )
    return Dataset(data_path, metadata, name)


_format2builder = {"coco-seg": _builder__coco}


def build_dataset(
    data_dir: str | Path,
    format: str,
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

    builder = _format2builder[format]

    if is_split:
        assert (num_split_dirs := len(_get_dataset_split_dirs(data_dir_path))) , (f"The parameter is_split was set to True but only {num_split_dirs} splits were identified.")

        split_paths = [
            sub_path
            for sub_path in data_dir_path.iterdir()
            if sub_path.is_dir()
        ]
        return [
            builder(
                split_path,
                labeling_platform,
                name=data_dir_path.stem + " " + split_path.stem,
            )
            for split_path in split_paths
        ]

    else:
        return builder(
            data_path=data_dir_path,
            labeling_platform=labeling_platform,
            name=data_dir_path.stem,
        )
