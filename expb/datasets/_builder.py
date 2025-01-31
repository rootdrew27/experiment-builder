from pathlib import Path

from .dataset import Dataset
from .metadata import CocoMetadata
from . import utils

VALID_FORMATS = ["coco-seg"]
VALID_SOURCE = ["roboflow"]


# NOTE: Roboflow exhibits a strange and automatically creates a superclass. It is effectively ignored via the following function.
# Revisions will need to be made when roboflow changes this behavior.
def _build_category_name2id_coco__roboflow(annot_categories) -> dict:
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


def _get_tags_coco(annot_img: dict | None, all_tags: list) -> None:
    extra = annot_img.get("extra", False)
    if extra:
        tags = annot_img.get("user_tags", False)
    if tags:
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)


def _build_name2value_coco(annot_data) -> tuple[dict, list]:
    name2value = {}
    all_tags = []

    for annot_img in annot_data["images"]:
        fname = annot_img["file_name"]
        height, width = annot_img["height"], annot_img["width"]
        tags = _get_tags_coco(annot_img, all_tags)

        name2value[fname] = {
            "width": width,
            "height": height,
            "tags": tags,
            "categories": [],
            "segmentations": [],
        }

    values = list(name2value.values())

    for annot_mask in annot_data["annotations"]:
        img_id = annot_mask["image_id"]
        category_id = annot_mask["category_id"]
        points = annot_mask["segmentation"][0]
        point_pairs = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]

        values[img_id]["segmentations"].append(point_pairs)
        values[img_id]["categories"].append(category_id)

    return name2value, all_tags


def _build_metadata__coco(annot_path: Path, source: str) -> CocoMetadata:
    annot_data = utils._load_annotation_data(annot_path)

    if source == "roboflow":
        category_name2id = _build_category_name2id_coco__roboflow(
            annot_data["categories"]
        )
    else:
        category_name2id = None
        raise ValueError(
            "The source argument must be specified as roboflow until others are implemented."
        )

    fname2value, all_tags = _build_name2value_coco(annot_data)

    return CocoMetadata(
        path=annot_path,
        fname2value=fname2value,
        category_name2id=category_name2id,
        all_tags=all_tags,
    )


def _builder__coco(data_path: Path, annotation_filename: str, source: str, name:str) -> Dataset:
    if annotation_filename:
        annot_path = data_path / annotation_filename
    else:
        annot_path = (
            data_path / "_annotations.coco.json"
        )  # else use default filename when source="roboflow"

    metadata = _build_metadata__coco(annot_path=annot_path)
    return Dataset(data_path, metadata, name)


_format2builder = {"coco-seg": _builder__coco}


def build_dataset(
    data_path: str | Path, format: str, source: str = "roboflow", name: str = None
) -> Dataset:
    assert source == "roboflow", (
        'The source parameter only supports "roboflow" as of now.'
    )

    assert format in VALID_FORMATS, (
        f"The format ({format}) is not one of the valid formats: {VALID_FORMATS}"
    )

    data_path = Path(data_path)
    assert data_path.exists() and any(data_path.iterdir()), (
        "The path argument must be a valid path"
    )

    builder = _format2builder[format]
    return builder(data_path=data_path, source=source, name=name)
