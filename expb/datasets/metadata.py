from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

class Metadata(object):
    def __init__(self, path: Path | None, fname2info: Dict[str, Dict]):
        self.path = path
        self.fname2info = fname2info
        # NOTE: having the following lists is not memory effective
        self.fnames = list(fname2info.keys())
        self.fname2batchnum = {fname: i for i, fname in enumerate(self.fnames)}
        self.fname_and_info_items = list(fname2info.items())
        self.infos = list(fname2info.values())
        self.format: str

    def __len__(self) -> int:
        return len(self.fnames)

    def __str__(self) -> str:
        return json.dumps(self.fname2info, indent=4)

    def _getitem_by_index(self, id: int) -> tuple[str, Dict]:
        fname, info = self.fname_and_info_items[id]
        return fname, info

    def _getitem_by_name(self, id: str) -> tuple[str, Dict]:
        fname, info = id, self.fname2info[id]
        return fname, info

    def __getitem__(self, id: int | str) -> tuple[str, Dict]:
        if isinstance(id, int):
            item = self._getitem_by_index(id)
        elif isinstance(id, str):
            item = self._getitem_by_name(id)
            # TODO: implement advanced idxing
        else:
            raise ValueError("Arguement for id must be one of [int, str]")

        return item

    def _new_metadata(self, fname2info: Dict) -> Metadata:
        raise NotImplementedError()

    def _split_metadata(self, idx_splits: list[NDArray]) -> list[Metadata]:
        return [
            self._new_metadata({self.fnames[idx]: self.infos[idx] for idx in idx_split})
            for idx_split in idx_splits
        ]


class SegmMetadata(Metadata):
    """An object to store and manage the metadata of a dataset. This class defines a categoryname2id dictionary which simultaneously maps each category name to an integer and encodes the class heirarchy (i.e. a category with a larger category number with take precedence in the case of overlapping segmentations). The SegmMetadata.fname2info dict has the following format:
    ```
    {
        <file_name:str> : {
            "width": <width:int>,
            "height": <height:int>,
            "tags": <tags_on_file:list[str]>,
            "categories": <categories_of_segmentations:list[int]>,
            "segmentations": <collections_of_segmentation_points:list[list[list[float]]]>,
        }
        .
        .
    }
    """

    def __init__(
        self,
        path: Path | None,
        fname2info: dict,
        categoryname2id: Dict[str, int],
        all_tags: list[str],
    ):
        super().__init__(path, fname2info)
        self.task = "segm"
        self.categoryname2id = categoryname2id
        self.all_tags = all_tags

    def set_category_hierarchy(self, hierarchy: Dict[str, int]) -> None:
        # TODO: assert hierarchy values are unique and are sequential
        # ignore supercategories
        old2new = {self.categoryname2id[k]: v for k, v in hierarchy.items()}
        for info in self.infos:
            categories = info["categories"]
            for i, cat_num in enumerate(categories):
                categories[i] = old2new[cat_num]

        self.categoryname2id.update(hierarchy)

    def _build_mask(self, segm: list[list[float]], img_w: int, img_h: int) -> np.ndarray:
        mask = Image.new("L", (img_w, img_h), 0)  # creates a new image with all pixels set to 0
        ImageDraw.Draw(mask).polygon(segm, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)

    def _get_label(self, id: int | str) -> np.ndarray:
        info: Dict

        if isinstance(id, str):
            info = self.fname2info[id]
        else:
            info = self.infos[id]

        width, height = info["width"], info["height"]
        mask = np.full((height, width), 0)

        for cat_num, segm in zip(info["categories"], info["segmentations"]):
            binary_mask = self._build_mask(segm, width, height)
            mask = np.maximum(
                mask, binary_mask * cat_num
            )  # categories are numbered such that bigger numbers take precedence in the case of overlaps.

        return mask

    def _display_label(self, id: int | str) -> None:
        label = self._get_label(id)
        max_cat_id = len(self.categoryname2id) - 1
        adj_label = label * (255 / max_cat_id)
        plt.imshow(adj_label, vmin=0, vmax=255, cmap="viridis")

    # TODO: drop categories and tags no longer present
    def _new_metadata(self, fname2info: Dict) -> Metadata:
        return SegmMetadata(
            None,
            fname2info=fname2info,
            categoryname2id=self.categoryname2id,
            all_tags=self.all_tags,
        )
