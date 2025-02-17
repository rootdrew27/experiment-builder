from pathlib import Path
from typing import Iterator, Any, Dict, Callable
from typing_extensions import Self

import numpy as np
import cupy as cp  # type: ignore
from PIL import Image
import torch
from torch import Tensor

from ._typing import AnyMetadata, MetadataWithLabel
from .by import By, ByOption


class _Dataset(object):
    def __init__(
        self,
        path: Path,
        metadata: AnyMetadata,
        for_torch: bool = False,
        transform: Any | None = None,
        target_transform: Any | None = None,
        name: str | None = None,
    ) -> None:
        self._data: np.ndarray | Tensor
        self.path = path
        self.metadata = metadata
        self.name = name
        self.device = "cpu"  # or 'cuda'
        self.for_torch = for_torch
        self.transform = transform
        self.target_transform = target_transform

    def __str__(self) -> str:
        return (
            f"{self.name}\n===\nDataset Directory: {self.path}\n===\nFormat: {self.metadata.format}"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_img(self, fname: str) -> np.ndarray:
        return np.asarray(Image.open(self.path / fname).convert("RGB"))

    def _get_data_by_name(self, fname: str) -> np.ndarray:
        data = self._load_img(fname)

        return data

    def _get_data_by_index(self, idx: int) -> np.ndarray:
        fname = self.metadata.fnames[idx]
        data = self._load_img(fname)

        return data

    def _get_data(self, id: int | str) -> np.ndarray:
        if isinstance(id, str):
            data = self._get_data_by_name(id)

        else:
            data = self._get_data_by_index(id)

        return data

    def __getitem__(self, id: int | str) -> tuple[np.ndarray, str, Dict] | tuple[Tensor, Any]:
        assert isinstance(id, (int, str)), "A Dataset may only be indexed with one of [int, str]"

        try:
            data = self._get_data(id)

            if not self.for_torch:
                fname, metavalue = self.metadata[id]
                return data, fname, metavalue

            else:
                assert isinstance(self.metadata, MetadataWithLabel) # NOTE: This will never be hit right now. Yet it is necessary to appease the MyPy gods.
                tensor = torch.from_numpy(data.copy())
                label = self.metadata._get_label(id)
                if self.transform:
                    data = self.transform(data)
                if self.target_transform:
                    label = self.target_transform(label)
                return tensor, label

        except IndexError:
            raise IndexError(f"The key ({id}) did not match an entry.")
        

    def __iter__(self) -> Iterator:
        return DatasetIterator(dataset=self)

    def _to(self, device: str) -> None:
        self.device = device

    def _load(self, quantity:int, device: str) -> None:
        self._data = np.stack(
            [self._load_img(fname) for fname in self.metadata.fnames[:quantity]]
        )
        if self.for_torch:
            self._data = Tensor(self._data)        
            self._to(device)

    # TODO: Add more assertions in cases
    def _get_condition(self, by: ByOption, value: Any) -> Callable[[str, Dict[str, Dict]], bool]:
        match by:
            case By.CATEGORY:
                assert hasattr(self.metadata, "categoryname2id"), ("Dataset must use Categories to subset by CATEGORY.")
                assert value in self.metadata.categoryname2id.keys(), (
                    f"The category ({value}) is not a valid category for this dataset. This dataset has categories: {list(self.metadata.categoryname2id.keys())}"
                )

                c_id = self.metadata.categoryname2id[value]
                def condition(fname, meta_value) -> bool:
                    return (
                        c_id in meta_value["categories"]
                    )  # check if the image has a mask (segmentation) with this category.

            case By.IMG_SHAPE:
                assert isinstance(value, tuple), (
                    f"The shape ({value}) must be a tuple."
                )

                def condition(fname, meta_value) -> bool:
                    return (meta_value["height"], meta_value["width"]) == value

            case By.TAG:
                assert hasattr(self.metadata, "all_tags"), ("Dataset must use the Tags to subset by TAG.")
                assert value in self.metadata.all_tags, (
                    f"The tag ({value}) is not valid. This dataset has tags: {self.metadata.all_tags}"
                )

                def condition(fname, meta_value) -> bool:
                    return value in meta_value["tags"]

        return condition

    def _subset(self, condition) -> Dict:
        fname2info = {}
        for fname, meta_value in self.metadata.fname2info.items():
            if condition(fname=fname, meta_value=meta_value):
                fname2info[fname] = meta_value
        return self.metadata._new_metadata(fname2info)

    def _subset_complement(self, condition) -> Dict:
        fname2info_c = {}
        for fname, meta_value in self.metadata.fname2info.items():
            if not condition(fname=fname, meta_value=meta_value):
                fname2info_c[fname] = meta_value

        return self.metadata._new_metadata(fname2info_c)
    
    def _subset_and_complement(self, condition) -> tuple[Dict, Dict]:
        fname2info = {}
        fname2info_c = {}
        for fname, meta_value in self.metadata.fname2info.items():
            if condition(fname, meta_value):
                fname2info[fname] = meta_value
            else:
                fname2info_c[fname] = meta_value

        return self.metadata._new_metadata(fname2info), self.metadata._new_metadata(fname2info_c)


class DatasetIterator(Iterator):
    def __init__(self, dataset: _Dataset):
        self.dataset = dataset
        self.items = dataset.metadata.fname_and_info_items
        self._index = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, str, dict]:
        if self._index >= len(self.items):
            raise StopIteration
        fname, info = self.items[self._index]
        data = self.dataset._get_data_by_index(self._index)
        self._index += 1
        return data, fname, info
