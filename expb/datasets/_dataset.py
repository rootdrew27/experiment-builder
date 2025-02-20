from ast import Call
from pathlib import Path
from typing import Iterator, Any, Dict, Callable
from typing_extensions import Self
from tempfile import TemporaryFile

import numpy as np
import cupy as cp  # type: ignore
from PIL import Image
import torch
from torch import Tensor

from expb.datasets.metadata import Metadata

from ._typing import AnyMetadata, MetadataWithLabel
from ._helpers import _get_idx_splits, _create_new_mmap_from_ndarray
from .by import By, ByOption


class _Dataset(object):
    def __init__(
        self,
        data: np.memmap | np.ndarray | Tensor,
        path: Path,
        metadata: AnyMetadata,
        for_torch: bool = False,
        transform: Any | None = None,
        target_transform: Any | None = None,
        name: str | None = None,
    ) -> None:
        self._data: np.ndarray | np.memmap | Tensor = data
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

    def _get_data_by_name(self, fname: str) -> np.ndarray:
        idx = self.metadata.fname2batchnum[fname]
        return np.array(self._data[idx])

    def _get_data_by_index(self, idx: int) -> np.ndarray:
        return np.array(self._data[idx])

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

    def _cache(self, cache:bool) -> None:
        if cache:
            if self.for_torch:
                self._data = Tensor(self._data)    
            else:
                self._data = np.array(self._data)
        else:
            if not isinstance(self._data, np.memmap):
                if isinstance(self._data, Tensor):
                    self._data = self._data.numpy()
                self._data = _create_new_mmap_from_ndarray(self._data)


    # TODO: Add more assertions in cases
    def _get_condition(self, by: ByOption, value: Any) -> Callable[[str, Dict[str, Dict]], bool]:
        match by:
            case By.CATEGORY:
                assert hasattr(self.metadata, "categoryname2id"), ("Dataset must use Categories to subset by CATEGORY.")
                assert value in self.metadata.categoryname2id.keys(), (
                    f"The category ({value}) is not a valid category for this dataset. This dataset has categories: {list(self.metadata.categoryname2id.keys())}"
                )

                c_id = self.metadata.categoryname2id[value]
                def condition(fname, info) -> bool:
                    return (
                        c_id in info["categories"]
                    )  # check if the image has a mask (segmentation) with this category.

            case By.IMG_SHAPE:
                assert isinstance(value, tuple), (
                    f"The shape ({value}) must be a tuple."
                )

                def condition(fname, info) -> bool:
                    return (info["height"], info["width"]) == value

            case By.TAG:
                assert hasattr(self.metadata, "all_tags"), ("Dataset must use the Tags to subset by TAG.")
                assert value in self.metadata.all_tags, (
                    f"The tag ({value}) is not valid. This dataset has tags: {self.metadata.all_tags}"
                )

                def condition(fname, info) -> bool:
                    return value in info["tags"]

        return condition

    def _subset(self, condition) -> Metadata:
        fname2info = {}
        for fname, info in self.metadata.fname2info.items():
            if condition(fname=fname, info=info):
                fname2info[fname] = info
        return self.metadata._new_metadata(fname2info)

    def _subset_complement(self, condition) -> Metadata:
        fname2info_c = {}
        for fname, info in self.metadata.fname2info.items():
            if not condition(fname=fname, info=info):
                fname2info_c[fname] = info

        return self.metadata._new_metadata(fname2info_c)
    
    def _subset_and_complement(self, condition) -> tuple[Metadata, Metadata]:
        fname2info = {}
        fname2info_c = {}
        for fname, info in self.metadata.fname2info.items():
            if condition(fname, info):
                fname2info[fname] = info
            else:
                fname2info_c[fname] = info

        return self.metadata._new_metadata(fname2info), self.metadata._new_metadata(fname2info_c)

    # TODO: Clean this up
    def _apply(self, action, action_params):
        if isinstance(self._data, np.ndarray):
            data = self._data
        elif isinstance(self._data, np.memmap):
            data = np.array(self._data)

        if isinstance(action, list):
            for i in len(action):
                if i < len(action_params):
                    params = action_params[i]
                    data = action[i](data, **params)
                else:
                    data = action[i](data)
        else:
            data = action(data, **action_params)

        return data

        # TODO: move to helpers or builder or _dataset
    # TODO: clean logic
    def _new_dataset(
        self, new_data: np.ndarray | np.memmap | Tensor | None = None, new_metadata:  Metadata | None = None
    ) -> Self:
        new_dataset = copy(self)
        if isinstance(new_data, np.memmap):
                new_dataset._data = new_data
        elif isinstance(new_data, np.ndarray):
                new_dataset._data = _create_new_mmap_from_ndarray(new_data)
        else:
            pass
        if new_metadata:
            new_dataset.metadata = new_metadata

        return new_dataset



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
