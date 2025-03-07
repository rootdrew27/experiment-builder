from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Sequence

# from memory_profiler import profile # type : ignore
import numpy as np
from numpy.typing import NDArray
import cupy as cp  # type: ignore
import torch
from torch import Tensor
from typing_extensions import Self

from .metadata import Metadata
from ._helpers import _create_new_mmap_from_ndarray, _get_idx_splits
from ._actions import _to
from ._typing import AnyMetadata, MetadataWithLabel, DatasetData
from .by import By, ByOption


class _Dataset(object):
    def __init__(
        self,
        data: DatasetData,
        path: Path,
        metadata: AnyMetadata,
        for_torch: bool = False,
        transform: Any | None = None,
        target_transform: Any | None = None,
        name: str | None = None,
    ) -> None:
        self._data: DatasetData = data
        self.path = path
        self.metadata = metadata
        self._action_queue: list[tuple[Callable, tuple, Dict[str, Any]]] = []
        self.name = name
        self.device = "cpu"  # or 'cuda'
        self._xp = np
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
                return self._xp.asarray(data), fname, metavalue

            else:
                assert isinstance(
                    self.metadata, MetadataWithLabel
                )  # NOTE: This will never be hit right now. Yet it is necessary to appease the MyPy gods.
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
        match device:
            case "cuda":
                self._xp = cp
                self._apply((_to, (cp,), {}))
            case "cpu":
                self._xp = np
                self._apply((_to, (np,), {}))
        self.device = device
        

    def _cache(self, cache: bool) -> None:
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
                assert hasattr(self.metadata, "categoryname2id"), (
                    "Dataset must use Categories to subset by CATEGORY."
                )
                assert value in self.metadata.categoryname2id.keys(), (
                    f"The category ({value}) is not a valid category for this dataset. This dataset has categories: {list(self.metadata.categoryname2id.keys())}"
                )

                c_id = self.metadata.categoryname2id[value]

                def condition(fname: str, info: Dict) -> bool:
                    return (
                        c_id in info["categories"]
                    )  # check if the image has a mask (segmentation) with this category.

            case By.IMGSHAPE:
                assert isinstance(value, tuple), f"The shape ({value}) must be a tuple."

                def condition(fname: str, info: Dict) -> bool:
                    return (info["height"], info["width"]) == value

            case By.TAG:
                assert hasattr(self.metadata, "all_tags"), (
                    "Dataset must use the Tags to subset by TAG."
                )
                assert value in self.metadata.all_tags, (
                    f"The tag ({value}) is not valid. This dataset has tags: {self.metadata.all_tags}"
                )

                def condition(fname: str, info: Dict) -> bool:
                    return value in info["tags"]

        return condition

    def _subset(self, condition: Callable) -> Metadata | Any:
        fname2info = {}
        for fname, info in self.metadata.fname2info.items():
            if condition(fname=fname, info=info):
                fname2info[fname] = info
        return self.metadata._new_metadata(fname2info)

    def _subset_complement(self, condition: Callable) -> Metadata:
        fname2info_c = {}
        for fname, info in self.metadata.fname2info.items():
            if not condition(fname=fname, info=info):
                fname2info_c[fname] = info

        return self.metadata._new_metadata(fname2info_c)

    def _subset_and_complement(self, condition: Callable) -> tuple[Metadata, Metadata]:
        fname2info = {}
        fname2info_c = {}
        for fname, info in self.metadata.fname2info.items():
            if condition(fname, info):
                fname2info[fname] = info
            else:
                fname2info_c[fname] = info

        return self.metadata._new_metadata(fname2info), self.metadata._new_metadata(fname2info_c)

    def _split(
        self, split_fractions: Sequence[float], shuffle: bool, random_seed: int | None
    ) -> list:
        n_data = len(self)

        idx_splits = _get_idx_splits(split_fractions, n_data, shuffle, random_seed)
        data_splits = [self._data[idx_split] for idx_split in idx_splits]
        metadata_splits = self.metadata._split_metadata(idx_splits)

        return [
            self._new_dataset(new_data=data_split, new_metadata=metadata_split)
            for data_split, metadata_split in zip(data_splits, metadata_splits)
        ]

    def _apply(self, action: tuple[Callable, tuple, Dict[str, Any]]) -> None:
        self._action_queue.append(action)

    # TODO: optimize
    def _execute(self, return_dataset: bool) -> _Dataset | NDArray:
        try:
            data = np.asarray(self._data[:])
            for action in self._action_queue:
                func, params, kw_params = action
                data = func(data, *params, **kw_params)

            # check if the last action returns a dataset
            if return_dataset:
                return self._new_dataset(data)
            else:
                return data
            
        except Exception as ex:
            raise ex
        finally:
            self._action_queue.clear()
            if self.device == "cuda":
                p_mempool = cp.get_default_pinned_memory_pool()
                mempool = cp.get_default_memory_pool()
                p_mempool.free_all_blocks()
                mempool.free_all_blocks()


    # TODO: clean logic
    def _new_dataset(
        self,
        new_data: DatasetData | None = None,
        new_metadata: Metadata | None = None,
    ) -> Self:
        new_dataset = copy(self)
        if isinstance(new_data, np.memmap):
            new_dataset._data = new_data
        elif isinstance(new_data, (np.ndarray | cp.ndarray)):
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
