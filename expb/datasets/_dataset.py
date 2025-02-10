from pathlib import Path
from typing import Iterator, Any, Dict
from typing_extensions import Self

import numpy as np
import cupy as cp  # type: ignore
from PIL import Image
import torch
from torch import Tensor

from .metadata import Metadata


class _Dataset(object):
    def __init__(
        self,
        path: Path,
        metadata: Metadata,
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
                data.setflags(write=True)
                tensor = torch.from_numpy(data)
                label = self.metadata._get_label(id)
                if self.transform:
                    data = self.transform(data)
                if self.target_transform:
                    label = self.target_transform(label)
                return tensor, label

        except ValueError as ve:
            raise ve
        except IndexError:
            raise IndexError(f"The key ({id}) did not match an entry.")

    def __iter__(self) -> Iterator:
        return DatasetIterator(dataset=self)

    def _to(self, device: str) -> None:
        self.device = device

    # def _load(self, device: str) -> None:
    #     self._data = np.stack(
    #         [self._load_img(fname) for fname in self.metadata.fnames]
    #     )
    #     if self.for_torch:
    #         self._data = Tensor(self._data)

    #     self._to(device)


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
