from pathlib import Path

import numpy as np
import cupy as cp # type: ignore
import cv2


from .metadata import Metadata


class Dataset(object):
    def __init__(self, path: Path, metadata: Metadata, name: str):
        self._data: np.ndarray
        self.path = path
        self.metadata = metadata
        self.name = name
        self.device = "cpu"  # or 'cuda'

    def __str__(self) -> str:
        return f"{self.name}\n===\nDataset Directory: {self.path}\n===\nFormat: {self.metadata.format}"

    def __len__(self) -> int:
        return len(self.metadata)

    def _get_datum_by_index(self, id: int) -> tuple[np.ndarray, str, dict]:
        fname, meta_value = self.metadata[id]
        return self._data[id], fname, meta_value

    def _get_datum_by_name(self, id: str) -> tuple[np.ndarray, str, dict]:
        fname, meta_value = self.metadata[id]
        idx = self.metadata.fnames.index(id)
        return self._data[idx], fname, meta_value

    def __getitem__(self, id: int | str) -> tuple[np.ndarray, str, dict]:
        try:
            if isinstance(id, int):
                datum = self._get_datum_by_index(id)

            if isinstance(id, str):
                datum = self._get_datum_by_name(id)

            # TODO: Implement support for adv. indexing
            # if isinstance(idx, (Iterable[int])):
            else:
                raise ValueError("A Dataset may only be indexed with one of [int, str]")

            return datum

        except ValueError as ve:
            raise ve
        except IndexError:
            raise IndexError(f"The key ({id}) did not match an entry.")

    def _load(self, device: str) -> None:
        self._data = np.stack(
            [cv2.imread(self.path / fname) for fname in self.metadata.fnames]
        )
        self._to(device)

    def load(self, device: str) -> None:
        if hasattr(self, "_data") is False:
            self._load(device)
        elif device != self.device:
            self._to(device)
        else:
            print(f"Dataset is already loaded onto ({self._data.device})")

    def _to(self, device: str) -> None:
        match device:
            case "cuda":
                self._data = cp.asarray(self._data)
            case "cpu":
                self._data = cp.asnumpy(self._data)
            case _:
                raise ValueError(f"({device} is an invalid device. Choose from one of ['cpu', 'cuda'])")
        
    def to(self, device: str) -> None:
        self.device = device if isinstance(device, str) else "cpu"
        if self._data.device != device:
            self._to(device)
        else:
            print(f"Dataset is already loaded onto ({self._data.device})")
