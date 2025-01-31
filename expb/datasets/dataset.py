from pathlib import Path

import numpy as np

from .metadata import Metadata


class Dataset(object):
    def __init__(self, path: Path, metadata: Metadata, name: str):
        self.data:np.ndarray|None = None
        self.path = path
        self.metadata = metadata
        self.name = name

    def __str__(self):
        return f"{self.name}\n===\nOrigin Path: {self.path}\n===\nFormat: {self.metadata.format}"

    def __len__(self):
        return self.data.shape[0]

    def _get_datum_by_index(self, id: int):
        fname, meta_value = self.meta_data[id]
        return self.data[id], fname, meta_value

    def _get_datum_by_name(self, id: str):
        fname, meta_value = self.meta_data[id]
        idx = self.meta_data.names.index(id)
        return self.data[idx], fname, meta_value

    def __getitem__(self, id: int | str):
        try:
            if isinstance(id, int):
                datum = self.get_datum_by_index(id)

            if isinstance(id, str):
                datum = self.get_datum_by_name(id)

            # TODO: Implement support for adv. indexing
            # if isinstance(idx, (Iterable[int])):
            else:
                raise ValueError("A Dataset may only be indexed with one of [int, str]")

            return datum

        except ValueError as ve:
            raise ve
        except IndexError:
            raise IndexError(f"The key ({id}) did not match an entry.")

    def shape(self):
        return self.data.shape()
    
    def _load(self, device):
        pass # TODO: implement this

    def load(self, device) -> None:
        if self.data is None:
            self._load(device)
        else:
            print(f"Dataset is already loaded onto ({self.data.device})")
        


