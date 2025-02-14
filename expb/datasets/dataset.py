from pathlib import Path
from typing import Any

from ._typing import MetadataWithLabel, AnyMetadata
from ._dataset import _Dataset

class Dataset(_Dataset):
    def __init__(
        self,
        path: Path,
        metadata: AnyMetadata,
        for_torch: bool = False,
        transform: Any | None = None,
        target_transform: Any | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.name = name
        self.device = "cpu"  # or 'cuda'
        self.for_torch = for_torch
        self.transform = transform
        self.target_transform = target_transform

    # def load(self, device: str) -> None:
    #     if hasattr(self, "_data") is False:
    #         self._load(device)
    #     elif device != self.device:
    #         self._to(device)
    #     else:
    #         print(f"Dataset is already loaded onto ({self._data.device})")

    def to(self, device: str) -> None:
        assert device in ["cpu", "cuda"], "Device must be one of ['cpu', 'cuda']"
        self._to(device)

    def display_metadata(self) -> None:
        print(self.metadata)

    def get_label(self, id:int|str):
        assert isinstance(self.metadata, MetadataWithLabel)
        return self.metadata._get_label(id)
    
    def display_label(self, id:int|str):
        assert isinstance(self.metadata, MetadataWithLabel)
        self.metadata._display_label(id)

    def torch(self, for_torch: bool = True) -> None:
        # check if this Dataset's Metadata implements a _get_label function # To allow Datasets with Tensors (but no labels) move this to _Dataset.__getitem__()
        assert isinstance(self.metadata, MetadataWithLabel), ("This Dataset does not contain a Metadata object that implements _get_label(). Thus, torch can not be used.")
        self.for_torch = for_torch
