from pathlib import Path
from typing import Any, Dict
from typing_extensions import Self
from copy import copy

from ._typing import MetadataWithLabel, AnyMetadata
from ._dataset import _Dataset
from .by import By, ByOption

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

    def subset(
        self,
        by: ByOption,
        value: Any,
        complement: bool = False,
        return_both: bool = False,
        **kwargs,
    ) -> Self | tuple[Self, Self]:
        """Create a subset of the dataset. Optionally return the complement, or return both sets.

        Args:
            by (ByOption): The method by which the dataset will be split. Use the By class attributes as arguments to this parameter.
            value (Any, optional): The value to be compared against.
            complement (bool): If True, return the complement of the set described by the 'by' and 'value' arguments. Otherwise return the described set. Defaults to False.
            return_both (bool): If True, return both the described set and its complement, otherwise return the set described. Defaults to False.
        Returns:
            Dataset|tuple[Dataset]: A Dataset or Datasets, as described by the condition and value.

        ---
        Example:
        grass_ds = ds.subset(by=By.CATEGORY, value="grass")
        """

        condition = self._get_condition(by, value)

        if return_both:
            subset, subset_c = self._subset_and_complement(condition)
            if complement:
                return self.new_dataset(subset_c)
            else:
                return self.new_dataset(subset), self.new_dataset(subset_c)
        elif complement:
            subset_c = self._subset_complement(condition)
            return self.new_dataset(subset_c)
        else:
            subset = self._subset(condition)
            return self.new_dataset(subset)

    def new_dataset(self, new_metadata):
        new_dataset = copy(self)
        new_dataset.path = None
        new_dataset.metadata = new_metadata
        new_dataset.name = self.name + " Subset"
        new_dataset.device = self.device
        return new_dataset
        