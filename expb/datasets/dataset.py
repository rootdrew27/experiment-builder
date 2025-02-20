from pathlib import Path
from typing import Any, Callable, Dict, Sequence
from typing_extensions import Self
from copy import copy
from tempfile import TemporaryFile

import numpy as np
from torch import Tensor

from expb.datasets.metadata import Metadata

from ._typing import MetadataWithLabel, AnyMetadata
from ._dataset import _Dataset
from .by import ByOption


class Dataset(_Dataset):
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
        super().__init__(data, path, metadata, for_torch, transform, target_transform, name)

    def cache(self, cache: bool) -> None:
        self._cache(cache)

    def to(self, device: str) -> None:
        assert device in ["cpu", "cuda"], "Device must be one of ['cpu', 'cuda']"
        self._to(device)

    # TODO: Make this pretty
    def display_metadata(self) -> None:
        print(self.metadata)

    def get_label(self, id: int | str):
        assert isinstance(self.metadata, MetadataWithLabel)
        return self.metadata._get_label(id)

    def display_label(self, id: int | str):
        assert isinstance(self.metadata, MetadataWithLabel)
        self.metadata._display_label(id)

    def torch(self, for_torch: bool = True) -> None:
        # check if this Dataset's Metadata implements a _get_label function # To allow Datasets with Tensors (but no labels) move this to _Dataset.__getitem__()
        assert isinstance(self.metadata, MetadataWithLabel), (
            "This Dataset does not contain a Metadata object that implements _get_label(). Thus, torch can not be used."
        )
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
                return self.new_dataset(new_metadata=subset_c)
            else:
                return self.new_dataset(new_metadata=subset), self.new_dataset(
                    new_metadata=subset_c
                )
        elif complement:
            subset_c = self._subset_complement(condition)
            return self.new_dataset(new_metadata=subset_c)
        else:
            subset = self._subset(condition)
            return self.new_dataset(new_metadata=subset)

    def apply(
        self,
        action: Callable | list[Callable],
        action_params: Dict[str, Any] | list[Dict[str, Any]],
        return_dataset: bool,
    ) -> Self | Any:
        result = self._apply(action, action_params)
        if return_dataset:
            return self._new_dataset(new_data=result)
        else:
            return result

