from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import Self

from ._dataset import _Dataset
from ._typing import AnyMetadata, MetadataWithLabel
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

    def cache(self, cache: bool=True) -> None:
        self._cache(cache)

    def to(self, device: str) -> None:
        assert device in ["cpu", "cuda"], "Device must be one of ['cpu', 'cuda']"
        self._to(device)

    # TODO: Make the output pretty
    def display_metadata(self) -> None:
        print(self.metadata)

    def get_label(self, id: int | str) -> NDArray:
        assert isinstance(self.metadata, MetadataWithLabel)
        return self.metadata._get_label(id)

    def display_label(self, id: int | str) -> None:
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
        **kwargs: Any,
    ) -> _Dataset | tuple[_Dataset, _Dataset]:
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
                return self._new_dataset(new_metadata=subset_c)
            else:
                return self._new_dataset(new_metadata=subset), self._new_dataset(
                    new_metadata=subset_c
                )
        elif complement:
            subset_c = self._subset_complement(condition)
            return self._new_dataset(new_metadata=subset_c)
        else:
            subset = self._subset(condition)
            return self._new_dataset(new_metadata=subset)

    def split(
        self, split_fracs: Sequence[float], shuffle: bool, random_seed: int | None = None
    ) -> list[Dataset]:
        assert sum(split_fracs) == 1, "The split fractions must sum to 1."

        return self._split(split_fracs, shuffle, random_seed)

    def apply(
        self,
        action: Callable | list[Callable],
        action_params: Dict[str, Any] | list[Dict[str, Any]],
        return_dataset: bool,
    ) -> Dataset | Any:
        result = self._apply(action, action_params)
        if return_dataset:
            return self._new_dataset(new_data=result)
        else:
            return result
