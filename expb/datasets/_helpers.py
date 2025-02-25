from typing import Sequence
from tempfile import TemporaryFile

import numpy as np
from numpy.typing import NDArray
import cupy as cp


def _get_idx_splits(
    split_fracs: Sequence[float], data_count: int, shuffle: bool, random_seed: int | None
) -> list[NDArray]:
    idxs = np.arange(data_count)
    if shuffle:
        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(idxs)

    split_counts = []
    prev = 0
    for split_frac in split_fracs:
        cur = int(np.ceil(split_frac * data_count)) + prev
        split_counts.append(cur)
        prev = cur
    return np.split(idxs, split_counts)[:-1]


def _create_new_mmap_from_ndarray(arr: NDArray | cp.ndarray) -> np.memmap:
    assert not isinstance(arr, np.memmap), (
        "Creating a memmap from another memmap is likely undesired. Please note that the _data attribute of Dataset is read only and as such, multiple references to it are acceptable."
    )
    file = TemporaryFile()
    mm = np.memmap(file, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[:] = arr[:] if isinstance(arr, np.ndarray) else arr[:].get() # gets arr from cupy if not an ndarray
    mm.flush()
    mm.setflags(write=False)
    return mm
