from pathlib import Path
from typing import Iterator

import numpy as np
import cupy as cp  # type: ignore
import cv2


from .metadata import Metadata


class Dataset(object):
	def __init__(self, path: Path, metadata: Metadata, name: str):
		self._data: np.ndarray
		self.path = path
		self.metadata = metadata
		self.name = name
		self.device = 'cpu'  # or 'cuda'

	def __str__(self) -> str:
		return (
			f'{self.name}\n===\nDataset Directory: {self.path}\n===\nFormat: {self.metadata.format}'
		)

	def __len__(self) -> int:
		return len(self.metadata)

	def _get_datum_by_index(self, id: int) -> tuple[np.ndarray, str, dict]:
		fname, meta_value = self.metadata[id]
		return self._data[id], fname, meta_value

	def _get_datum_by_name(self, id: str) -> tuple[np.ndarray, str, dict]:
		fname, meta_value = self.metadata[id]
		idx = self.metadata.fnames.index(id)
		return self._data[idx], fname, meta_value

	def _get_unloaded_datum_by_index(id):
		raise NotImplementedError()

	def _get_unloaded_datum_by_name(id):
		raise NotImplementedError()

	def __getitem__(self, id: int | str) -> tuple[np.ndarray, str, dict]:
		assert isinstance(id, (int, str)), 'A Dataset may only be indexed with one of [int, str]'
		try:
			if hasattr(self, '_data'):
				if isinstance(id, int):
					datum = self._get_datum_by_index(id)

				elif isinstance(id, str):
					datum = self._get_datum_by_name(id)

				# else: TODO: Implement support for adv. indexing
			else:
				if isinstance(id, int):
					datum = self._get_unloaded_datum_by_index(id)

				elif isinstance(id, str):
					datum = self._get_unloaded_datum_by_name(id)

				# else: TODO: implement adv. indexing

			return datum

		except ValueError as ve:
			raise ve
		except IndexError:
			raise IndexError(f'The key ({id}) did not match an entry.')

	def __iter__(self) -> Iterator:
		return DatasetIterator(dataset=self)

	def _load(self, device: str) -> None:
		self._data = np.stack([cv2.imread(self.path / fname) for fname in self.metadata.fnames])
		self._to(device)

	def load(self, device: str) -> None:
		if hasattr(self, '_data') is False:
			self._load(device)
		elif device != self.device:
			self._to(device)
		else:
			print(f'Dataset is already loaded onto ({self._data.device})')

	def _to(self, device: str) -> None:
		match device:
			case 'cuda':
				self._data = cp.asarray(self._data)
			case 'cpu':
				self._data = cp.asnumpy(self._data)
			case _:
				raise ValueError(
					f"({device} is an invalid device. Choose from one of ['cpu', 'cuda'])"
				)

	def to(self, device: str) -> None:
		self.device = device if isinstance(device, str) else 'cpu'
		if self._data.device != device:
			self._to(device)
		else:
			print(f'Dataset is already loaded onto ({self._data.device})')


class DatasetIterator(Iterator):
	def __init__(self, dataset: Dataset):
		self._data = dataset._data if hasattr(dataset, '_data') else None
		self.ds_path = dataset.path
		self.items = dataset.metadata.items
		self._index = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self._index >= len(self.items):
			raise StopIteration

		fname, info = self.items[self._index]
		if self._data is None:  # if the data is not loaded into memory
			data = np.asnumpy(cv2.imread(self.ds_path / fname))
		else:
			data = self._data[self._index]

		self._index += 1
		return data, fname, info
