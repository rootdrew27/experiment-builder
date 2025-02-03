from pathlib import Path


class Metadata(object):
	def __init__(self, path: Path, fname2info: dict):
		self.path = path
		self.fname2info = fname2info
		self.fnames = list(fname2info.keys())
		self.items = list(fname2info.items())
		self.format: str

	def _getitem_by_index(self, id: int) -> tuple[str, dict]:
		fname, info = self.items[id]
		return fname, info

	def _getitem_by_name(self, id: str) -> tuple[str, dict]:
		fname, info = id, self.fname2info[id]
		return fname, info

	def __getitem__(self, id: str | int) -> tuple[str, dict]:
		if isinstance(id, int):
			item = self._getitem_by_index(id)
		elif isinstance(id, str):
			item = self._getitem_by_name(id)
			# TODO: implement advanced idxing
		else:
			raise ValueError('Arguement for id must be one of [int, str]')

		return item

	def __len__(self) -> int:
		return len(self.fnames)


class CocoMetadata(Metadata):
	"""An object to store and manage the metadata of a dataset. The CocoMetadata.fname2info has the following format:
	```
	{
	    <file_name:str> : {
	        "width": <width:int>,
	        "height": <height:int>,
	        "tags": <tags_on_file:list[str]>,
	        "categories": <categories_of_segmentations:list[int]>,
	        "segmentations": <collections_of_segmentation_points:list[list[list[float]]]>,
	    }
	    .
	    .
	}
	"""

	def __init__(self, path: Path, fname2info: dict, category_name2id: dict, all_tags: list[str]):
		super().__init__(path, fname2info)
		self.format = 'coco'
		self.category_name2id = category_name2id
		self.all_tags = all_tags
