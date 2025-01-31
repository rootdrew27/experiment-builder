from pathlib import Path

class Metadata(object):
    def __init__(self, path:Path, fname2value:dict):
        self.path = path
        self.fname2data = fname2value
        self.fnames = list(fname2value.keys())
        self.items = list(fname2value.items())
        self.format = None

    def _getitem_by_index(self, id:int):
        fname, data = self.items[id]
        return fname, data

    def _getitem_by_name(self, id:str):
        fname, data = id, self.fname2data[id]
        return fname, data

    def __getitem__(self, id:str|int):
        if isinstance(id, int):
            item = self._getitem_by_index(id)
        if isinstance(id, str):
            item = self._getitem_by_name(id)
            # TODO: implement advanced idxing
        else:
            raise ValueError("Arguemtn for id must be one of [int, str]")
        
        return item
    
    def __len__(self):
        return len(self.fnames)
    
class CocoMetadata(Metadata):
    def __init__(self, path:Path, fname2value:dict, category_name2id:dict, all_tags:list[str]):
        super().__init__(path, fname2value)
        self.format = "coco"
        self.category_name2id = category_name2id
        self.all_tags = all_tags

