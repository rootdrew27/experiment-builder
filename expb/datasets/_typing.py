from typing import Union
from expb.datasets.metadata import Metadata, SegmMetadata


AnyMetadata = Union[Metadata, SegmMetadata]
MetadataWithLabel = Union[SegmMetadata,] # NOTE: This type implements a variety of functions: _get_label, _display_label, etc. (This should be rigorously defined and enforced).