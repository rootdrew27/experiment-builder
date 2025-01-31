from expb import build_dataset
from expb import VALID_FORMATS


SAMPLE_DATA_PATH = r""
FORMATS = VALID_FORMATS
NAME = "Test Dataset"

def build_dataset_test():
    # Test the creation of a dataset with a data path as a str.
    # Test all valid formats
    # Test with and without name
    # Verfiy that Metadata attributes are set properly
    # Verify that Dataset attributes are set properly
    for format in VALID_FORMATS:
        ds = build_dataset()
        assert ds.name == NAME, f"The ds.name attribute should equal NAME but ds.name={ds.name}"
        # TODO: complete