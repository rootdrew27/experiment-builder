import unittest
import shutil
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path

from expb import build_dataset
from expb import VALID_FORMATS
from expb import extract_zip_dataset
from expb import LabelingPlatform
from expb.datasets.dataset import Dataset

test_zip_path = r".\tests\data\FOD Objects.v2i.coco-segmentation.zip"

test_bad_path = r"tests\data\sadkfjskf"
test_formats = VALID_FORMATS
test_name = "Test Dataset"

text_trap = io.StringIO() # used with redirect_stdout() to suppress print statements


class TestDatasetExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = r".\tests\data\test_data_path1"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_data_dir)

    def test_extract_dataset(self):
        test_data_dir = TestDatasetExtraction.test_data_dir

        with redirect_stdout(text_trap):
            extract_zip_dataset(
                test_zip_path,
                test_data_dir,
                overwrite=False,
                labeling_platform=LabelingPlatform.ROBOFLOW,
            )

        self.assertEqual(
            len([path for path in Path(test_data_dir).iterdir() if path.is_dir()]), 0
        )


class TestDatasetCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = r".\tests\data\test_data_path2"

        with redirect_stdout(text_trap):
            extract_zip_dataset(
                test_zip_path,
                cls.test_data_dir,
                overwrite=False,
                labeling_platform=LabelingPlatform.ROBOFLOW,
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_data_dir)

    def test_build_dataset_coco_no_splits(self):
        test_format = "coco-seg"
        test_data_dir = TestDatasetCreation.test_data_dir

        ds = build_dataset(data_dir=test_data_dir, format=test_format, is_split=False)

        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.name, Path(test_data_dir).stem)
        self.assertEqual(ds.path, Path(test_data_dir))
        self.assertEqual(len(ds), 8)

        with self.assertRaises(
            AssertionError,
            msg=f"The data_path: {test_data_dir} is empty! It must contain images and annotations file.",
        ):
            build_dataset(data_dir=test_bad_path, format=test_format)

        with self.assertRaises(
            AssertionError,
            msg="The parameter is_split was set to True but only 0 splits were identified.",
        ):
            build_dataset(data_dir=test_data_dir, format=test_format, is_split=True)


if __name__ == "__main__":
    unittest.main()
