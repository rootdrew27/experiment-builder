import unittest
import shutil
from pathlib import Path

from expb import build_dataset
from expb import VALID_FORMATS
from expb import download_dataset, extract_zip_dataset
from expb import LabelingPlatform
from expb.datasets.dataset import Dataset

test_zip_path = r".\tests\data\FOD Objects.v2i.coco-segmentation.zip"
test_data_dir = r".\tests\data\test_data_path"
test_bad_path = r"tests\data\sadkfjskf"
test_formats = VALID_FORMATS
test_name = "Test Dataset"


class TestDatasetExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = Path(test_data_dir)
        cls.test_data_path.mkdir(exist_ok=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_data_path)

    def test_extract_dataset(self):
        extract_zip_dataset(
            test_zip_path,
            test_data_dir,
            overwrite=False,
            labeling_platform=LabelingPlatform.ROBOFLOW,
        )
        self.assertEqual(len([path for path in Path(test_data_dir).iterdir() if path.is_dir()]), 0)


class TestDatasetCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_path = Path(test_data_dir)
        test_data_path.mkdir(exist_ok=False)
        extract_zip_dataset(
            test_zip_path,
            test_data_dir,
            overwrite=False,
            labeling_platform=LabelingPlatform.ROBOFLOW,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(test_data_dir)

    def test_build_dataset_coco(self):
        test_format = "coco-seg"

        ds = build_dataset(
            data_path=test_data_dir, format=test_format, is_split=False
        )

        self.assertIsInstance(ds, Dataset)

        self.assertEqual(ds.name, Path(test_data_dir).stem)
        self.assertEqual(ds.data, None)
        self.assertEqual(ds.path, Path(test_data_dir))
        self.assertEqual(len(ds), 8)

        with self.assertRaises(AssertionError):
            build_dataset(data_path=test_bad_path, format=test_format)


if __name__ == "__main__":
    unittest.main()
