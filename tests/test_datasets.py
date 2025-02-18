import unittest
from unittest import TestCase
import shutil
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
import numpy as np
from numpy.testing import assert_array_equal

from expb import build_dataset
from expb import VALID_FORMATS
from expb import extract_zip_dataset
from expb import LabelingPlatform
from expb.datasets.dataset import Dataset
from expb.datasets.metadata import SegmMetadata
from expb.datasets.by import By

test_zip_path = r".\tests\data\FOD Objects.v5i.coco-segmentation.zip"

test_bad_path = r"tests\data\sadkfjskf"
test_formats = VALID_FORMATS
test_name = "Test Dataset"

text_trap = io.StringIO()  # used with redirect_stdout() to suppress print statements


# TODO: Move this Test Case to a utility test file
class TestDatasetExtraction(TestCase):
    test_data_dir = r".\tests\data\test_data_path1"

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_data_dir)

    def test_extract_dataset(self) -> None:
        test_data_dir = TestDatasetExtraction.test_data_dir

        with redirect_stdout(text_trap):
            extract_zip_dataset(
                test_zip_path,
                test_data_dir,
                overwrite=False,
                labeling_platform=LabelingPlatform.ROBOFLOW,
            )

        self.assertEqual(len([path for path in Path(test_data_dir).iterdir() if path.is_dir()]), 0)


class TestDatasetWithSegmMetadata(TestCase):
    test_data_dir = r".\tests\data\test_data_path2"
    test_format = "coco"
    test_task = "segm"

    @classmethod
    def setUpClass(cls) -> None:
        with redirect_stdout(text_trap):
            extract_zip_dataset(
                test_zip_path,
                cls.test_data_dir,
                overwrite=False,
                labeling_platform=LabelingPlatform.ROBOFLOW,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_data_dir)

    def setUp(self):
        self.test_data_dir = TestDatasetWithSegmMetadata.test_data_dir
        self.test_format = TestDatasetWithSegmMetadata.test_format
        self.test_task = TestDatasetWithSegmMetadata.test_task

    def test_build_dataset_coco_no_splits(self) -> None:

        ds: Dataset = build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=False)

        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.name, Path(self.test_data_dir).stem)
        self.assertEqual(ds.path, Path(self.test_data_dir))
        self.assertEqual(len(ds), 8)
        self.assertIsInstance(ds.metadata, SegmMetadata)
        self.assertEqual(
            len(ds.metadata.categoryname2id), 3
        )  # check the number of categories. Super categories are ignored, but a background class is added.
        self.assertEqual(ds.metadata.path, Path(self.test_data_dir) / "_annotations.coco.json")
        self.assertEqual(len(ds.metadata.all_tags), 1)

        # TODO: Check metadata info on each image (tags, categories, segmentations). Perhaps use a config file to load the expected values in.

        with self.assertRaises(
            AssertionError,
            msg=f"The data_path: {self.test_data_dir} is empty! It must contain images and annotations file.",
        ):
            build_dataset(data_dir=test_bad_path, format=self.test_format, task=self.test_task)

        with self.assertRaises(
            AssertionError,
            msg="The parameter is_split was set to True but only 0 splits were identified.",
        ):
            build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=True)

        with self.assertRaises(AssertionError):
            build_dataset(data_dir=self.test_data_dir, format=self.test_format, task="bbox")

    def test_dataset_iteration(self):

        ds: Dataset = build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=False)

        for i, datum in enumerate(ds):
            self.assertEqual(len(datum), 3)
            data, fname, info = datum
            self.assertIsInstance(data, np.ndarray)
            self.assertIsInstance(fname, str)
            self.assertIsInstance(info, Dict)

            data1, fname1, info1 = ds[fname]
            assert_array_equal(data, data1)
            self.assertIs(info, info1)
            self.assertIs(fname, fname1)

            data2, fname2, info2 = ds[i]
            assert_array_equal(data, data2)
            self.assertIs(info, info2)
            self.assertIs(fname, fname2)

    def test_torch_support(self):

        ds: Dataset = build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=False)

        ds.torch()
        self.assertTrue(ds.for_torch)

        data, label = ds[0]
        self.assertIsInstance(data, Tensor)
        self.assertIsInstance(label, np.ndarray)

        # TODO: Test with Dataloader

    def test_setting_cls_hieracrchy(self):

        ds: Dataset = build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=False)
        self.assertEqual(len(ds.metadata.categoryname2id), 3)
        # define category hierarchy
        cat_hierarchy = {"misc": 1, "FO": 2} # background class will be automatically set to 0
        ds.metadata.set_category_hierarchy(hierarchy=cat_hierarchy)
        self.assertEqual(len(ds.metadata.categoryname2id), 3)
        label_0 = ds.get_label(6)
        self.assertIn(0, label_0)
        self.assertIn(1, label_0)
        self.assertIn(2, label_0)

    def test_subset(self):

        ds: Dataset = build_dataset(data_dir=self.test_data_dir, format=self.test_format, task=self.test_task, is_split=False)

        has_misc = ds.subset(by=By.CATEGORY, value="misc")
        self.assertIn("Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg", has_misc.metadata.fnames)
        self.assertIn("BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg", has_misc.metadata.fnames)
        self.assertEqual(len(has_misc), 2)

        no_misc = ds.subset(by=By.CATEGORY, value="misc", complement=True)
        self.assertNotIn("Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg", no_misc.metadata.fnames)
        self.assertNotIn("BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg", no_misc.metadata.fnames)
        self.assertEqual(len(no_misc), 6)

        has_misc, no_misc = ds.subset(by=By.CATEGORY, value="misc", return_both=True)
        self.assertIn("Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg", has_misc.metadata.fnames)
        self.assertIn("BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg", has_misc.metadata.fnames)
        self.assertEqual(len(has_misc), 2)
        self.assertNotIn("Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg", no_misc.metadata.fnames)
        self.assertNotIn("BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg", no_misc.metadata.fnames)
        self.assertEqual(len(no_misc), 6)


if __name__ == "__main__":
    unittest.main()
