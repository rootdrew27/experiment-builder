import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict
from unittest import TestCase

import numpy as np
import cupy as cp  # type: ignore
from numpy.testing import assert_array_equal
from torch import Tensor

from expb import VALID_FORMATS, LabelingPlatform, build_dataset, extract_zip_dataset
from expb.datasets.by import By
from expb.datasets.dataset import Dataset
from expb.datasets.metadata import SegmMetadata

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
        self.ds: Dataset = build_dataset(
            data_dir=self.test_data_dir,
            format=self.test_format,
            task=self.test_task,
            is_split=False,
        )

    def test_build_dataset_coco_no_splits(self) -> None:
        self.assertIsInstance(self.ds, Dataset)
        self.assertIsInstance(self.ds._data, np.memmap)
        self.assertEqual(self.ds.name, Path(self.test_data_dir).stem)
        self.assertEqual(self.ds.path, Path(self.test_data_dir))
        self.assertEqual(len(self.ds), 8)
        self.assertIsInstance(self.ds.metadata, SegmMetadata)
        self.assertEqual(
            len(self.ds.metadata.categoryname2id), 3
        )  # check the number of categories. Super categories are ignored, but a background class is added.
        self.assertEqual(self.ds.metadata.path, Path(self.test_data_dir) / "_annotations.coco.json")
        self.assertEqual(len(self.ds.metadata.all_tags), 1)

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
            build_dataset(
                data_dir=self.test_data_dir,
                format=self.test_format,
                task=self.test_task,
                is_split=True,
            )

        with self.assertRaises(AssertionError):
            build_dataset(data_dir=self.test_data_dir, format=self.test_format, task="bbox")

    def test_memmapping(self):
        with self.assertRaises(Exception):  # the _data attribute is read only
            self.ds._data[0] = 10

    def test_caching(self):
        self.assertIsInstance(self.ds._data, np.memmap)

        self.ds.cache()

        self.assertNotIsInstance(self.ds._data, np.memmap)
        self.assertIsInstance(self.ds._data, np.ndarray)

        self.ds.cache(False)

        self.assertIsInstance(self.ds._data, np.memmap)

    def test_dataset_iteration(self):
        for i, datum in enumerate(self.ds):
            self.assertEqual(len(datum), 3)
            data, fname, info = datum
            self.assertIsInstance(
                data, np.ndarray
            )  # Data accessed through iteration and indexing is converted to np.ndarray
            self.assertIsInstance(fname, str)
            self.assertIsInstance(info, Dict)

            data1, fname1, info1 = self.ds[fname]
            assert_array_equal(data, data1)
            self.assertIs(info, info1)
            self.assertIs(fname, fname1)

            data2, fname2, info2 = self.ds[i]
            assert_array_equal(data, data2)
            self.assertIs(info, info2)
            self.assertIs(fname, fname2)

    def test_torch_support(self):
        self.ds.torch()
        self.assertTrue(self.ds.for_torch)

        data, label = self.ds[0]
        self.assertIsInstance(data, Tensor)
        self.assertIsInstance(label, np.ndarray)

        # TODO: Test with Dataloader

    def test_setting_cls_hieracrchy(self):
        self.assertEqual(len(self.ds.metadata.categoryname2id), 3)
        # define category hierarchy
        cat_hierarchy = {"misc": 1, "FO": 2}  # background class will be automatically set to 0
        self.ds.metadata.set_category_hierarchy(hierarchy=cat_hierarchy)
        self.assertEqual(len(self.ds.metadata.categoryname2id), 3)
        label_1 = self.ds.get_label(1)
        self.assertIn(0, label_1)
        self.assertIn(2, label_1)
        label_3 = self.ds.get_label(3)
        self.assertIn(1, label_3)
        self.assertIn(2, label_3)

    def test_subset(self):
        has_misc = self.ds.subset(by=By.CATEGORY, value="misc")
        self.assertIn(
            "Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg",
            has_misc.metadata.fnames,
        )
        self.assertIn(
            "BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg",
            has_misc.metadata.fnames,
        )
        self.assertEqual(len(has_misc), 2)

        no_misc = self.ds.subset(by=By.CATEGORY, value="misc", complement=True)
        self.assertNotIn(
            "Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg",
            no_misc.metadata.fnames,
        )
        self.assertNotIn(
            "BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg",
            no_misc.metadata.fnames,
        )
        self.assertEqual(len(no_misc), 6)

        has_misc, no_misc = self.ds.subset(by=By.CATEGORY, value="misc", return_both=True)
        self.assertIn(
            "Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg",
            has_misc.metadata.fnames,
        )
        self.assertIn(
            "BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg",
            has_misc.metadata.fnames,
        )
        self.assertEqual(len(has_misc), 2)
        self.assertNotIn(
            "Bolt1a_frame_000086_png.rf.6fa64bafe1c80a9c973c27ae4cb1d8bb.jpg",
            no_misc.metadata.fnames,
        )
        self.assertNotIn(
            "BoltWasher3_frame_000055_png.rf.aae08f33784ac37fdfb69b7c23d34258.jpg",
            no_misc.metadata.fnames,
        )
        self.assertEqual(len(no_misc), 6)

        exlcudes_ignore = self.ds.subset(by=By.TAG, value="ignore", complement=True)
        self.assertEqual(len(exlcudes_ignore), 7)
        self.assertNotIn(
            "ClampPart1a_frame_000032_png.rf.40aed52812528fa29028b95931043e8b.jpg",
            exlcudes_ignore.metadata.fnames,
        )

    def test_split(self):
        train_ds, test_ds = self.ds.split(split_fracs=[0.8, 0.2], shuffle=False)
        self.assertEqual(len(train_ds), 7)
        self.assertEqual(len(test_ds), 1)
        self.assertIn(
            "Cutter3_frame_000088_png.rf.cbbb3bc149fbf20e2476aca1c3e7656a.jpg",
            test_ds.metadata.fnames,
        )

        train_ds, test_ds = self.ds.split(split_fracs=[0.5, 0.5], shuffle=False, random_seed=0)
        self.assertEqual(len(train_ds), len(test_ds))

        with self.assertRaises(AssertionError):
            _, _ = self.ds.split(split_fracs=[0.8, 0.15], shuffle=False)

    def test_actions(self):

        def rgb2gray(data, weights):
            xp = cp.get_array_module(data)
            if xp is cp:
                weights = cp.asarray(weights)
            return xp.dot(data, weights)

        # create parameters for each method of passing parameters
        kw_params = {"weights": [0.2125, 0.7154, 0.0721]}
        params = ([0.2125, 0.7154, 0.0721],)

        # test rgb2gray action
        self.ds.apply(rgb2gray, kw_params=kw_params)
        # verify that the action has NOT occurred yet
        self.assertTupleEqual(
            self.ds._get_data(0).shape, (400, 400, 3)
        )
        ds_gray = self.ds.execute(return_dataset=True)
        # verify that the action has occurred
        self.assertIsInstance(ds_gray, Dataset)
        self.assertTupleEqual(
            ds_gray._get_data(0).shape, (400, 400)
        )
        # verify that the action did not affect the original dataset
        self.assertTupleEqual(
            self.ds._get_data(0).shape, (400, 400, 3)
        )
        # check to see if action_queue is empty after execute call
        with self.assertRaises(AssertionError):
            self.ds.execute(return_dataset=True)

        # test on gpu
        self.ds.to("cuda")
        ds_gray2 = self.ds.apply(rgb2gray, params=params).execute(True)
        # verify that the action has occurred
        self.assertTupleEqual(
            ds_gray2._get_data(0).shape, (400, 400)
        )
        # verify that new dataset is on CPU
        self.assertEqual(ds_gray2._data.device, "cpu") 
        self.ds.to("cpu")

        # test returning data (rather than dataset)
        data = self.ds.apply(rgb2gray, params=params).execute(return_dataset=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertTupleEqual(data[0].shape, (400, 400))  # verify action occurred


if __name__ == "__main__":
    unittest.main()
