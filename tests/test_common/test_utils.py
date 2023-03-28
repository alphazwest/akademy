import os.path
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from akademy.common import utils
from akademy.common import project_paths
from akademy.common.utils import sample_ohlcv_data
from akademy.models import OHLCV


class TestFileHandlers(TestCase):
    """
    Tests functions in the common.filehandlers module.
    """
    def setUp(self) -> None:
        """
        Defines some class-based constants for test cases to utilize.
        """
        if not os.path.exists(project_paths.TMP_DIR):
            os.mkdir(project_paths.TMP_DIR)

        # mapping as file_and_extension: extension
        self.files_and_extensions = {
            "file": "file",
            "file.dat": "dat",
            "file.tar.gz": "tar.gz",
            "file.one.two.three.four": "one.two.three.four"
        }

        # create some tmp files
        self.old_version_filename = "old_file"
        self.old_version_file_ext = "old"
        self.old_version_file_key_remove = "old_file"
        self.old_version_file_key_keep = "keep"
        self.old_version_file_count = 10
        self.old_version_file_keep_count = 3
        self.old_version_files = []

    def _create_tmp_files(self):
        """
        Helper function to create some temporary files that may be used during
        misc. tests. Will all be deleted during tearDown function.
        """
        # flush files
        self.old_version_files = []
        self._delete_old_files()

        # create some tmp files
        for i in range(self.old_version_file_count):
            fname = f"{self.old_version_filename}_{i}.{self.old_version_file_ext}"
            output_fname = os.path.abspath(os.path.join(
                project_paths.TMP_DIR,
                fname
            ))
            self.old_version_files.append(output_fname)
            with open(output_fname, 'w')as file:
                file.write(f"temp file checkpoint_save_dir {i}\n")

    def _delete_old_files(self):
        """
        Delete all the old files in the TMP directory. Used to flush out file
        system during some tests.
        """
        for file in os.listdir(project_paths.TMP_DIR):
            if self.old_version_filename in file:
                os.remove(os.path.abspath(os.path.join(
                    project_paths.TMP_DIR,
                    file
                )))

    def _make_keeper_file(self) -> str:
        """
        makes a file that should be kept during deletions when the mask "keep"
        is used.
        """
        # create one keep file
        keeper = f'{self.old_version_filename}_keep.{self.old_version_file_ext}'
        keeper_fullpath = os.path.abspath(os.path.join(
            project_paths.TMP_DIR,
            keeper
        ))
        with open(keeper_fullpath, 'w') as kfile:
            kfile.write("keeper file should be kept.\n")

        return keeper_fullpath

    def tearDown(self) -> None:
        """
        Performs cleanup actions after all tests are ran:
            1. Delete tmp directory recursively.
        """
        shutil.rmtree(project_paths.TMP_DIR)

    def _validate_ohlcv_data(self, data: pd.DataFrame):
        """
        Helper method to validate e.g. OHLCV data
        """
        self.assertTrue(data is not None)
        self.assertTrue(type(data) == pd.DataFrame)
        self.assertEqual(len(data), 500)
        self.assertTrue('open' in data.columns)
        self.assertTrue('high' in data.columns)
        self.assertTrue('low' in data.columns)
        self.assertTrue('close' in data.columns)
        self.assertTrue('volume' in data.columns)
        self.assertTrue('date' in data.columns or data.index.name == "date")

    def test_format_float(self):
        """
        Test that the format_float function produces checkpoint_save_dir in the
        proper format.
        """
        initial = 123456.7891
        expected_rounded = f'123,456.79'
        expected_unrounded = "123,456.7891"

        # test that rounding mode works as expected
        self.assertEqual(
            utils.format_float(value=initial, do_rounding=True),
            expected_rounded
        )

        # test that un-rounded mode works as expected
        self.assertEqual(
            utils.format_float(value=initial, do_rounding=False),
            expected_unrounded
        )

    def test_get_file_extension(self):
        """
        Tests that all filenames result in the expected extension
        """
        # tests the expected checkpoint_save_dir extension is produced for all test cases.
        for file, ext in self.files_and_extensions.items():
            self.assertTrue(utils.get_file_extension(file) == ext)

    def test_remove_old_file_versions(self):
        """
        Test the removal of older versions of a file.
        """
        # creates tmp files, deleting previous if existing
        self._create_tmp_files()
        _count = len(self.old_version_files)

        keeper_file = self._make_keeper_file()

        # calls the function to handle removal of some files
        utils.remove_old_file_versions(
            filepath=self.old_version_files[0],  # first tmp file found
            remove_key=self.old_version_file_key_remove,
            keep_key=self.old_version_file_key_keep,
            keep_count=self.old_version_file_keep_count,
        )

        # test number of files removed is the expected amount
        files = [f for f in os.listdir(project_paths.TMP_DIR)
                 if self.old_version_filename in f
                 and self.old_version_file_key_keep not in f
                 ]
        self.assertTrue(len(files) == self.old_version_file_keep_count)

        # test there is a keep file still there
        self.assertTrue(os.path.exists(keeper_file))

    def test_load_csv_ohlcv_data(self):
        """
        Test that the sample data can be loaded and results in the expected
        DataFrame object of expected length and with the expected columns.
        """
        data = utils.load_csv_ohlcv_data(
            filepath=project_paths.SPY_DATA,
            count=500
        )
        self._validate_ohlcv_data(data=data)

    def load_spy_daily(self):
        """
        test that the loading of the SPY dataset is as expected
        """
        data = utils.load_spy_daily(count=500)
        self._validate_ohlcv_data(data=data)

    def test_minmax_normalize(self):
        """
        Test the normalization of data to the range o 0-1
        """
        data = np.array([1, 2, 3, 4, 5])
        n = utils.minmax_normalize(data=data)
        self.assertTrue(list(n) == list(np.array([0, .25, .5, .75, 1])))

        # these were produced with sklearn's minmax_scale to assure the
        # same functionality present after removing a dependency.
        floats = np.array([
            5.47859245e-04, 3.82998975e-01, 4.33758930e-01, 1.88521653e-02,
            4.26537772e-01, 2.68788025e-01, 5.63736560e-01, 7.03400592e-01,
            9.56029558e-01, 1.99128215e-01
        ])

        # intended normalized results
        floats_result = [
            0.0, 0.4002704774495804, 0.4533954667258172, 0.01915714982176074,
            0.44583785676907056, 0.28073815134765945, 0.5894290821988942,
            0.735600413561895, 1.0, 0.20783271517785398
        ]

        # normalize the floats and test
        n = utils.minmax_normalize(data=floats)
        self.assertTrue(list(n), floats_result)

    def test_sample_ohlcv_data(self):
        """Test generating some sample OHLCV data"""
        data = sample_ohlcv_data(10)
        self.assertTrue(len(data) == 10)
        self.assertTrue(type(data[0]) == OHLCV)
