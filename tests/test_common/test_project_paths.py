import os.path
from unittest import TestCase
from akademy.common import project_paths


class TestProjectPaths(TestCase):
    """
    Runs o suite of tests to ensure all essential project paths are defined
    and present.
    """
    def setUp(self) -> None:
        """
        Creates a tmp directory if one isn't there already for testing and,
        since it's a tmp directory and can't be depended on being, or not being,
        doesn't bother cleaning it up at all.
        """
        tmp = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'akademy',
            'tmp'
        ))
        if not os.path.exists(tmp):
            os.mkdir(tmp)

    def test_project_paths(self):
        """
        tests that project paths exist.
        """
        self.assertTrue(
            os.path.exists(project_paths.PROJECT_ROOT)
        )
        self.assertTrue(
            os.path.exists(project_paths.DATA_DIR)
        )
        self.assertTrue(
            os.path.exists(project_paths.TMP_DIR)
        )
        self.assertTrue(
            os.path.exists(project_paths.SPY_DATA)
        )
