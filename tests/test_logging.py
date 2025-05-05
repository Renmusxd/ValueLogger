import unittest

import numpy as np

from src.value_logger import ValueLogger
import os
import tempfile


class MyTestCase(unittest.TestCase):
    def test_simple_logger(self): # add assertion here
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ.setdefault("LOGTO", tmpdir)

            ValueLogger.log_to_directory("d", "x", np.array([1,2,3]))

            ValueLogger.reset()
            os.environ.setdefault("LOGTO", tmpdir)
            os.environ.setdefault("COMPARE", tmpdir)

            ValueLogger.log_to_directory("d", "x", np.array([1,2,3]))
        ValueLogger.reset()

    def test_simple_logger_fail(self): # add assertion here
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ.setdefault("LOGTO", tmpdir)

            ValueLogger.log_to_directory("d", "x", np.array([1,2,3]))

            ValueLogger.reset()
            os.environ.setdefault("LOGTO", tmpdir)
            os.environ.setdefault("COMPARE", tmpdir)

            try:
                ValueLogger.log_to_directory("d", "x", np.array([1,2,3,4]))
                ValueLogger.reset()
                raise RuntimeError("Expected Exception")
            except RuntimeError:
                ValueLogger.reset()

if __name__ == '__main__':
    unittest.main()
