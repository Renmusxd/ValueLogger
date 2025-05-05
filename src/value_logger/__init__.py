import sys
from typing import Any, Callable
import os

import numpy
import numpy as np


class ValueLogger:
    # Singletons for each log file name
    loggers = {}

    def __init__(self, directory):
        self.directory = directory
        self.count = 0
        self.log_values_to = None
        self.compare_from = None
        self.panic = False
        self.panic_fail_log = None
        if os.environ.get("LOGTO") is not None:
            self.log_values_to = os.environ.get("LOGTO")
            os.makedirs(os.path.join(self.log_values_to, self.directory), exist_ok=True)
        if os.environ.get("COMPARE") is not None:
            self.compare_from = os.environ.get("COMPARE")
            if os.environ.get("PANICONFAIL") is not None:
                self.panic = bool(os.environ.get("PANICONFAIL"))
        if os.environ.get("FAILLOG") is not None:
            self.panic_fail_log = os.environ.get("FAILLOG")
            os.makedirs(os.path.join(self.panic_fail_log, self.directory), exist_ok=True)
        self.zfill = 5


    @staticmethod
    def reset():
        ValueLogger.loggers = {}

    @staticmethod
    def log_to_directory(directory: str, key: str, value: Any):
        if directory not in ValueLogger.loggers:
            ValueLogger.loggers[directory] = ValueLogger(directory)
        ValueLogger.loggers[directory].log(key, lambda: value)

    @staticmethod
    def log_result_to_directory(directory: str, key: str, value: Callable[[], Any]):
        if directory not in ValueLogger.loggers:
            ValueLogger.loggers[directory] = ValueLogger(directory)
        ValueLogger.loggers[directory].log(key, value)

    def log(self, key: str, value: Callable[[], np.ndarray]):
        if self.log_values_to is None and self.compare_from is None:
            return

        base_filename = f"{key}_{str(self.count).zfill(self.zfill)}.npz"
        self.count += 1

        value = value()
        if self.log_values_to is not None:
            filename = os.path.join(self.log_values_to, self.directory, base_filename)
            numpy.savez_compressed(filename, value=value)

        if self.compare_from is not None:
            comp_filename = os.path.join(self.compare_from, self.directory, base_filename)
            compare_value = numpy.load(comp_filename)['value']

            are_all_close = np.allclose(value.flatten(), compare_value.flatten())
            if not are_all_close:
                if self.panic_fail_log is not None:
                    filename = os.path.join(self.panic_fail_log, self.directory, base_filename)
                    numpy.savez_compressed(filename, value=value)

                ddif = (value - compare_value)
                low, high, mean = ddif.min(), ddif.max(), np.absolute(ddif).mean()
                if self.panic:
                    raise RuntimeError(f"Log comparison for {key} failed: {low:.6f}\t{high:.6f}\t{mean:.6f}")
                else:
                    print(f"Log comparison for {key} failed: {low:.6f}\t{high:.6f}\t{mean:.6f}", file=sys.stderr)


