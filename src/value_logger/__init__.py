import sys
from typing import Any, Callable
import os

import numpy
import numpy as np
import cupy as cp

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get('LOGLEVEL', 'INFO').upper()
)

class ValueLogger:
    # Singletons for each log file name
    loggers = {}

    def __init__(self, directory):
        """
        Logs values to a directory, behavior set by environment variables:
        COMPRTOL: relative tolerance for comparison.
        COMPATOL: absolute tolerance for comparison.
        LOGTO: directory to write log files to.
        COMPARE: directory read from and compare logged values to.
        PANICONFAIL: if set, raise an exception if comparison fails.
        FAILLOG: directory to write failed values to.
        OVERWRITECOMP: if set, overwrite values with values from COMPARE.
        LOGSUCC: if set, log success of comparison otherwise remain silent.
        LOGZFILL: zfill length for logging.
        :param directory: directory to log to or read from.
        """
        self.directory = directory
        self.count = 0
        self.log_values_to = None
        self.compare_from = None
        self.panic = False
        self.panic_fail_log = None
        self.rtol = float(os.environ.get("COMPRTOL", 1e-5))
        self.atol = float(os.environ.get("COMPATOL", 1e-8))

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
        self.overwrite_comp = bool(os.environ.get("OVERWRITECOMP"))
        self.logsucc = bool(os.environ.get("LOGSUCC"))
        self.zfill = int(os.environ.get("LOGZFILL",5))
        self.directory_whitelist = None
        self.key_whitelist = None
        if os.environ.get("LOGDIRWHITELIST") is not None:
            self.directory_whitelist = os.environ.get("LOGDIRWHITELIST").split(",")
        if os.environ.get("LOGKEYWHITELIST") is not None:
            self.key_whitelist = os.environ.get("LOGKEYWHITELIST").split(",")

    @staticmethod
    def reset():
        ValueLogger.loggers = {}

    @staticmethod
    def log_to_directory(directory: str, key: str, value: Any):
        if directory not in ValueLogger.loggers:
            ValueLogger.loggers[directory] = ValueLogger(directory)
        ValueLogger.loggers[directory].log(key, lambda: value)

    @staticmethod
    def log_result_to_directory(directory: str, key: str, value: Callable[[], Any]) -> np.ndarray:
        """
        Logs the result of the value callable.
        """
        if directory not in ValueLogger.loggers:
            ValueLogger.loggers[directory] = ValueLogger(directory)
        return ValueLogger.loggers[directory].log(key, value)

    @staticmethod
    def log_cupy(directory: str, key: str, value: cp.ndarray):
        """
        Logs a cupy array. If COMPARE and OVERWRITECOMP are set, overwrites value with data from the saved array.
        This allows the user to substitute in known correct values during runtime.
        """
        if directory not in ValueLogger.loggers:
            ValueLogger.loggers[directory] = ValueLogger(directory)
        logger = ValueLogger.loggers[directory]
        res = logger.log(key, lambda: value.get())
        if res is not None:
            logger.overwrite_with_compare(value, res)

    def overwrite_with_compare(self, value: cp.ndarray, comp_value: np.ndarray):
        if self.overwrite_comp:
            value[:] = cp.array(comp_value)
            assert np.allclose(value.get(), comp_value), "FAILED TO OVERWRITE"

    def log(self, key: str, value: Callable[[], np.ndarray]):
        if self.directory_whitelist is not None and self.directory not in self.directory_whitelist:
            return None
        if self.key_whitelist is not None and key not in self.key_whitelist:
            return None

        if self.log_values_to is None and self.compare_from is None:
            return None

        base_filename = f"{key}_{str(self.count).zfill(self.zfill)}.npz"
        self.count += 1

        value = value()
        if self.log_values_to is not None:
            filename = os.path.join(self.log_values_to, self.directory, base_filename)
            numpy.savez_compressed(filename, value=value)

        if self.compare_from is not None:
            comp_filename = os.path.join(self.compare_from, self.directory, base_filename)
            compare_value = numpy.load(comp_filename)['value']

            are_all_close = np.allclose(value, compare_value, rtol=self.rtol, atol=self.atol)
            if not are_all_close:
                if self.panic_fail_log is not None:
                    filename = os.path.join(self.panic_fail_log, self.directory, base_filename)
                    numpy.savez_compressed(filename, value=value)

                ddif = (value - compare_value)
                rel_ddif = ddif / compare_value
                max_ddif = np.abs(ddif).max()
                max_rel_ddif = np.abs(rel_ddif).max()
                if self.panic:
                    raise RuntimeError(f"Comparison for {self.directory}/{key} failed: {max_ddif:.3e}\trel: {max_rel_ddif:.3e}")
                else:
                    logging.debug(f"[-] Comparison for {self.directory}/{key} failed: {max_ddif:.3e}\trel: {max_rel_ddif:.3e}")
            elif self.logsucc:
                ddif = (value - compare_value)
                rel_ddif = ddif / compare_value
                max_ddif = np.abs(ddif).max()
                max_rel_ddif = np.abs(rel_ddif).max()
                logging.debug(f"[+] Comparison for {self.directory}/{key} succeeded: {max_ddif:.3e}\trel: {max_rel_ddif:.3e}")
            return compare_value

        return value