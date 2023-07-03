"""Module for API exceptions"""


class NotCachedError(Exception):
    """Exception raised when attempting to read data that should be cached,
    but is currently not cached"""
