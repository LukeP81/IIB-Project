"""Module for utility functions and classes"""
from enum import Enum


class StateEnum(Enum):
    """Class for holding possible states of the application"""

    INITIALISING = "initialising"
    OPTIMISING = "optimising"
    INTERPRETING = "interpreting"
