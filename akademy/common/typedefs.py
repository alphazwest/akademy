"""
This file contains a listing of different type definitions used throughout
the project.
"""
from typing import TypeVar

# input features to Neural Networks
State = TypeVar("State")

# values of checkpoint_save_dir features from Neural Networks (discrete or continuous)
Action = TypeVar("Action", int, float)
