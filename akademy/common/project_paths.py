"""
This file defines some common filepaths used across many functions.
"""
import os

# defines top-level paths and current directory for reference
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))

# defines secondary paths used misc. throughout project files
TMP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "tmp"))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
SPY_DATA = os.path.abspath(os.path.join(DATA_DIR, 'SPY.csv'))
