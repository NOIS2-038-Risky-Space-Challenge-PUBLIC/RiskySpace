"""
GCD Text-to-Risk project
========================

Submodule: constants
"""

from enum import Enum


DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MODELS_DIR = './models'
SYMBOLS = ['.', ',', '?', '!', ':', ';', '$', '%', '&', '(', ')', '*', '+', '-',
           '/', '<', '=', '>', '@', '"', '\'']
SYMBOLS_AND_DIGITS = SYMBOLS + DIGITS


class RiskCategories(Enum):

    FINANCIAL = 0
    NATURAL = 1
    PROGRAMMATIC = 2
    RESOURCE = 3
    SAFETY = 4
    SCHEDULE = 5
    SCIENTIFIC = 6
    SOCIAL = 7
    TECHNICAL = 8

class PrintLevels(Enum):

    INFO = '[.]'
    WARNING = '[!]'
