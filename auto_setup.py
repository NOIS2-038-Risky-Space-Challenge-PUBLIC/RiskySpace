#!/usr/bin/env python3
"""
GCD Text-to-Risk project
========================

Software: auto_setup
"""

from os import makedirs
from os.path import exists

from gcd_text_to_risk.constants import MODELS_DIR, PrintLevels
from gcd_text_to_risk.functions import consoleprint, module_setup
from gcd_text_to_risk.textprocessing import BAG_OF_WORDS_PATH, FILTERED_WORDS_PATH


def main():
    """
    Provide main functionality
    ==========================
    """

    error_count = 0
    if not exists(MODELS_DIR):
        makedirs(MODELS_DIR)
        consoleprint('Folder for models\' state dicts are created.')
    else:
        consoleprint('Folder for models\' state dicts already exists.')
    consoleprint('Setting up WordNet tokenizer...')
    module_setup()
    consoleprint('Done.')
    if exists(BAG_OF_WORDS_PATH):
        consoleprint('Data for "bag of words" found.')
    else:
        consoleprint('Data for "bag of words" not found. Please download the' +
                     ' repository again.', PrintLevels.WARNING)
        error_count += 1
    if exists(FILTERED_WORDS_PATH):
        consoleprint('Data for "filtered words" found.')
    else:
        consoleprint('Data for "filtered words" not found. Please download ' +
                     'the repository again.', PrintLevels.WARNING)
        error_count += 1
    consoleprint('Software "auto_setup" ends with {} error(s)'
                 .format(error_count))


if __name__ == '__main__':
    main()
else:
    consoleprint('This is a software from the GCD Text-to-Risk project, not a ' +
                 'module.', PrintLevels.WARNING)
