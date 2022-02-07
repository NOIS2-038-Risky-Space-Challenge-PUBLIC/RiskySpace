"""
GCD Text-to-Risk project
========================

Submodule: functions
"""

from time import gmtime, strftime

from .constants import PrintLevels


def consoleprint(text : str, level : PrintLevels = PrintLevels.INFO,
                 add_timestamp : bool = True):
    """
    Print log alike text to the standard console output
    ===================================================

    Parameters
    ----------
    text : str
        Text to print.
    level : PrintLevels, optional (PrintLevels.INFO if omitted)
        Log level alike prefix to use. By default it prints info prefix.
    add_timestamp : bool, optional (True if omitted)
        Whether to add timestamp to the log or not.

    Notes
    -----
        Time format is based on this documentation:
        https://pds.nasa.gov/datastandards/pds3/standards/sr/stdref3.7/Chapter07.pdf
    """

    if add_timestamp:
        print('{} {} {}'.format(level.value,
                                strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()),
                                text))
    else:
        print('{} {}'.format(level.value, text))


def get_inital_wrapper() -> dict:
    """
    Get an empty wrapper for dataset or result
    ==========================================
    """

    return {'model_1' : {}, 'model_2' : {}, 'model_3' : {}}


def module_setup():
    """
    Check whether wordnet dataset is downloaded
    ===========================================
    """

    from .textprocessing import DEFAULT_LEMMATIZER

    try:
        DEFAULT_LEMMATIZER.lemmatize('NASA')
    except LookupError:
        consoleprint('Wordnet database should be downloaded.',
                     PrintLevels.WARNING)
        from nltk import download as nltk_download
        nltk_download('wordnet')
        consoleprint('Wordnet database has been downloaded.')
