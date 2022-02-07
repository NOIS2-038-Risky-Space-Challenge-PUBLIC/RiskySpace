"""
GCD Text-to-Risk project
========================

Submodule: textprocessing
"""

from json import load as json_load

from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

from .constants import PrintLevels, SYMBOLS_AND_DIGITS


BAG_OF_WORDS_PATH = './data/bag_of_words.json'
BAG_SIZE = 0
DEFAULT_LEMMATIZER = WordNetLemmatizer()
FILTERED_WORDS = []
FILTERED_WORDS_PATH = './data/filtered_words.json'
VEC_TO_WORD = {}
WORD_TO_VEC = {}


def get_bag_id(word : str) -> int:
    """
    Get bag of words ID of a word
    =============================

    Parameters
    ----------
    word : str
        Word to search in the bag for.

    Returns
    -------
    int
        The ID of the word or the ID 0 if the word isn't part of the bag.
    """

    result = 0
    if word in WORD_TO_VEC:
        result = WORD_TO_VEC[word]
    return result


def get_bag_word(bag_id : int) -> str:
    """
    Get word from the bag by ID
    ===========================

    Parameters
    ----------
    bag_id : int
        The ID to search for.

    Returns
    -------
    str
        The word that belongs to the ID, or the string <OTHER> if the word is
        not part of the bag of words.
    """


    result = VEC_TO_WORD[0]
    if bag_id in VEC_TO_WORD:
        result = VEC_TO_WORD[bag_id]
    return result


def load_bag_of_words():
    """
    Load beg of words data
    ======================
    """

    global BAG_SIZE
    global VEC_TO_WORD
    global WORD_TO_VEC


    with open(BAG_OF_WORDS_PATH, 'r', encoding='utf8') as instream:
        data = json_load(instream)
    VEC_TO_WORD = {int(k) : v for k, v in data.items()}
    WORD_TO_VEC = {v : k for k, v in VEC_TO_WORD.items()}
    BAG_SIZE = len(VEC_TO_WORD)


def load_filtered_words():
    """
    Load filtered words data
    ========================
    """

    global FILTERED_WORDS

    with open(FILTERED_WORDS_PATH, 'r', encoding='utf8') as instream:
        FILTERED_WORDS = json_load(instream)


def text_to_lemmas(text : str) -> list:
    """
    Convert text to lemmas
    ======================

    Parameters
    ----------
    text : str
        Text to convert.

    Returns
    -------
    list[str]
        List of lemmas in the text.

    Notes
    -----
        The output of this function fits the needs of vectorization, therefore
        only those words get lemmatized which are part of the bag of words, any
        other words will return the term <OTHER>.
    """

    def append_result():
        _element = DEFAULT_LEMMATIZER.lemmatize(element)
        if _element not in FILTERED_WORDS:
            result.append(_element)

    _text = unidecode(text).lower()
    element = ''
    result = []
    for character in _text:
        if character in SYMBOLS_AND_DIGITS:
            if element != '':
                append_result()
            element = ''
            result.append(character)
        elif character == ' ':
            if element != '':
                append_result()
                element = ''
        else:
            _ord = ord(character)
            if 122 >= _ord >= 97:
                element += character
    if element != '':
        append_result()
    return result


def text_to_vector(text : str) -> list:
    """
    Convert text to vector
    ======================

    Parameters
    ----------
    text : str
        Text to convert.

    Returns
    -------
    list[int]
        The text vector as list of integers.
    """

    data = text_to_lemmas(text)
    return [get_bag_id(w) for w in data]


def vector_to_lemmas(vector : list) -> list:
    """
    Convert vector to lemmas
    ========================

    Parameters
    ----------
    vector : list[int]
        Text vector as list of integers.

    Returns
    -------
    list[str]
        List of text lemmas (bag of words entries).
    """

    return [get_bag_word(i) for i in vector]


if __name__ != '__main__':
    load_bag_of_words()
    load_filtered_words()
