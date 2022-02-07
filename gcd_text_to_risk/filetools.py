"""
GCD Text-to-Risk project
========================

Submodule: filetools
"""


CSV_AS_DICT_LIST = 'csv_as_dict_list'
CSV_AS_LIST = 'csv_as_list'
CSV_AS_LIST_INCL_HEADER = 'csv_as_list_incl_header'


def extract_text(file_name : str) -> str:
    """
    Extract text from file
    ======================

    Parameters
    ----------
    file_name : str
        Name of the file to extract text from.

    Returns
    -------
    str
        The extracted text.
    """

    _extension = file_name.split('.')[-1].lower()
    result = ''
    if _extension in ['docx']:
        result = text_from_ms_document(file_name)
    elif _extension in ['pdf']:
        result = text_from_pdf(file_name)
    elif _extension in ['odt']:
        result = text_from_opendocument(file_name)
    else:
        result = text_from_textfile(file_name)
    return result


def read_csv(file_name : str, delimiter : str = '\t',
             result_as : str = CSV_AS_DICT_LIST) -> list:
    """
    Read CSV file content
    =====================

    Parameters
    ----------
    file_name : str
        Name of the file to read.
    delimiter : str, optional ('\t' if omitted)
        Delimiter character, if omitted TAB is used.
    result_as : str, optional (CSV_AS_DICT_LIST if omitted)
        Result type, this function will return result according to this setting.

    Returns
    -------
    list[dict] | list[list]
        List of CSV records. If format CSV_AS_DICT_LIST is selected, list
        entries will be dicts, where key is the column name. Else entries are
        lists which contains data of a row. If CSV_AS_LIST_INCL_HEADER is set,
        the 1st entry [0] is the lisf of column names and the first row is the
        2nd entry [1], else first row is the 1st entry [0].

    Raises
    ------
    ValueError
        If the parameter result_as is neither CSV_AS_DICT_LIST nor CSV_AS_LIST
        nor CSV_AS_LIST_INCL_HEADER.

    Notes
    -----
    I.
        This function assumes that the CSV file has a header.
    II.
        Python CSV reader handles each row entries as strings, if they are
        not string, the should be converted after reading.
    """

    from csv import reader as csv_reader

    result = []
    with open(file_name, 'r', encoding='utf_8') as instream:
        reader = csv_reader(instream, delimiter=delimiter)
        first_row = next(reader)
        if result_as == CSV_AS_DICT_LIST:
            for row in reader:
                result.append({k : v for k, v in zip(first_row, row)})
        elif result_as in [CSV_AS_LIST, CSV_AS_LIST_INCL_HEADER]:
            if result_as == CSV_AS_LIST_INCL_HEADER:
                result.append(first_row)
            for row in reader:
                result.append(row)
        else:
            raise ValueError('Unsupported result_as parameter.')
    return result


def text_from_ms_document(file_name : str) -> str:
    """
    Extract text from MS Office document file
    =========================================

    Parameters
    ----------
    file_name : str
        Name of the file to extract text from.

    Returns
    -------
    str
        The extracted text.
    """

    from docx import Document as Docx_Document

    text = ''
    document = Docx_Document(file_name)
    for paragraph in document.paragraphs:
        text += '{}\n'.format(paragraph.text)
    return text


def text_from_opendocument(file_name : str) -> str:
    """
    Extract text from OpenDocument file
    ===================================

    Parameters
    ----------
    file_name : str
        Name of the file to extract text from.

    Returns
    -------
    str
        The extracted text.
    """

    from odf import text as odf_text
    from odf.opendocument import load as odf_load

    content = odf_load(file_name)
    text = ''
    for paragraph in content.getElementsByType(odf_text.P):
        text += '{}\n'.format(paragraph)
    return text


def text_from_pdf(file_name : str) -> str:
    """
    Extract text from PDF file
    ==========================

    Parameters
    ----------
    file_name : str
        Name of the file to extract text from.

    Returns
    -------
    str
        The extracted text.
    """

    from PyPDF4 import PdfFileReader

    text = ''
    with open(file_name, 'rb') as instream:
        reader = PdfFileReader(instream)
        for i in range(reader.numPages):
            text += '{}\n'.format(reader.getPage(i).extractText())
    return text


def text_from_textfile(file_name : str, encoding : str = 'utf_8') -> str:
    """
    Extract text from text file
    ===========================

    Parameters
    ----------
    file_name : str
        Name of the file to extract text from.
    encoding : str, optional ('utf_8' if omitted)
        The encoding of the file.

    Returns
    -------
    str
        The extracted text.

    Notes
    -----
        For list of encoding types please consult:
        https://docs.python.org/3/library/codecs.html#standard-encodings
    """

    with open(file_name, 'r', encoding=encoding) as instream:
        content = instream.readlines()
    return ''.join(content)
