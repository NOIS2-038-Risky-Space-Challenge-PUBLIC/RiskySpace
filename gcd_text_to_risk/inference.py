"""
GCD Text-to-Risk project
========================

Submodule: inference
"""


import torch

from .functions import get_inital_wrapper
from .models import load_model, Models
from .textprocessing import text_to_vector

def predict(text : str, allow_cuda : bool = False) -> dict:
    """
    Make prediction with all models
    ===============================

    Parameters
    ----------
    text : str
        Text to use for prediction.
    allow_cuda : bool = False
        Whether to allow CUDA if available or not.

    Returns
    -------
    dict[dict[key:list]]
        The result of the inference.
    """

    if allow_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    result = get_inital_wrapper()
    if any([Models.MODEL_1 is None, Models.MODEL_2 is None,
            Models.MODEL_3 is None]):
        load_model()
    Models.MODEL_1.to(device)
    Models.MODEL_2.to(device)
    Models.MODEL_3.to(device)
    Models.MODEL_1.eval()
    Models.MODEL_2.eval()
    Models.MODEL_3.eval()
    _text = torch.tensor(text_to_vector(text), dtype=torch.int64).to(device)
    _offset = torch.tensor([0], dtype=torch.int64).to(device)
    result['model_1']['inference'] = Models.MODEL_1(_text, _offset)
    result['model_2']['inference'] = \
        Models.MODEL_2(_text, _offset,
                result['model_1']['inference']).detach().cpu().numpy().tolist()
    result['model_3']['inference'] = \
                        Models.MODEL_3(_text,
                                       _offset).detach().cpu().numpy().tolist()
    result['model_1']['inference'] = \
                result['model_1']['inference'].detach().cpu().numpy().tolist()
    return result


def predict_model_1(text : str, allow_cuda : bool = False) -> dict:
    """
    Make prediction with model 1
    ============================

    Parameters
    ----------
    text : str
        Text to use for prediction.
    allow_cuda : bool = False
        Whether to allow CUDA if available or not.

    Returns
    -------
    dict[dict[key:list]]
        The result of the inference.
    """

    if allow_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    result = get_inital_wrapper()
    if Models.MODEL_1 is None:
        load_model('model_1')
    Models.MODEL_1.to(device)
    Models.MODEL_1.eval()
    _text = torch.tensor(text_to_vector(text), dtype=torch.int64).to(device)
    _offset = torch.tensor([0], dtype=torch.int64).to(device)
    result['model_1']['inference'] = \
                        Models.MODEL_1(_text,
                                       _offset).detach().cpu().numpy().tolist()
    return result


def predict_model_2(text : str, allow_cuda : bool = False) -> dict:
    """
    Make prediction with model 2
    ============================

    Parameters
    ----------
    text : str
        Text to use for prediction.
    allow_cuda : bool = False
        Whether to allow CUDA if available or not.

    Returns
    -------
    dict[dict[key:list]]
        The result of the inference.
    """

    if allow_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    result = get_inital_wrapper()
    if any([Models.MODEL_1 is None, Models.MODEL_2 is None]):
        load_model(['model_1', 'model_2'])
    Models.MODEL_1.to(device)
    Models.MODEL_2.to(device)
    Models.MODEL_1.eval()
    Models.MODEL_2.eval()
    _text = torch.tensor(text_to_vector(text), dtype=torch.int64).to(device)
    _offset = torch.tensor([0], dtype=torch.int64).to(device)
    model_1_outputs = Models.MODEL_1(_text, _offset)
    result['model_2']['inference'] = \
        Models.MODEL_2(_text, _offset,
                       model_1_outputs).detach().cpu().numpy().tolist()
    return result


def predict_model_3(text : str, allow_cuda : bool = False) -> dict:
    """
    Make prediction with model 3
    ============================

    Parameters
    ----------
    text : str
        Text to use for prediction.
    allow_cuda : bool = False
        Whether to allow CUDA if available or not.

    Returns
    -------
    dict[dict[key:list]]
        The result of the inference.
    """

    if allow_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    result = get_inital_wrapper()
    if Models.MODEL_3 is None:
        load_model('model_3')
    Models.MODEL_3.to(device)
    Models.MODEL_3.eval()
    _text = torch.tensor(text_to_vector(text), dtype=torch.int64).to(device)
    _offset = torch.tensor([0], dtype=torch.int64).to(device)
    with torch.no_grad():
        result['model_3']['inference'] = \
                        Models.MODEL_3(_text,
                                       _offset).detach().cpu().numpy().tolist()
    return result
