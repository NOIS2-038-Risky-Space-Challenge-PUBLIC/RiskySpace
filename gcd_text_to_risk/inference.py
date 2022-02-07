"""
GCD Text-to-Risk project
========================

Submodule: inference
"""

from .models import load, MODEL_1, MODEL_2, MODEL_3


def predict(text : str) -> dict:
    """
    """


def predict_model_1(text : str) -> dict:
    """
    """


def predict_model_2(text : str) -> dict:
    """
    """


def predict_model_3(text : str) -> dict:
    """
    """


if __name__ != '__main__':
    if any([MODEL_1 is None, MODEL_2 is None, MODEL_3 is None]):
        load()
