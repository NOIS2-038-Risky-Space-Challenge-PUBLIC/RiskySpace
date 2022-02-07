#!/usr/bin/env python3
"""
GCD Text-to-Risk project
========================

Software: inference
"""

from json import dump as json_dump
from os.path import isfile
from sys import exit

import click

from gcd_text_to_risk.constants import PrintLevels
from gcd_text_to_risk.filetools import extract_text
from gcd_text_to_risk.functions import consoleprint
from gcd_text_to_risk.inference import (predict, predict_model_1,
                                        predict_model_2, predict_model_3)

@click.command()
@click.option('--file', '-F', 'file_name', default='',
              help='File ([path]/name) to input text.', type=str)
@click.option('--text', '-T', 'text_content', default='',
              help='Direct text input.', type=str)
@click.option('--models', '-M', 'models_list', default=[],
              help='List of models to inference on.', multiple=True, type=str)
@click.option('--result', '-R', 'result_file', default='',
              help='Result JSON file lcoation. If omitted result.json is used.',
              type=str)
def main(file_name : str, text_content : str, models_list : tuple,
         result_file : str):
    """
    Software to perform GCD Text-to-Risk inference on text from command line or file.
    """

    known_models = ['model_1', 'model_2', 'model_3']

    _texts = {}
    if file_name == '' and text_content == '':
        consoleprint('No text for inference, please use --file or --text.',
                     PrintLevels.WARNING)
        print_help_and_exit()
    if file_name != '':
        if isfile(file_name):
            _texts['file_content'] = extract_text(file_name)
        else:
            consoleprint('None-existing file is given.', PrintLevels.WARNING)
            print_help_and_exit()
    if text_content != '':
        _texts['text_content'] = text_content
    _models = []
    if len(models_list) == 0:
        _models = known_models
    else:
        for model_name in models_list:
            if model_name not in known_models:
                consoleprint('Unkown model "{}" is given.'.format(model_name),
                PrintLevels.WARNING)
                print_help_and_exit()
            else:
                _models.append(model_name)
        if len(_models) != len(set(_models)):
            consoleprint('At least one model is given at least twice.',
                         PrintLevels.WARNING)
            print_help_and_exit()
    result = {}
    if len(_models) == 3:
        consoleprint('Performig inference on all models...')
        for key, text in _texts.items():
            result[key] = predict(text)
    else:
        for key in _texts.keys():
            result[key] = {}
        for model_name in _models:
            consoleprint('Performig inference on "{}"...'.format(model_name))
            if model_name == 'model_1':
                for key, text in _texts.items():
                    result[key][model_name] = predict_model_1(text)[model_name]
            if model_name == 'model_2':
                for key, text in _texts.items():
                    result[key][model_name] = predict_model_2(text)[model_name]
            if model_name == 'model_3':
                for key, text in _texts.items():
                    result[key][model_name] = predict_model_3(text)[model_name]
    if result_file == '':
        result_file = 'result.json'
    with open(result_file, 'w', encoding='utf_8') as outstream:
        json_dump(result, outstream, indent='\t')
    consoleprint('Results saved to "{}".'.format(result_file))



def print_help_and_exit():
    """
    Print help screen on error and exit.
    ====================================
    """

    with click.Context(main) as click_context:
            click.echo(main.get_help(click_context))
    exit()


if __name__ == '__main__':
    main()
else:
    consoleprint('This is a software from the GCD Text-to-Risk project, not a ' +
                 'module.', PrintLevels.WARNING)
