#!/usr/bin/env python3
"""
GCD Text-to-Risk project
========================

Software: train
"""

from json import dump as json_dump
from os.path import isfile
from sys import exit

import click

from gcd_text_to_risk.constants import PrintLevels
from gcd_text_to_risk.functions import consoleprint
from gcd_text_to_risk.training import auto_train

@click.command()
@click.option('--training_dataset', '-T', 'training_file', default='',
              help='CSV file ([path]/name) for training dataset.', type=str)
@click.option('--validation_dataset', '-V', 'validation_file', default='',
              help='CSV file ([path]/name) for training dataset.', type=str)
@click.option('--result', '-R', 'result_file', default='',
              help='Result JSON file lcoation. If omitted result.json is used.',
              type=str)
def main(training_file : str, validation_file : str, result_file : str):
    """
    Software to perform GCD Text-to-Risk automated trining from CSV file datasets.
    """

    if training_file == '':
        consoleprint('Training dataset is mandatory.', PrintLevels.WARNING)
        print_help_and_exit()
    if not isfile(training_file):
        consoleprint('Training dataset is a non-existing file.',
                     PrintLevels.WARNING)
        print_help_and_exit()
    if not isfile(validation_file):
        consoleprint('Validation dataset is a non-existing file.',
                     PrintLevels.WARNING)
        print_help_and_exit()
    result = auto_train(training_file, validation_file)
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
