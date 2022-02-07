"""
GCD Text-to-Risk project
========================

Submodule: training
"""

from os import listdir
from os.path import isdir, isfile, join as path_join, split as path_split
from random import shuffle

import torch

from .constants import MODELS_DIR
from .filetools import extract_text, read_csv
from .functions import consoleprint
from .models import (batchify, save_model, Models, Model_1, Model_2, Model_3)
from .textprocessing import text_to_vector


CSV_FIELD_FILE = 'File'
CSV_FIELD_TEXT = 'Text'
CSV_FIELDS_MODEL_1 = ['Category_F', 'Category_N', 'Category_P', 'Category_R',
                      'Category_Sa', 'Category_Sh', 'Category_Sc', 'Category_So',
                      'Category_T']
CSV_FIELDS_MODEL_2 = ['Severity_F', 'Severity_N', 'Severity_P', 'Severity_R',
                      'Severity_Sa', 'Severity_Sh', 'Severity_Sc', 'Severity_So',
                      'Severity_T', 'Likelihood_F', 'Likelihood_N',
                      'Likelihood_P', 'Likelihood_R', 'Likelihood_Sa',
                      'Likelihood_Sh', 'Likelihood_Sc', 'Likelihood_So',
                      'Likelihood_T']
DATA_DIR = './data'


def auto_train() -> dict:
    """
    Perform automated training with pre-defined settings
    ====================================================

    Returns
    -------
    dict
    """

    data = make_dataset_from_csv('./data/train_demo/rixk_demo.csv')
    return train(data, data)


def default_save_criterion(epoch_id : int, epoch_max : int, train_losses : list,
                           validation_losses : list, train_accuracies : list,
                           validation_accuracies : list) -> bool:
    """
    Criterion function to decide whether save the model or not
    ==========================================================

    Parameters
    ----------
    epoch_id : int
        ID of the actual epoch (1..epoch_max[incl]).
    epoch_max : int
        Number of maximum epochs.
    train_losses : list
        List of epoch train losses.
    validation_losses : list
        List of epoch validation losses.
    train_accuracies : list
        List of epoch train accuracies.
    validation_accuracy : list
        List of epoch validation accuracies.

    Returns
    -------
    bool
        True, if the model should be saved, False if not.
    """

    result = True
    if len(validation_losses) > 1:
        result = validation_losses[-1] < min(validation_losses[:-1])
    elif len(train_losses) > 1:
        result = train_losses[-1] < min(train_losses[:-1])
    return result


def default_stop_criterion(epoch_id : int, epoch_max : int, train_losses : list,
                           validation_losses : list, train_accuracies : list,
                           validation_accuracies : list) -> bool:
    """
    Criterion function to decide whether stop the training or not
    =============================================================

    Parameters
    ----------
    epoch_id : int
        ID of the actual epoch (1..epoch_max[incl]).
    epoch_max : int
        Number of maximum epochs.
    train_losses : list
        List of epoch train losses.
    validation_losses : list
        List of epoch validation losses.
    train_accuracies : list
        List of epoch train accuracies.
    validation_accuracy : list
        List of epoch validation accuracies.

    Returns
    -------
    bool
        True, if the training should be stopped, False if not.
    """

    return False


def get_inital_dataset() -> dict:
    """
    Get an empty dataset
    ====================
    """

    return {'model_1' : {}, 'model_2' : {}, 'model_3' : {}}


def make_dataset_from_csv(file_name : str) -> dict:
    """
    Create dataset from CSV file
    ============================

    Parameters
    ----------
    file_name : str
        File name (and path) of the CSV file.

    Returns
    -------
    dict[key:[dict[key:[list]]]]
        Dict of model_1, model_2 and model_3 datasets in a form of dicts. Inner
        dict contain contain ID alike keys and tuple values. Tuples have the
        form (input, target). Keys are useful to train model_2 since it requires
        the output of model_1 for the same input. Values are pure Python
        primitives, further conversion is needed in most cases.
    """

    result = get_inital_dataset()
    base_path = path_split(file_name)[0]
    data = read_csv(file_name)
    if not all([k in data[0] for k in CSV_FIELDS_MODEL_1]) or\
       not all([k in data[0] for k in CSV_FIELDS_MODEL_2]):
        raise KeyError('Dataset CSV file misses some required fields.')
    for i, row in enumerate(data):
        texts = []
        if CSV_FIELD_TEXT in row:
            if row[CSV_FIELD_TEXT] != '':
                texts.append(row[CSV_FIELD_TEXT])
        if CSV_FIELD_FILE in row:
            if row[CSV_FIELD_FILE] != '':
                if isfile(row[CSV_FIELD_FILE]):
                    texts.append(extract_text(row[CSV_FIELD_FILE]))
                elif isfile(path_join(base_path, row[CSV_FIELD_FILE])):
                    texts.append(extract_text(path_join(base_path,
                                                        row[CSV_FIELD_FILE])))
        if len(texts) > 0:
            model_1_values = [int(row[k]) for k in CSV_FIELDS_MODEL_1]
            model_2_values = [float(row[k]) for k in CSV_FIELDS_MODEL_2]
            model_3_value = 0 if sum(model_1_values) == 0 else 1
            for j, text in enumerate(texts):
                vector = text_to_vector(text)
                result['model_1']['{}:{}'.format(i, j)] = (vector, model_1_values[:])
                result['model_2']['{}:{}'.format(i, j)] = (vector, model_2_values[:])
                result['model_3']['{}:{}'.format(i, j)] = (vector, model_3_value)
    return result


def make_dataset_from_folder(root_folder : str, apply_shuffle : bool = True) -> list:
    """
    Create dataset for model_3 from folder
    ======================================

    Parameters
    ----------
    root_folder : str
    apply_shuffle : bool = True

    Returns
    -------
    dict[key:[dict[key:[list]]]]
        Dict of model_1, model_2 and model_3 datasets in a form of dicts. Inner
        dict contain contain ID alike keys and tuple values. Tuples have the
        form (input, target). Keys are useful to train model_2 since it requires
        the output of model_1 for the same input. Values are pure Python
        primitives, further conversion is needed in most cases. Lists for
        model_1 and model_2 are empty.
    """

    result = get_inital_dataset()
    if isdir(root_folder):
        if all([isdir(path_join(root_folder, '0')),
                isdir(path_join(root_folder, '1'))]):
            dataset = []
            _id = 0
            for target in [0, 1]:
                folder = path_join(root_folder, str(target))
                for file_name in listdir(folder):
                    text = extract_text(path_join(folder, file_name))
                    dataset.append((text_to_vector(text), target))
                    _id += 1
            if apply_shuffle:
                shuffle(dataset)
            for i, row in enumerate(dataset):
                result['model_3']['{}:0'.format(i)] = row[:]
    return result


def train(train_dataset : dict, validation_dataset : dict, batch_size : int = 16,
          num_epochs : int = 1000, allow_cuda : bool = True,
          use_shuffle : bool = True, save_criterion : callable = None,
          stop_criterion : callable = None, save_on_finish : bool = True) -> dict:
    """
    Perform training
    ================

    Parameters
    ----------
    train_dataset : dict[dict]
        Dataset of the training. Dataset should have keys for all three models
        but keys' content can be empty. Models that belong to empty dataset will
        not get trained.
    validation_dataset : dict[dict]
        Dataset of the validation. Dataset should have keys for all three models
        but keys' content can be empty. Models that belong to empty dataset will
        not get trained.
    batch_size : int, optional (16 if omitted)
        Size of a batch.
    num_epochs : int, optional (1000 if omitted)
        Number of epochs to train. In the process count of epochs begins with 1
        and includes also the value of num_epochs.
    allow_cuda : bool, optional (True if omitted)
        Whether to allow the use of GPU training on CUDA devices or not. In case
        if this option is set to True and the hardware dosen't support CUDA
        the training will happen on CPU.
    use_shuffle : bool, optional (True if omitted)
        Whether to shuffle dataset in each epoch or not.
    save_criterion : callable|Nonetype, optional (None if omitted)
        Crietiron function to decide whether to save the model at the end of the
        epoch or not.
    stop_criterion : callable|Nonetype, optional (None if omitted)
        Crietiron function to decide whether to stop the training at the end of
        the epoch or not.
    save_on_finish : bool = True
        Whether to save model after training ends or not. If save_criterion and

    Returns
    -------
    dict[list]
        The result of the training.

    Raises
    ------
    KeyError
        When dataset dosen't have all the keys 'model_1', 'model_2' and
        'model_3'. Keys can be empty but they must ecist.

    Notes
    -----
        This function is split into sections with one-line comments. This is to
        let developers use of the code in their own approach.
    """

    global Models

    # Default values
    Learning_rate_model_1 = 5e1
    Learning_rate_model_2 = 5e1
    Learning_rate_model_3 = 5e1
    # Criterion functions
    if save_criterion is None:
        save_criterion = default_save_criterion
    if stop_criterion is None:
        stop_criterion = default_stop_criterion
    # Initialize result
    result = {'model_1' : {}, 'model_2' : {}, 'model_3' : {}}
    # Check dataset's keys
    for key in ['model_1', 'model_2', 'model_3']:
        if key not in train_dataset:
            raise KeyError('There is at least a missing key in the train ' +
                           'dataset.')
    for key in ['model_1', 'model_2', 'model_3']:
        if key not in validation_dataset:
            raise KeyError('There is at least a missing key in the validation' +
                           'dataset.')
    # Set device
    if allow_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # Model_1 training
    train_keys = list(train_dataset['model_1'].keys())
    validation_keys = list(validation_dataset['model_1'].keys())
    if len(train_keys) > 0:
        consoleprint('Training of Model 1. Starts.')
        # Prepare model
        if Models.MODEL_1 is None:
            Models.MODEL_1 = Model_1()
        Models.MODEL_1.to(device)
        # Prepare model helpers
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(Models.MODEL_1.parameters(),
                                     lr=Learning_rate_model_1)
        train_losses, train_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        train_len = len(train_dataset['model_1'])
        validation_len = len(validation_dataset['model_1'])
        # Prepare helper variables
        for epoch in range(1, num_epochs + 1):
            # Training
            section_losses = []
            section_corrects = 0
            Models.MODEL_1.train()
            if use_shuffle:
                shuffle(train_keys)
            for inputs, offsets, targets in batchify(train_dataset['model_1'],
                                                     train_keys,
                                                     batch_size):
                inputs = inputs.to(device)
                offsets = offsets.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                prediction = Models.MODEL_1(inputs, offsets)
                loss = criterion(prediction, targets)
                loss.backward()
                # This line here is to decrease the risk of wanishing gradients!
                torch.nn.utils.clip_grad_norm_(Models.MODEL_1.parameters(), 0.1)
                optimizer.step()
                predicted_values = torch.nn.functional.log_softmax(prediction,
                                                                   dim=1)
                for row in (predicted_values == targets).tolist():
                    for correct in row:
                        if correct:
                            section_corrects += 1
                section_losses += [loss.item() for i in range(targets.size(0))]
            train_losses.append(sum(section_losses) / train_len)
            train_accuracies.append(section_corrects / train_len)
            consoleprint('Epoch: {}/{} - TRAIN - loss: {:.6f} - accuracy: {:.6f}'
                         .format(epoch, num_epochs, train_losses[-1],
                                 train_accuracies[-1]))
            # Validation
            if len(validation_dataset['model_1']) > 0:
                # Training
                section_losses = []
                section_corrects = 0
                Models.MODEL_1.eval()
                with torch.no_grad():
                    for inputs, offsets, targets in batchify(validation_dataset
                                                             ['model_1'],
                                                             validation_keys,
                                                             batch_size):
                        inputs = inputs.to(device)
                        offsets = offsets.to(device)
                        targets = targets.to(device)
                        prediction = Models.MODEL_1(inputs, offsets)
                        loss = criterion(prediction, targets)
                        predicted_values = \
                            torch.nn.functional.log_softmax(prediction, dim=1)
                        for row in (predicted_values == targets).tolist():
                            for correct in row:
                                if correct:
                                    section_corrects += 1
                        section_losses += [loss.item()
                                           for i in range(targets.size(0))]
                validation_losses.append(sum(section_losses) / validation_len)
                validation_accuracies.append(section_corrects / validation_len)
                consoleprint('Epoch: {}/{} - VALIDATION - loss: {:.6f} - accuracy: {:.6f}'
                             .format(epoch, num_epochs, validation_losses[-1],
                                     validation_accuracies[-1]))
            # Check if save
            if save_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                save_model('model_1')
            # Check if stop
            if stop_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                break
        result['model_1']['train_losses'] = train_losses[:]
        result['model_1']['train_accuracies'] = train_accuracies[:]
        result['model_1']['validation_losses'] = validation_losses[:]
        result['model_1']['validation_accuracies'] = validation_accuracies[:]
        consoleprint('Training of Model 1. Stopped.')
    # Model_2 training
    train_keys = list(train_dataset['model_2'].keys())
    validation_keys = list(validation_dataset['model_2'].keys())
    if len(train_keys) > 0:
        consoleprint('Training of Model 2. Starts.')
        # Prepare model
        if Models.MODEL_2 is None:
            Models.MODEL_2 = Model_2()
        Models.MODEL_2.to(device)
        # Prepare model helpers
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(Models.MODEL_2.parameters(),
                                     lr=Learning_rate_model_2)
        train_losses, train_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        train_len = len(train_dataset['model_2'])
        validation_len = len(validation_dataset['model_2'])
        # Prepare helper variables
        Models.MODEL_1.eval()
        for epoch in range(1, num_epochs + 1):
            # Training
            section_losses = []
            section_corrects = 0
            Models.MODEL_2.train()
            if use_shuffle:
                shuffle(train_keys)
            for inputs, offsets, targets in batchify(train_dataset['model_2'],
                                                     train_keys,
                                                     batch_size):
                inputs = inputs.to(device)
                offsets = offsets.to(device)
                targets = targets.to(device)
                model_1_outputs = Models.MODEL_1(inputs, offsets)
                optimizer.zero_grad()
                prediction = Models.MODEL_2(inputs, offsets, model_1_outputs)
                loss = criterion(prediction, targets)
                loss.backward()
                # This line here is to decrease the risk of wanishing gradients!
                torch.nn.utils.clip_grad_norm_(Models.MODEL_2.parameters(), 0.1)
                optimizer.step()
                for row in (prediction == targets).tolist():
                    for correct in row:
                        if correct:
                            section_corrects += 1
                section_losses += [loss.item() for i in range(targets.size(0))]
            train_losses.append(sum(section_losses) / train_len)
            train_accuracies.append(section_corrects / train_len)
            consoleprint('Epoch: {}/{} - TRAIN - loss: {:.6f} - accuracy: {:.6f}'
                         .format(epoch, num_epochs, train_losses[-1],
                                 train_accuracies[-1]))
            # Validation
            if len(validation_dataset['model_2']) > 0:
                # Training
                section_losses = []
                section_corrects = 0
                Models.MODEL_2.eval()
                with torch.no_grad():
                    for inputs, offsets, targets in batchify(validation_dataset
                                                             ['model_2'],
                                                             validation_keys,
                                                             batch_size):
                        inputs = inputs.to(device)
                        offsets = offsets.to(device)
                        targets = targets.to(device)
                        model_1_outputs = Models.MODEL_1(inputs, offsets)
                        prediction = Models.MODEL_2(inputs, offsets,
                                                    model_1_outputs)
                        loss = criterion(prediction, targets)
                        for row in (prediction == targets).tolist():
                            for correct in row:
                                if correct:
                                    section_corrects += 1
                        section_losses += [loss.item()
                                           for i in range(targets.size(0))]
                validation_losses.append(sum(section_losses) / validation_len)
                validation_accuracies.append(section_corrects / validation_len)
                consoleprint('Epoch: {}/{} - VALIDATION - loss: {:.6f} - accuracy: {:.6f}'
                             .format(epoch, num_epochs, validation_losses[-1],
                                     validation_accuracies[-1]))
            # Check if save
            if save_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                save_model('model_2')
            # Check if stop
            if stop_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                break
        result['model_2']['train_losses'] = train_losses[:]
        result['model_2']['train_accuracies'] = train_accuracies[:]
        result['model_2']['validation_losses'] = validation_losses[:]
        result['model_2']['validation_accuracies'] = validation_accuracies[:]
        consoleprint('Training of Model 2. Stopped.')
    # Model_3 training
    train_keys = list(train_dataset['model_3'].keys())
    validation_keys = list(validation_dataset['model_3'].keys())
    if len(train_keys) > 0:
        consoleprint('Training of Model 3. Starts.')
        # Prepare model
        if Models.MODEL_3 is None:
            Models.MODEL_3 = Model_3()
        Models.MODEL_3.to(device)
        # Prepare model helpers
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(Models.MODEL_3.parameters(),
                                     lr=Learning_rate_model_3)
        train_losses, train_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        train_len = len(train_dataset['model_3'])
        validation_len = len(validation_dataset['model_3'])
        # Prepare helper variables
        for epoch in range(1, num_epochs + 1):
            # Training
            section_losses = []
            section_corrects = 0
            Models.MODEL_3.train()
            if use_shuffle:
                shuffle(train_keys)
            for inputs, offsets, targets in batchify(train_dataset['model_3'],
                                                     train_keys,
                                                     batch_size):
                inputs = inputs.to(device)
                offsets = offsets.to(device)
                targets = targets.unsqueeze(dim=-1)
                targets = targets.to(device)
                optimizer.zero_grad()
                prediction = Models.MODEL_3(inputs, offsets)
                loss = criterion(prediction, targets)
                loss.backward()
                # This line here is to decrease the risk of wanishing gradients!
                torch.nn.utils.clip_grad_norm_(Models.MODEL_3.parameters(), 0.1)
                optimizer.step()
                for row in (prediction == targets).tolist():
                    for correct in row:
                        if correct:
                            section_corrects += 1
                section_losses += [loss.item() for i in range(targets.size(0))]
            train_losses.append(sum(section_losses) / train_len)
            train_accuracies.append(section_corrects / train_len)
            consoleprint('Epoch: {}/{} - TRAIN - loss: {:.6f} - accuracy: {:.6f}'
                         .format(epoch, num_epochs, train_losses[-1],
                                 train_accuracies[-1]))
            # Validation
            if len(validation_dataset['model_3']) > 0:
                # Training
                section_losses = []
                section_corrects = 0
                Models.MODEL_3.eval()
                with torch.no_grad():
                    for inputs, offsets, targets in batchify(validation_dataset
                                                             ['model_3'],
                                                             validation_keys,
                                                             batch_size):
                        inputs = inputs.to(device)
                        offsets = offsets.to(device)
                        targets = targets.unsqueeze(dim=-1)
                        targets = targets.to(device)
                        prediction = Models.MODEL_3(inputs, offsets)
                        loss = criterion(prediction, targets)
                        for row in (prediction == targets).tolist():
                            for correct in row:
                                if correct:
                                    section_corrects += 1
                        section_losses += [loss.item()
                                           for i in range(targets.size(0))]
                validation_losses.append(sum(section_losses) / validation_len)
                validation_accuracies.append(section_corrects / validation_len)
                consoleprint('Epoch: {}/{} - VALIDATION - loss: {:.6f} - accuracy: {:.6f}'
                             .format(epoch, num_epochs, validation_losses[-1],
                                     validation_accuracies[-1]))
            # Check if save
            if save_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                save_model('model_3')
            # Check if stop
            if stop_criterion(epoch, num_epochs, train_losses, validation_losses,
                              train_accuracies, validation_accuracies):
                break
        result['model_3']['train_losses'] = train_losses[:]
        result['model_3']['train_accuracies'] = train_accuracies[:]
        result['model_3']['validation_losses'] = validation_losses[:]
        result['model_3']['validation_accuracies'] = validation_accuracies[:]
        consoleprint('Training of Model 3. Stopped.')
    if save_on_finish:
        save_model()
    return result
