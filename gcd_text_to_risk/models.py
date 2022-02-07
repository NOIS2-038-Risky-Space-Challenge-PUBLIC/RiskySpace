"""
GCD Text-to-Risk project
========================

Submodule: models
"""

from os.path import isfile, join as path_join

import torch

from .constants import MODELS_DIR
from .textprocessing import BAG_SIZE


class Models:

    MODEL_1 = None
    MODEL_2 = None
    MODEL_3 = None


class Model_1(torch.nn.Module):
    """
    Model to identify risk occurrances
    ==================================
    """

    def __init__(self):
        """
        Initialize the object
        =====================
        """

        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(BAG_SIZE, 256, sparse=True)
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 9)
        self.init_weights()


    def forward(self, batch_text : torch.Tensor,
                batch_offsets : torch.Tensor) -> torch.Tensor:
        """
        Perform forward process
        =======================

        Parameters
        ----------
        batch_text : torch.Tensor
            Batch of text as vector. Shape: [batch_size]
        batch_offsets : torch.Tensor
            Offsets of first tokens of batch entities. Shape: [batch_size]

        Returns
        -------
        torch.Tensor
            Result of the forward process (prediction). Shape: [batch_size, 9]
        """

        x = self.embedding(batch_text, batch_offsets)
        x = torch.nn.functional.relu(self.fc1(x), inplace=True)
        x = torch.nn.functional.relu(self.fc2(x), inplace=True)
        return self.fc3(x)


    def init_weights(self, init_range : float = 0.5):
        """
        Initialize model weights and biases
        ===================================

        Parameters
        ----------
        init_range : float, optional (0.5 if omitted)
            Range (-/+) in which the initial random uniform values spread.
        """

        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-init_range, init_range)
        self.fc3.bias.data.zero_()


class Model_2(torch.nn.Module):
    """
    Model to identify severities and likelihoods of risks
    =====================================================
    """

    def __init__(self):
        """
        Initialize the object
        =====================
        """

        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(BAG_SIZE, 256, sparse=True)
        self.fc1 = torch.nn.Linear(265, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 18)
        self.init_weights()


    def forward(self, batch_text : torch.Tensor, batch_offsets : torch.Tensor,
                model_1_outputs : torch.Tensor) -> torch.Tensor:
        """
        Perform forward process
        =======================

        Parameters
        ----------
        batch_text : torch.Tensor
            Batch of text as vector. Shape: [batch_size]
        batch_offsets : torch.Tensor
            Offsets of first tokens of batch entities. Shape: [batch_size]
        model_1_outputs : torch.Tensor
            Outpot of Model 1 for the same batch. Shape: [batch_size, 9]

        Returns
        -------
        torch.Tensor
            Result of the forward process (prediction). Shape: [batch_size, 18]
        """

        x = self.embedding(batch_text, batch_offsets)
        x = torch.cat((x, model_1_outputs), dim=1)
        x = torch.nn.functional.relu(self.fc1(x), inplace=True)
        x = torch.nn.functional.relu(self.fc2(x), inplace=True)
        return torch.tanh(self.fc3(x))


    def init_weights(self, init_range : float = 0.5):
        """
        Initialize model weights and biases
        ===================================

        Parameters
        ----------
        init_range : float, optional (0.5 if omitted)
            Range (-/+) in which the initial random uniform values spread.
        """

        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-init_range, init_range)
        self.fc3.bias.data.zero_()


class Model_3(torch.nn.Module):
    """
    Model to identify existence of risk
    ===================================
    """

    def __init__(self):
        """
        Initialize the object
        =====================
        """

        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(BAG_SIZE, 256, sparse=True)
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.init_weights()


    def forward(self, batch_text : torch.Tensor,
                batch_offsets : torch.Tensor) -> torch.Tensor:
        """
        Perform forward process
        =======================

        Parameters
        ----------
        batch_text : torch.Tensor
            Batch of text as vector. Shape: [batch_size]
        batch_offsets : torch.Tensor
            Offsets of first tokens of batch entities. Shape: [batch_size]

        Returns
        -------
        torch.Tensor
            Result of the forward process (prediction). Shape: [batch_size, 1]
        """

        x = self.embedding(batch_text, batch_offsets)
        x = torch.nn.functional.relu(self.fc1(x), inplace=True)
        x = torch.nn.functional.relu(self.fc2(x), inplace=True)
        return torch.sigmoid(self.fc3(x))


    def init_weights(self, init_range : float = 0.5):
        """
        Initialize model weights and biases
        ===================================

        Parameters
        ----------
        init_range : float, optional (0.5 if omitted)
            Range (-/+) in which the initial random uniform values spread.
        """

        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-init_range, init_range)
        self.fc3.bias.data.zero_()


def batchify(dataset : dict, keys : list, batch_size : int,
             drop_short : bool = False) -> tuple:
    """
    Generator function to create batch
    ==================================

    Parameters
    ----------
    dataset : dict
        Dataset to use.
    keys : list
        Dataset keys.
    batch_size : int
        Size of batches to yield.
    drop_short : bool, optional (False if omitted)
        Whether to drop last batch if its length is smaller than the batch_size
        or not.

    Yields
    ------
    tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        Tuple of input texts, text offsets and prediction targets.
    """

    pointer = 0
    total_len = len(keys)
    while pointer + batch_size <= total_len:
        inputs, offsets, targets = [], [0], []
        for key in keys[pointer:pointer + batch_size]:
            inputs.append(torch.tensor(dataset[key][0], dtype=torch.int64))
            offsets.append(len(dataset[key][0]))
            targets.append(dataset[key][1])
        inputs = torch.cat(inputs)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        targets = torch.tensor(targets, dtype=torch.float)
        yield (inputs, offsets, targets)
        pointer += batch_size
    if pointer < total_len:
        inputs, offsets, targets = [], [0], []
        for key in keys[pointer:]:
            inputs.append(torch.tensor(dataset[key][0], dtype=torch.int64))
            offsets.append(len(dataset[key][0]))
            targets.append(dataset[key][1])
        inputs = torch.cat(inputs)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        targets = torch.tensor(targets, dtype=torch.float)
        yield (inputs, offsets, targets)


def load_model(model_name : any = None, model_location : any = None,
               error_on_failure : bool = True):
    """
    Load model state dict
    =====================

    Parameters
    ----------
    model_name : list|str|NoneType, optional (None if omitted)
        Model name(s) to load. If a string is given, only one model gets loaded.
        If a list is given none or all model can get loaded depending on the
        content of the list. If None is given, all model get loaded.
    model_location : list|str|NoneType, optional (None if omitted)
        Model filename(s) or path(s). If a string is given, it is considered as
        folder name. If a list is given it is considered as file names for the
        given model names in the same order. Default order of model names is:
        model_1, model_2, model_3. Number of list entries must match number of
        models to load. If None is given, default file names are used.
    error_on_failure : bool, optional (True if omitted)
        Whhether to raise error on load failure or not. This parameter works
        only at factual loading process. Parsing function parameters can raise
        error even if this value is set to False.

    Raises
    ------
    ValueError
        If a given model_name doesn't exist.
    TypeError
        If model_name is not NoneType, neiter list, nor str.
    ValueError
        If more model location is given than the count of models.
    TypeError
        If model_location is not NoneType, neiter list, nor str.
    FileNotFoundError
        If error_on_failure is set to True and model_1 file doesn't exist.
    FileNotFoundError
        If error_on_failure is set to True and model_2 file doesn't exist.
    FileNotFoundError
        If error_on_failure is set to True and model_3 file doesn't exist.
    """

    global Models

    _models = ['model_1', 'model_2', 'model_3']
    _folder = MODELS_DIR
    _files = {n : '{}.statedict'.format(n) for n in _models}
    if model_name is not None:
        _files = {}
        if isinstance(model_name, list):
            for element in model_name:
                if element not in _models:
                    raise ValueError('Non-existing model name were given.')
                _files[element] = '{}.statedict'.format(element)
            _models = model_name[:]
        elif isinstance(model_name, str):
            if model_name not in _models:
                raise ValueError('Non-existing model name were given.')
            _files[model_name] = '{}.statedict'.format(model_name)
        else:
            raise TypeError('Not supported type for model_name parameter.')
    if model_location is not None:
        if isinstance(model_location, list):
            if len(model_location) > len(_files):
                raise ValueError('Count of model_location parameter doesn\'t ' +
                                 'count of models.')
            for i in range(len(model_location)):
                _files[_models[i]] = '{}.statedict'.format(model_location[i])
        elif isinstance(model_location, str):
            _folder = model_location
        else:
            raise TypeError('Not supported type for model_location parameter.')
    if 'model_1' in _files:
        file_name = path_join(_folder, _files['model_1'])
        if isfile(file_name):
            if Models.MODEL_1 is None:
                Models.MODEL_1 = Model_1()
            Models.MODEL_1.load_state_dict(torch.load(file_name))
        elif error_on_failure:
            raise FileNotFoundError('State dict for model_1 doesn\'t exist.')
    if 'model_2' in _files:
        file_name = path_join(_folder, _files['model_2'])
        if isfile(file_name):
            if Models.MODEL_2 is None:
                Models.MODEL_2 = Model_2()
            Models.MODEL_2.load_state_dict(torch.load(file_name))
        elif error_on_failure:
            raise FileNotFoundError('State dict for model_2 doesn\'t exist.')
    if 'model_3' in _files:
        file_name = path_join(_folder, _files['model_3'])
        if isfile(file_name):
            if Models.MODEL_3 is None:
                Models.MODEL_3 = Model_3()
            Models.MODEL_3.load_state_dict(torch.load(file_name))
        elif error_on_failure:
            raise FileNotFoundError('State dict for model_3 doesn\'t exist.')


def save_model(model_name : any = None, model_location : any = None,
               move_to_cpu_on_save : bool = False):
    """
    Save model state dict
    =====================

    Parameters
    ----------
    model_name : list|str|NoneType, optional (None if omitted)
        Model name(s) to save. If a string is given, only one model gets saved.
        If a list is given none or all model can get saved depending on the
        content of the list. If None is given, all model get saved.
    model_location : list|str|NoneType, optional (None if omitted)
        Model filename(s) or path(s). If a string is given, it is considered as
        folder name. If a list is given it is considered as file names for the
        given model names in the same order. Default order of model names is:
        model_1, model_2, model_3. Number of list entries must match number of
        models to save. If None is given, default file names are used.
    move_to_cpu_on_save : bool, optional (False if omitted)
        Whether to move the model cpu on save or not. Setting this value to True
        is a match safer approach especially if models get interchanged between
        machines with different hardware setup but movoing models between devies
        is very slow.

    Raises
    ------
    ValueError
        If a given model_name doesn't exist.
    TypeError
        If model_name is not NoneType, neiter list, nor str.
    ValueError
        If more model location is given than the count of models.
    TypeError
        If model_location is not NoneType, neiter list, nor str.
    RuntimeError
        If model_1 is not initialized.
    RuntimeError
        If model_2 is not initialized.
    RuntimeError
        If model_3 is not initialized.
    """

    _models = ['model_1', 'model_2', 'model_3']
    _folder = MODELS_DIR
    _files = {n : '{}.statedict'.format(n) for n in _models}
    if model_name is not None:
        _files = {}
        if isinstance(model_name, list):
            for element in model_name:
                if element not in _models:
                    raise ValueError('Non-existing model name were given.')
                _files[element] = '{}.statedict'.format(element)
            _models = model_name[:]
        elif isinstance(model_name, str):
            if model_name not in _models:
                raise ValueError('Non-existing model name were given.')
            _files[model_name] = '{}.statedict'.format(model_name)
        else:
            raise TypeError('Not supported type for model_name parameter.')
    if model_location is not None:
        if isinstance(model_location, list):
            if len(model_location) > len(_files):
                raise ValueError('Count of model_location parameter doesn\'t ' +
                                 'count of models.')
            for i in range(len(model_location)):
                _files[_models[i]] = '{}.statedict'.format(model_location[i])
        elif isinstance(model_location, str):
            _folder = model_location
        else:
            raise TypeError('Not supported type for model_location parameter.')
    if 'model_1' in _files:
        file_name = path_join(_folder, _files['model_1'])
        if Models.MODEL_1 is None:
            raise RuntimeError('Model_1 is not yet Initialized.')
        if move_to_cpu_on_save:
            _device = next(Models.MODEL_1.parameters()).device
            Models.MODEL_1.to('cpu')
            torch.save(Models.MODEL_1.state_dict(), file_name)
            Models.MODEL_1.to(_device)
        else:
            torch.save(Models.MODEL_1.state_dict(), file_name)
    if 'model_2' in _files:
        file_name = path_join(_folder, _files['model_2'])
        if Models.MODEL_2 is None:
            raise RuntimeError('Model_2 is not yet Initialized.')
        if move_to_cpu_on_save:
            _device = next(Models.MODEL_2.parameters()).device
            Models.MODEL_2.to('cpu')
            torch.save(Models.MODEL_2.state_dict(), file_name)
            Models.MODEL_2.to(_device)
        else:
            torch.save(Models.MODEL_2.state_dict(), file_name)
    if 'model_3' in _files:
        file_name = path_join(_folder, _files['model_3'])
        if Models.MODEL_3 is None:
            raise RuntimeError('Model_3 is not yet Initialized.')
        if move_to_cpu_on_save:
            _device = next(Models.MODEL_3.parameters()).device
            Models.MODEL_3.to('cpu')
            torch.save(Models.MODEL_3.state_dict(), file_name)
            Models.MODEL_3.to(_device)
        else:
            torch.save(Models.MODEL_3.state_dict(), file_name)
