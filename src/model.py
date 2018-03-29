from typing import Generator, ClassVar, Any, Union, Dict
from os.path import isdir, isfile
from abc import abstractmethod
from enum import Enum
from warnings import warn
from pprint import pprint, pformat
import json
from keras.models import Model, load_model
from keras.callbacks import History

_model_dir_path = '../data/output'


class TrainState(Enum):
    notTrained = 0
    trainingIncomplete = 1
    trained = 2


class TrainDevTestSplit:
    kind: ClassVar[str] = 'train-dev'
    train_range: range
    dev_range: range
    batch_size: int
    shuffle: bool

    def __init__(self, train_split_dict: Dict[str, Any]):
        self.train_range = range(train_split_dict['train_range'][0], train_split_dict['train_range'][1])
        self.dev_range = range(train_split_dict['dev_range'][0], train_split_dict['dev_range'][1])
        self.batch_size = train_split_dict['batch_size']
        self.shuffle = train_split_dict['shuffle']

    def __repr__(self):
        return pformat(vars(self))


class ModelDescriptor:

    name: str
    data_path: str
    train_state: TrainState
    model: Model
    history: History
    trained_epochs: int
    training_target_epochs: int
    training_split: Union[TrainDevTestSplit, None]

    def __init__(self, name: str):
        self.name = name
        self.data_path = f'{_model_dir_path}/{self.name}'
        self._load_model_data()

    def _load_model_data(self):
        model_path = f'{self.data_path}/model.h5'
        history_path = f'{self.data_path}/history.json'
        descriptor_path = f'{self.data_path}/descriptor.json'

        def init_origin():
            self.train_state = TrainState.notTrained
            self.model = self.create_model()
            self.history = History()
            self.history.epoch = []
            self.history.history = {}
            self.trained_epochs = 0
            self.training_target_epochs = 0
            self.training_split = None

        if not isdir(self.data_path):
            return init_origin()
        if not isfile(model_path):
            warn(f'Directory({self.data_path}) exists but no model file({model_path}) found')
            return init_origin()
        if not isfile(history_path):
            warn(f'Directory({self.data_path}) exists but no history file({history_path}) found')
            return init_origin()
        if not isfile(descriptor_path):
            warn(f'Directory({model_dir}) exists but no descriptor file({descriptor_path}) found')
            return init_origin()

        self.model = load_model(model_path, compile=True)

        # recreating history object
        self.history = History()
        with open(history_path, encoding='utf-8') as history_file:
            self.history.history = json.load(history_file)

        # assuming all keras history entries have same length
        history_entry = next(iter(self.history.history.values()))
        if not history_entry:
            raise Exception(f'History file({history_path}) contains no entries, while expected')
        self.history.epoch = list(range(1, len(history_entry) + 1))

        # parsing descriptor
        with open(descriptor_path, encoding='utf-8') as descriptor_file:
            descriptor_dict = json.load(descriptor_file)
            self.trained_epochs = descriptor_dict['trained_epochs']
            self.training_target_epochs = descriptor_dict['training_target_epochs']
            self.train_state = TrainState.trained if self.trained_epochs == self.training_target_epochs else TrainState.trainingIncomplete
            if descriptor_dict['training_split']['kind'] == 'train-dev':
                self.training_split = TrainDevTestSplit(descriptor_dict['training_split'])

    @abstractmethod
    def data_generator(self) -> Generator:
        raise NotImplemented

    @abstractmethod
    def create_model(self) -> Model:
        raise NotImplementedError


