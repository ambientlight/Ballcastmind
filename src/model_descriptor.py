from collections import namedtuple
from enum import Enum
from typing import ClassVar, Union, Optional, Callable, Any, Sequence, Generator, Dict, Tuple
from abc import abstractmethod
from os import listdir
from os.path import isdir, isfile
from warnings import warn
from pprint import pformat
import json
import re
import math

from keras.models import Model, load_model
from hyperas import optim
from hyperopt import tpe
from src.model_state_saver import ModelStateSaver


_model_dir_path = '../data/output'
BuildAndVersion = namedtuple('BuildAndVersion', ['build', 'version'])


class ModelDescriptorState(Enum):
    idle = 0
    trainingDev = 1
    optimizing = 2
    kFoldValidating = 3
    trainingProd = 4


class GradientDescentVariant(Enum):
    miniBatch = 0
    batch = 1
    stochastic = 2


class TrainingStateRegular:
    trained_epochs: int
    training_target_epochs: int
    version: int
    build: int

    def __init__(self, version: int, build: int, training_state_dict: Optional[Dict[str, Any]] = None):
        self.version = version
        self.build = build
        self.trained_epochs = int(training_state_dict['trained_epochs']) if training_state_dict else 0
        self.training_target_epochs = int(training_state_dict['training_target_epochs']) if training_state_dict else 0

    def __repr__(self):
        return pformat(vars(self))

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'trained_epochs': self.trained_epochs,
                'training_target_epochs': self.training_target_epochs
            }), file=json_file)


class TrainingStateKFold(TrainingStateRegular):
    folds: int
    current_fold: int

    def __init__(self, version: int, build: int, training_state_dict: Optional[Dict[str, Any]] = None):
        super().__init__(version=version, build=build, training_state_dict=training_state_dict)
        self.folds = int(training_state_dict['folds'])
        self.current_fold = int(training_state_dict['current_fold'])

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'trained_epochs': self.trained_epochs,
                'training_target_epochs': self.training_target_epochs,
                'folds': self.folds,
                'current_fold': self.current_fold
            }), file=json_file)


class ModelDescriptor:

    # increment the version when any changes to the model/data/train-setup has been made
    _version: ClassVar[int] = 0
    # what fraction of all data returned from data_location used for training
    _trainFraction: ClassVar[int] = 0.8
    # There are three variants of gradient descent,
    # which differ in how much data we use to compute the gradient of the objective function.
    # Depending on the amount of data, we make a trade-off
    # between the accuracy of the parameter update and the time it takes to perform an update.
    _gradientVariant: ClassVar[GradientDescentVariant] = GradientDescentVariant.miniBatch
    # when using mini-batch gradient descent this value will be used to determine batch size
    _miniBatchSize: ClassVar[Optional[int]] = 64

    name: str
    state: ModelDescriptorState

    @property
    def data_path(self) -> str: return f'{_model_dir_path}/{self.name}'

    @property
    def is_idle(self) -> bool: return self.state == ModelDescriptorState.idle

    '''
        should return already compiled model
    '''
    @abstractmethod
    def create_model(self, for_optimization: bool) -> Model:
        raise NotImplemented

    @abstractmethod
    def data_locators(self) -> Sequence[Any]:
        raise NotImplemented

    @abstractmethod
    def data_generator(self, data_locators: Sequence[Any], batch_size: int) -> Generator:
        raise NotImplemented

    def __init__(self, name: str):
        self.name = name
        descriptor_path = f'{self.data_path}/descriptor.json'

        def init_origin():
            self.state = ModelDescriptorState.idle

        if not isdir(self.data_path):
            init_origin()
            return
        if not isfile(descriptor_path):
            init_origin()
            return

        with open(descriptor_path, encoding='utf-8') as descriptor_file:
            descriptor_dict = json.load(descriptor_file)
            self.state = descriptor_dict['state']

    '''
    descriptor.json
    /dev contains all development models
        /0.v0
        /1.v0
            model.h5
            training_state.json
            history.json
            kfold_history.json
    /prod contains all production models
    /temp contains intermediate cache data that is preserved to resume training
    '''

    '''
    TODO: base apis
    '''

    @property
    def _model_dir_pattern(self):
        return r'^(\d+)\.v(\d+)$'

    def latest_dev_model(self) -> Optional[
        Tuple[
            Model,
            Union[TrainingStateRegular, TrainingStateKFold],
            Dict[str, Any]
        ]
    ]:

        dev_directory_path = f'{self.data_path}/dev'
        if not isdir(dev_directory_path):
            return None

        dev_subdirs_matches = [re.search(self._model_dir_pattern, dir_name) for dir_name
                               in listdir(dev_directory_path)
                               if isdir(f'{dev_directory_path}/{dir_name}')
                               and re.search(self._model_dir_pattern, dir_name)]
        dev_build_version_tuples = [BuildAndVersion(int(match[1]), int(match[2])) for match in dev_subdirs_matches]
        dev_build_versions_matching_version = sorted((build_version for build_version
                                                      in dev_build_version_tuples
                                                      if build_version.version == self._version),
                                                     key=lambda build_version: build_version.build)

        if len(dev_build_versions_matching_version) == 0:
            return None

        latest_build_version = dev_build_versions_matching_version[-1]
        model_dir = f'{dev_directory_path}/{latest_build_version.build}.v{latest_build_version.version}'
        model_path = f'{model_dir}/model.h5'
        training_state_path = f'{model_dir}/state.json'
        history_path = f'{model_dir}/history.json'
        if not isfile(model_path):
            warn(f'Model data is not available at {model_path}, while expected')
            return None
        if not isfile(training_state_path):
            warn(f'Model state is not available at {training_state_path}, while expected')
            return None
        if not isfile(history_path):
            warn(f'Model history is not available at {history_path}, while expected')
            return None

        with open(training_state_path, encoding='utf-8') as training_state_file:
            training_state_dict = json.load(training_state_file)
            if training_state_dict['type'] == 'k-fold':
                training_state = TrainingStateKFold(training_state_dict=training_state_dict,
                                                    version=latest_build_version.version,
                                                    build=latest_build_version.build)
            else:
                training_state = TrainingStateRegular(training_state_dict=training_state_dict,
                                                      version=latest_build_version.version,
                                                      build=latest_build_version.build)

        with open(history_path, encoding='utf-8') as history_file:
            history_dict = json.load(history_file)

        return load_model(model_path), training_state, history_dict

    def train_validate(self, epoch: int = 10, build: Optional[int] = None):
        # search if dev model exists for this version
        latest_dev_model_res = self.latest_dev_model()
        if not latest_dev_model_res:
            model = self.create_model(for_optimization=False)
            model_state = TrainingStateRegular(version=self._version, build=build if build else 0)
            history_dict = {}
        else:
            model, model_state, history_dict = latest_dev_model_res

        model_state.training_target_epochs += epoch
        model_dir = f'{self.data_path}/dev/{model_state.build}.v{model_state.version}'
        model_state_saver = ModelStateSaver(training_state=model_state, history=history_dict, model_folder=model_dir)

        data_entries = self.data_locators()
        data_entries_len = len(data_entries)
        train_entries_total = int(math.floor(data_entries_len * self._trainFraction))

        if self._gradientVariant == GradientDescentVariant.miniBatch:
            batches = int(math.floor(train_entries_total / self._miniBatchSize))
            # our test entries should be a multiples of batch size
            train_entries_len = self._miniBatchSize * batches
            train_generator = self.data_generator(data_entries[:train_entries_len],
                                                  batch_size=self._miniBatchSize)

            val_entries_total = data_entries_len - train_entries_len
            val_batches = int(math.floor(val_entries_total / self._miniBatchSize))
            val_entries_len = self._miniBatchSize * val_batches
            val_generator = self.data_generator(data_entries[train_entries_len:train_entries_len+val_entries_len],
                                                batch_size=self._miniBatchSize)

            steps_per_epoch = batches
            validation_steps = val_batches

        elif self._gradientVariant == GradientDescentVariant.batch:
            steps_per_epoch = 1
            validation_steps = 1
            train_generator = self.data_generator(data_entries[:train_entries_total],
                                                  batch_size=train_entries_total)
            val_generator = self.data_generator(data_entries[train_entries_total:],
                                                batch_size=data_entries_len - train_entries_total)

        else:
            steps_per_epoch = train_entries_total
            validation_steps = data_entries_len - train_entries_total
            train_generator = self.data_generator(data_entries[:train_entries_total],
                                                  batch_size=1)
            val_generator = self.data_generator(data_entries[train_entries_total:],
                                                batch_size=1)

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[model_state_saver]
        )

    # should return best_run data
    def optimize(self, max_evals: int, search_algo: Callable = tpe.suggest) -> Optional[Dict[str, Any]]:
        pass

    def kfold_validate(self, folds: int = 4):
        pass

    def train_prod(self, epoch: Optional[int] = None, build: Optional[int] = None):
        pass

    def resume_if_needed(self):
        pass
