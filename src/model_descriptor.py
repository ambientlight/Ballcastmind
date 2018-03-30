from collections import namedtuple
from enum import Enum
from typing import ClassVar, Union, Optional, Callable, Any, Sequence, Generator, Dict, Tuple
from abc import abstractmethod
from os import listdir, remove
from os.path import isdir, isfile
from warnings import warn
import json
import re
import math

from keras.models import Model, load_model
#from hyperas import optim
from hyperopt import tpe

from src.model_descriptor_state import ModelDescriptorStateType, ModelDescriptorTrainingDevState, model_descriptor_state_from_file
from src.model_training_state import TrainingStateRegular, TrainingStateKFold, training_state_from_file
from src.model_state_saver import ModelStateSaver


_model_dir_path = '../data/output'
BuildAndVersion = namedtuple('BuildAndVersion', ['build', 'version'])


class GradientDescentVariant(Enum):
    miniBatch = 'mini-batch'
    batch = 'batch'
    stochastic = 'stochastic'


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
    state: Union[ModelDescriptorTrainingDevState, None]

    @property
    def data_path(self) -> str: return f'{_model_dir_path}/{self.name}'

    @property
    def is_idle(self) -> bool: return not self.state

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
            self.state = None

        if not isdir(self.data_path):
            init_origin()
            return
        if not isfile(descriptor_path):
            init_origin()
            return

        self.state = model_descriptor_state_from_file(descriptor_path)

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

    def load_model(self, stage: str, build: int, version: int) -> Optional[
        Tuple[
            Model,
            Union[TrainingStateRegular, TrainingStateKFold],
            Dict[str, Any]
        ]
    ]:
        directory_path = f'{self.data_path}/{stage}'
        if not isdir(directory_path):
            return None

        model_dir = f'{directory_path}/{build}.v{version}'
        if not isdir(model_dir):
            return None

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

        training_state = training_state_from_file(training_state_path,
                                                  version=version,
                                                  build=build)

        with open(history_path, encoding='utf-8') as history_file:
            history_dict = json.load(history_file)

        return load_model(model_path), training_state, history_dict

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
        return self.load_model(stage='dev', build=latest_build_version.build, version=latest_build_version.version)

    def _train_dev_regular(self, model: Model, model_state: TrainingStateRegular, history_dict: Dict[str, Any]):

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
            val_generator = self.data_generator(data_entries[train_entries_len:train_entries_len + val_entries_len],
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

        self._update_state(ModelDescriptorTrainingDevState(model_state.build))

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=model_state.training_target_epochs - model_state.trained_epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[model_state_saver]
        )

        self._update_state(None)

    def _update_state(self, state: Union[ModelDescriptorTrainingDevState, None]):
        state_path = f'{self.data_path}/descriptor.json'
        if state:
            state.save_to_file(state_path)
        elif isfile(state_path):
            remove(state_path)
        self.state = state

    def train_validate(self, epoch: int = 10, build: int = 0):
        # search if dev model exists for this version
        latest_dev_model_res = self.latest_dev_model()
        if not latest_dev_model_res:
            model = self.create_model(for_optimization=False)
            model_state = TrainingStateRegular(version=self._version, build=build)
            history_dict = {}
        else:
            model, model_state, history_dict = latest_dev_model_res

        model_state.training_target_epochs += epoch
        self._train_dev_regular(model, model_state, history_dict)

    # should return best_run data
    def optimize(self, max_evals: int, search_algo: Callable = tpe.suggest) -> Optional[Dict[str, Any]]:
        pass

    def kfold_validate(self, folds: int = 4):
        pass

    def train_prod(self, epoch: Optional[int] = None, build: Optional[int] = None):
        pass

    def resume_if_needed(self):
        if self.state.type == ModelDescriptorStateType.trainingDev:
            loaded_model_res = self.load_model(stage='dev', build=self.state.build, version=self._version)
            if not loaded_model_res:
                warn(f"Couldn't resume training. TrainingDev was set, "
                     f"but build:{self.state.build}, version:{self._version} dev-model is not available."
                     f"Version might have changed or model files were moved/deleted")
                self._update_state(None)

            model, model_state, history_dict = loaded_model_res
            self._train_dev_regular(model, model_state, history_dict)

        else:
            print('No task to resume')

