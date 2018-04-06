from collections import namedtuple
from enum import Enum
from typing import ClassVar, Optional, Any, Sequence, Generator, Dict, Tuple, List
from abc import abstractmethod
from os import listdir, remove
from os.path import isdir, isfile
from warnings import warn
import json
import re
import math
import pickle
import numpy

from keras.models import Model, load_model
from hyperopt import tpe, Trials, STATUS_OK, fmin

from src.model_descriptor_state import ModelDescriptorStateType, ModelDescriptorState, \
    ModelDescriptorTrainingDevState, ModelDescriptorOptimizingState, ModelDescriptorKFoldValidatingState, \
    ModelDescriptorTrainingProdState, model_descriptor_state_from_file
from src.model_training_state import TrainingState
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
    state: ModelDescriptorState

    # used internally by resume_if_needed and optimization's object
    _restored_model_info: Optional[Tuple[Model, TrainingState, Dict[str, Any]]]

    @property
    def data_path(self) -> str: return f'{_model_dir_path}/{self.name}'

    @property
    def descriptor_path(self) -> str: return f'{self.data_path}/descriptor.json'

    @property
    def trials_path(self) -> str: return f'{self.data_path}/trials.p'

    @property
    def is_idle(self) -> bool: return not self.state

    '''
        should return already compiled model
    '''

    @abstractmethod
    def create_model(self, space: Optional[Dict[str, Any]] = None) -> Model:
        return NotImplemented

    @abstractmethod
    def hyperopt_space(self) -> Dict[str, Any]:
        return NotImplemented

    @abstractmethod
    def data_locators(self) -> List[Any]:
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
        self._restored_model_info = None

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
        Tuple[Model, TrainingState, Dict[str, Any]]
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

        training_state = TrainingState.from_file(training_state_path,
                                                 version=version,
                                                 build=build)

        with open(history_path, encoding='utf-8') as history_file:
            history_dict = json.load(history_file)

        return load_model(model_path), training_state, history_dict

    def latest_build(self, stage: str) -> Optional[int]:
        stage_directory_path = f'{self.data_path}/{stage}'
        if not isdir(stage_directory_path):
            return None

        dev_subdirs_matches = [re.search(self._model_dir_pattern, dir_name) for dir_name
                               in listdir(stage_directory_path)
                               if isdir(f'{stage_directory_path}/{dir_name}')
                               and re.search(self._model_dir_pattern, dir_name)]
        dev_build_version_tuples = [BuildAndVersion(int(match[1]), int(match[2])) for match in dev_subdirs_matches]
        dev_build_versions_matching_version = sorted((build_version for build_version
                                                      in dev_build_version_tuples
                                                      if build_version.version == self._version),
                                                     key=lambda build_version: build_version.build)

        if len(dev_build_versions_matching_version) == 0:
            return None

        return dev_build_versions_matching_version[-1].build

    def latest_dev_model(self) -> Optional[Tuple[Model, TrainingState, Dict[str, Any]]]:

        latest_build = self.latest_build(stage='dev')
        if not latest_build:
            return None

        return self.load_model(stage='dev', build=latest_build, version=self._version)

    def _train_dev_core(self, model: Model, model_state: TrainingState, history_dict: Dict[str, Any]) -> float:

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

            print(f'Total training entries: {train_entries_total}, used: {train_entries_len}')
            print(f'Total validating entries: {val_entries_total}, used: {val_entries_len}')

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

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=model_state.training_target_epochs - model_state.trained_epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[model_state_saver]
        )
        return history.history['val_loss'][-1]

    def _kfold_core(self,
                    epoch: int,
                    build: Optional[int],
                    current_fold: int,
                    folds: int,
                    intermediate_val_scores: List[float],
                    restored_model: Optional[Tuple[Model, TrainingState, Dict[str, Any]]] = None) -> float:

        if folds < 2:
            raise Exception(f'Fold size({folds}) should be at least 2')
        if self._gradientVariant == GradientDescentVariant.miniBatch and self._miniBatchSize % folds != 0:
            raise Exception(f'Mini batch size({self._miniBatchSize}) should be devisible by number of folds({folds}).')

        data_entries = self.data_locators()
        data_entries_len = len(data_entries)

        if self._gradientVariant == GradientDescentVariant.miniBatch:
            batches = int(math.floor(data_entries_len / self._miniBatchSize))
            # clip the data so that the length is a multiple of batch size
            clipped_data_entries_len = batches * self._miniBatchSize
            fold_size = clipped_data_entries_len / folds
            if fold_size < self._miniBatchSize:
                raise Exception(f'Fold size({fold_size}) should not be smaller then batch size({self._miniBatchSize})')

            entries_len = clipped_data_entries_len
            val_batch_size = train_batch_size = self._miniBatchSize
            # fold_size / mini-batch size
            val_steps_per_epoch = int(int(entries_len / folds) / self._miniBatchSize)
            # all-except-single-fold / mini-batch size
            train_steps_per_epoch = int(int(math.floor(((folds - 1) / folds) * entries_len)) / self._miniBatchSize)

            print(f'Total entries: {data_entries_len}, used: {entries_len}')
            print(f'Batch size: {train_batch_size}')
            print(f'Train steps: {train_steps_per_epoch}, validation steps: {val_steps_per_epoch}')

        elif self._gradientVariant == GradientDescentVariant.batch:
            # clip the data so that the length is a multiple of folds
            entries_len = int(math.floor(data_entries_len / folds)) * folds
            # fold size
            val_batch_size = int(entries_len / folds)
            # all-except-single-fold
            train_batch_size = int(math.floor(((folds - 1) / folds) * entries_len))
            val_steps_per_epoch = train_steps_per_epoch = 1

        else:
            # clip the data so that the length is a multiple of folds
            entries_len = int(math.floor(data_entries_len / folds)) * folds
            val_batch_size = train_batch_size = 1
            val_steps_per_epoch = int(entries_len / folds)
            train_steps_per_epoch = int(math.floor(((folds - 1) / folds) * entries_len))

        fold_size = int(entries_len / folds)
        validation_scores: List[float] = intermediate_val_scores[:]
        for fold in range(current_fold, folds):
            print(f'Fold {current_fold + 1}/{folds}')
            if restored_model:
                model, model_state, history_dict = restored_model
                restored_model = None
            else:
                model = self.create_model()
                model_state = TrainingState(version=self._version, build=build if build else 0)
                model_state.training_target_epochs += epoch
                history_dict = {}

            model_dir = f'{self.data_path}/dev/{model_state.build}.v{model_state.version}'
            model_state_saver = ModelStateSaver(training_state=model_state, history=history_dict,
                                                model_folder=model_dir)

            val_generator = self.data_generator(
                data_entries[fold * fold_size: (fold + 1) * fold_size],
                val_batch_size)
            train_generator = self.data_generator(
                data_entries[:fold * fold_size] + data_entries[(fold + 1) * fold_size:],
                train_batch_size
            )

            history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=model_state.training_target_epochs - model_state.trained_epochs,
                validation_data=val_generator,
                validation_steps=val_steps_per_epoch,
                callbacks=[model_state_saver]
            )
            validation_scores.append(history.history['val_loss'][-1])
            self.state.intermediate_validation_scores = validation_scores[:]
            self.state.completed_folds += 1
            self._update_state(self.state)

        print('Validation scores')
        print(validation_scores)
        return numpy.average(validation_scores)

    def _train_prod_core(self, model: Model, model_state: TrainingState, history_dict: Dict[str, Any]) -> Model:

        model_dir = f'{self.data_path}/prod/{model_state.build}.v{model_state.version}'
        model_state_saver = ModelStateSaver(training_state=model_state, history=history_dict, model_folder=model_dir)

        data_entries = self.data_locators()
        data_entries_len = len(data_entries)

        if self._gradientVariant == GradientDescentVariant.miniBatch:
            batches = int(math.floor(data_entries_len / self._miniBatchSize))
            # clip the data so that the length is a multiple of batch size
            entries_len = batches * self._miniBatchSize
            batch_size = self._miniBatchSize
            steps_per_epoch = int(entries_len / self._miniBatchSize)

            print(f'Total entries: {data_entries_len}, used: {entries_len}')
            print(f'Batch size: {batch_size}')
            print(f'Steps: {steps_per_epoch}')

        elif self._gradientVariant == GradientDescentVariant.batch:
            entries_len = data_entries_len
            batch_size = entries_len
            steps_per_epoch = 1
        else:
            entries_len = data_entries_len
            batch_size = 1
            steps_per_epoch = entries_len

        generator = self.data_generator(data_locators=data_entries[:entries_len], batch_size=batch_size)
        model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=model_state.training_target_epochs - model_state.trained_epochs,
            callbacks=[model_state_saver]
        )
        return model

    def _update_state(self, state: ModelDescriptorState):
        if state:
            state.save_to_file(self.descriptor_path)
        elif isfile(self.descriptor_path):
            remove(self.descriptor_path)

        self.state = state

    def _dev_model_or_fresh_added_epoch(self, epoch: int = 10, build: Optional[int] = None, from_scratch: bool = False)\
            -> Tuple[Model, TrainingState, Dict[str, Any]]:

        # search if dev model exists for this version
        if not from_scratch:
            latest_dev_model_res = self.latest_dev_model() if not build else self.load_model(stage='dev',
                                                                                             build=build,
                                                                                             version=self._version)
        else:
            latest_dev_model_res = None

        if not latest_dev_model_res:
            model = self.create_model()
            model_state = TrainingState(version=self._version, build=build if build else 0)
            history_dict = {}
        else:
            model, model_state, history_dict = latest_dev_model_res

        model_state.training_target_epochs += epoch
        return model, model_state, history_dict

    def train_validate(self, epoch: int = 10, build: Optional[int] = None, from_scratch: bool = False):
        model, model_state, history_dict = self._dev_model_or_fresh_added_epoch(epoch, build, from_scratch)
        self._update_state(ModelDescriptorTrainingDevState(model_state.build))
        self._train_dev_core(model, model_state, history_dict)
        self._update_state(None)

    def kfold_validate(self, folds: int, epoch: int, build: Optional[int] = None) -> float:
        target_build = build if build else self.latest_build(stage='dev') if self.latest_build(stage='dev') else 0
        self._update_state(ModelDescriptorKFoldValidatingState(build=target_build,
                                                               folds=folds,
                                                               completed_folds=0,
                                                               intermediate_validation_scores=[]))
        validation_score = self._kfold_core(epoch=epoch,
                                            build=target_build,
                                            current_fold=0,
                                            folds=folds,
                                            intermediate_val_scores=[])
        self._update_state(None)
        return validation_score

    # should return best_run data
    def optimize(self,
                 max_evals: Optional[int] = None,
                 epoch: Optional[int] = None,
                 build: Optional[int] = None) -> Any:

        target_build = build if build else self.latest_build(stage='dev') if self.latest_build(stage='dev') else 0
        # if _restored_model_info was set and we are in optimizing state
        # we are resuming optimization, load trials for this evaluation
        if (self.state
                and self.state.type == ModelDescriptorStateType.optimizing
                and hasattr(self, '_restored_model_info')
                and self._restored_model_info
                and isfile(self.trials_path)):

            trials = pickle.load(open(self.trials_path, 'rb'))
            completed_evals = self.state.completed_evals
            target_evals = self.state.target_evals
        else:
            trials = Trials()
            # to make sure we don't hit warning on resume before first epoch completes
            pickle.dump(trials, open(self.trials_path, 'wb'))
            completed_evals = 0
            target_evals = max_evals if max_evals else 10

            self._update_state(
                ModelDescriptorOptimizingState(build=target_build, completed_evals=0, target_evals=max_evals)
            )

        def objective(space: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal epoch

            if (self.state
                    and self.state.type == ModelDescriptorStateType.optimizing
                    and hasattr(self, '_restored_model_info')
                    and self._restored_model_info
                    and isfile(self.trials_path)):

                model, model_state, history_dict = self._restored_model_info
                self._restored_model_info = None
                # model_state target_epoch should be used for all subsequent trainings
                epoch = model_state.training_target_epochs

                print(f'Eval: {self.state.completed_evals + 1}, using restored model')
            else:
                model = self.create_model(space)
                model_state = TrainingState(version=self._version, build=target_build)
                model_state.training_target_epochs += epoch if epoch else 10
                print(f'Eval: {self.state.completed_evals + 1}, using fresh model')

            validation_loss = self._train_dev_core(model, model_state, {})
            self.state.completed_evals += 1
            self.state.save_to_file(self.descriptor_path)
            return {'loss': validation_loss, 'status': STATUS_OK}

        best_run = None
        for i in range(completed_evals, target_evals):
            best_run = fmin(objective,
                            space=self.hyperopt_space(),
                            algo=tpe.suggest,
                            max_evals=i + 1,
                            trials=Trials(),
                            verbose=1)
            pickle.dump(trials, open(self.trials_path, 'wb'))

        self._update_state(None)
        return best_run

    def train_prod(self, epoch: Optional[int] = None, build: Optional[int] = None) -> Model:
        # if build passed check if the prod model exists
        if build:
            loaded_model_res = self.load_model('prod', build, self._version)
            if loaded_model_res:
                print(f'Prod model(build:{build}, version:{self._version} already exists. Nothing done.')
                _, _, history_dict = loaded_model_res
                return history_dict['val_loss'][-1]

            target_build = build
        else:
            # grab a build increment
            target_build = self.latest_build(stage='prod') + 1 if self.latest_build(stage='prod') else 0

        model = self.create_model()
        model_state = TrainingState(version=self._version, build=target_build)
        model_state.training_target_epochs += epoch
        history_dict = {}

        self._update_state(ModelDescriptorTrainingProdState(model_state.build))
        target_model = self._train_prod_core(model=model, model_state=model_state, history_dict=history_dict)
        self._update_state(None)
        return target_model

    def resume_if_needed(self) -> Any:
        if self.state.type == ModelDescriptorStateType.trainingDev:
            loaded_model_res = self.load_model(stage='dev', build=self.state.build, version=self._version)
            if not loaded_model_res:
                warn(f"Couldn't resume training. TrainingDev was set, "
                     f"but build:{self.state.build}, version:{self._version} dev-model is not available."
                     f"Version might have changed or model files were moved/deleted")
                self._update_state(None)
                return

            model, model_state, history_dict = loaded_model_res
            validation_score = self._train_dev_core(model, model_state, history_dict)
            self._update_state(None)
            return validation_score

        elif self.state.type == ModelDescriptorStateType.optimizing:
            loaded_model_res = self.load_model(stage='dev', build=self.state.build, version=self._version)
            if not loaded_model_res:
                warn(f"Couldn't resume training. Optimizing was set, "
                     f"but build:{self.state.build}, version:{self._version} dev-model is not available."
                     f"Version might have changed or model files were moved/deleted")
                self._update_state(None)
                return

            if not isfile(self.trials_path):
                warn(f"Couldn't resume training. Optimizing was set, "
                     f"but trials file not found")
                self._update_state(None)
                return

            self._restored_model_info = loaded_model_res
            return self.optimize()

        elif self.state.type == ModelDescriptorStateType.kFoldValidating:
            loaded_model_res = self.load_model(stage='dev', build=self.state.build, version=self._version)
            if not loaded_model_res:
                warn(f"Couldn't resume training. Kfold validating was set, "
                     f"but build:{self.state.build}, version:{self._version} dev-model is not available."
                     f"Version might have changed or model files were moved/deleted")
                self._update_state(None)
                return

            _, model_state, _ = loaded_model_res
            validation_score = self._kfold_core(epoch=model_state.training_target_epochs,
                                                build=self.state.build,
                                                current_fold=self.state.completed_folds,
                                                folds=self.state.folds,
                                                intermediate_val_scores=self.state.intermediate_validation_scores,
                                                restored_model=loaded_model_res)
            self._update_state(None)
            return validation_score

        elif self.state.type == ModelDescriptorStateType.trainingProd:
            loaded_model_res = self.load_model(stage='prod', build=self.state.build, version=self._version)
            if not loaded_model_res:
                warn(f"Couldn't resume training. TrainingProd was set, "
                     f"but build:{self.state.build}, version:{self._version} prod-model is not available."
                     f"Version might have changed or model files were moved/deleted")
                self._update_state(None)
                return

            model, model_state, history_dict = loaded_model_res
            target_model = self._train_prod_core(model, model_state, history_dict)
            self._update_state(None)
            return target_model

        else:
            print('No task to resume')
