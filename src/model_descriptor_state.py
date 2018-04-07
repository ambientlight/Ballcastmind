from typing import ClassVar, Union, List, Dict
from abc import abstractmethod
from enum import Enum
from os.path import isfile
from pprint import pformat
import json


class ModelDescriptorStateType(Enum):
    none = 'none'
    trainingDev = 'training-dev'
    optimizing = 'optimizing'
    kFoldValidating = 'k-fold-validating'
    trainingProd = 'training-prod'


class ModelDescriptorStateBase:
    @abstractmethod
    def save_to_file(self, file_path: str):
        raise NotImplemented

    def __repr__(self):
        return pformat(vars(self))


class ModelDescriptorTrainingDevState(ModelDescriptorStateBase):
    type: ClassVar[ModelDescriptorStateType] = ModelDescriptorStateType.trainingDev
    build: int

    def __init__(self, build: int):
        self.build = build

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'type': self.type.value,
                'build': self.build
            }), file=json_file)


class ModelDescriptorOptimizingState(ModelDescriptorStateBase):
    type: ClassVar[ModelDescriptorStateBase] = ModelDescriptorStateType.optimizing
    build: int
    completed_evals: int
    target_evals: int

    def __init__(self, build: int, completed_evals: int, target_evals: int):
        self.build = build
        self.completed_evals = completed_evals
        self.target_evals = target_evals

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'type': self.type.value,
                'build': self.build,
                'completed_evals': self.completed_evals,
                'target_evals': self.target_evals
            }), file=json_file)


class ModelDescriptorKFoldValidatingState(ModelDescriptorStateBase):
    type: ClassVar[ModelDescriptorStateBase] = ModelDescriptorStateType.kFoldValidating
    build: int
    completed_folds: int
    intermediate_histories: List[Dict[str, List[float]]]
    folds: int

    def __init__(self, build: int,
                 completed_folds: int, folds: int, intermediate_histories: List[Dict[str, List[float]]]):
        self.build = build
        self.completed_folds = completed_folds
        self.folds = folds
        self.intermediate_histories = intermediate_histories

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'type': self.type.value,
                'build': self.build,
                'completed_folds': self.completed_folds,
                'folds': self.folds,
                'intermediate_histories': self.intermediate_histories
            }), file=json_file)


class ModelDescriptorTrainingProdState(ModelDescriptorStateBase):
    type: ClassVar[ModelDescriptorStateType] = ModelDescriptorStateType.trainingProd
    build: int

    def __init__(self, build: int):
        self.build = build

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as json_file:
            print(json.dumps({
                'type': self.type.value,
                'build': self.build
            }), file=json_file)


ModelDescriptorState = Union[
    ModelDescriptorTrainingDevState,
    ModelDescriptorOptimizingState,
    ModelDescriptorKFoldValidatingState,
    ModelDescriptorTrainingProdState,
    None
]


def model_descriptor_state_from_file(file_path: str) -> ModelDescriptorState:

    if not isfile(file_path):
        return None

    with open(file_path, encoding='utf-8') as descriptor_state_file:
        descriptor_state_dict = json.load(descriptor_state_file)
        if descriptor_state_dict['type'] == ModelDescriptorStateType.trainingDev.value:
            return ModelDescriptorTrainingDevState(build=int(descriptor_state_dict['build']))
        elif descriptor_state_dict['type'] == ModelDescriptorStateType.optimizing.value:
            return ModelDescriptorOptimizingState(build=int(descriptor_state_dict['build']),
                                                  completed_evals=int(descriptor_state_dict['completed_evals']),
                                                  target_evals=int(descriptor_state_dict['target_evals']))
        elif descriptor_state_dict['type'] == ModelDescriptorStateType.kFoldValidating.value:
            def parse_history(history: Dict[str, List[str]]) -> Dict[str, List[float]]:
                target_dict = {}
                for key, value in history.items():
                    target_dict[key] = [float(string_entry) for string_entry in value]
                return target_dict

            return ModelDescriptorKFoldValidatingState(
                build=int(descriptor_state_dict['build']),
                completed_folds=int(descriptor_state_dict['completed_folds']),
                folds=int(descriptor_state_dict['folds']),
                intermediate_histories=[parse_history(history) for history in
                                        descriptor_state_dict['intermediate_histories']])
        elif descriptor_state_dict['type'] == ModelDescriptorStateType.trainingProd.value:
            return ModelDescriptorTrainingProdState(build=int(descriptor_state_dict['build']))
        else:
            return None
