from typing import ClassVar, Union
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


class ModeDescriptorState:
    @abstractmethod
    def save_to_file(self, file_path: str):
        raise NotImplemented

    def __repr__(self):
        return pformat(vars(self))


class ModelDescriptorTrainingDevState(ModeDescriptorState):
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


def model_descriptor_state_from_file(file_path: str) -> Union[ModelDescriptorTrainingDevState, None]:
    if not isfile(file_path):
        return None

    with open(file_path, encoding='utf-8') as descriptor_state_file:
        descriptor_state_dict = json.load(descriptor_state_file)
        if descriptor_state_dict['type'] == ModelDescriptorStateType.trainingDev.value:
            return ModelDescriptorTrainingDevState(build=int(descriptor_state_dict['build']))

