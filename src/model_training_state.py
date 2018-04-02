from typing import Optional, Dict, Any, Union, ClassVar, TypeVar, Type
from pprint import pformat
import json


class TrainingState:
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

    @classmethod
    def from_file(cls, file_path: str, version: int, build: int) -> 'TrainingState':
        with open(file_path, encoding='utf-8') as training_state_file:
            training_state_dict = json.load(training_state_file)
            return cls(training_state_dict=training_state_dict,
                       version=version,
                       build=build)