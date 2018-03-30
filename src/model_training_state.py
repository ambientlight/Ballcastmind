from typing import Optional, Dict, Any, Union
from pprint import pformat
import json


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
                'type': 'regular',
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
                'type': 'k-fold',
                'trained_epochs': self.trained_epochs,
                'training_target_epochs': self.training_target_epochs,
                'folds': self.folds,
                'current_fold': self.current_fold
            }), file=json_file)


def training_state_from_file(file_path: str, version: int, build: int)->Union[TrainingStateKFold, TrainingStateRegular]:
    with open(file_path, encoding='utf-8') as training_state_file:
        training_state_dict = json.load(training_state_file)
        if training_state_dict['type'] == 'k-fold':
            return TrainingStateKFold(training_state_dict=training_state_dict,
                                      version=version,
                                      build=build)
        else:
            return TrainingStateRegular(training_state_dict=training_state_dict,
                                        version=version,
                                        build=build)
