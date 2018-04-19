from os import makedirs
from os.path import isdir
from typing import Dict, Any, Union, Optional, List
import json

from keras.callbacks import Callback
from keras import Model


class ModelStateSaver(Callback):
    model: Model
    params: Any
    history: Dict[str, List[float]]
    training_state: Union[Any, Any]
    model_folder: str

    best_loss: Optional[float]

    def __init__(self,
                 training_state: Union[Any, Any],
                 history: Dict[str, Any],
                 model_folder: str):

        super().__init__()
        self.history = history
        self.training_state = training_state
        self.model_folder = model_folder
        self.best_loss = None

    def _save_training_state(self):
        model_path = f'{self.model_folder}/model.h5'
        training_state_path = f'{self.model_folder}/state.json'
        history_path = f'{self.model_folder}/history.json'

        if not isdir(self.model_folder):
            makedirs(self.model_folder)

        self.model.save(model_path)
        with open(history_path, 'w') as json_file:
            print(json.dumps(self.history), file=json_file)
        self.training_state.save_to_file(training_state_path)

        if 'loss' in self.history:
            losses = self.history['loss']
            if losses and len(losses) > 0:
                current_loss = losses[-1]
                if (self.best_loss is not None and current_loss < self.best_loss) or self.best_loss is None:
                    self.best_loss = current_loss
                    best_model_path = f'{self.model_folder}/best_model.h5'
                    self.model.save(best_model_path)

    def on_train_begin(self, logs=None):
        self._save_training_state()

    def on_epoch_end(self, epoch, logs=None):
        self.training_state.trained_epochs += 1
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self._save_training_state()








