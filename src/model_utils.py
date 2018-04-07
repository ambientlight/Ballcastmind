from typing import Dict, List
from os import mkdir
from os.path import isdir
import matplotlib.pyplot as plt


def write_charts_from_history(history: Dict[str, List[float]], charts_dir: str):
    for key in history.keys():
        if not key.startswith('val'):
            save_chart(charts_dir, history, key)


def save_chart(charts_dir: str, history: Dict[str, List[float]], key: str):
    metrics = history[key]
    metrics = metrics[1 if len(metrics) > 1 else 0:]
    val_metrics = history[f'val_{key}']
    val_metrics = val_metrics[1 if len(val_metrics) > 1 else 0:]
    epochs = range(1, len(metrics) + 1)

    if not isdir(charts_dir):
        mkdir(charts_dir)

    plt.plot(epochs, metrics, 'bo', label='Training')
    plt.plot(epochs, val_metrics, 'b', label='Validation')
    plt.title(f'Training and validation: {key}')
    plt.legend()
    plt.savefig(f'{charts_dir}/{key}.png')
    plt.cla()
    plt.clf()
