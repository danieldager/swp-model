import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.utils.datasets import get_phoneme_to_id
from swp.utils.grid_search import Grid
from swp.utils.task_queuer import create_jean_zay_train_repetition_queuer


def get_grid():
    phoneme_to_id = get_phoneme_to_id()
    grid: Grid = {
        "model_class": ["Ua"],
        "vocab_size": [len(phoneme_to_id)],
        "start_token_id": [phoneme_to_id["<SOS>"]],
        "cnn_args": [None],
        "fold_id": list(range(5)),
        "num_epochs": [50],
        "batch_size": [1024],
        "recur_type": ["RNN", "LSTM"],
        "hidden_size": [2**4, 2**5, 2**5, 2**7, 2**8, 2**9],
        "num_layers": [1],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        "droprate": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "tf_ratio": [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0],
        "include_stress": [False],
    }
    return grid


if __name__ == "__main__":
    grid = get_grid()
    create_jean_zay_train_repetition_queuer(grid=grid, bypass_datagen=True)
