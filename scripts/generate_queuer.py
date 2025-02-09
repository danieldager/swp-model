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
    grid = Grid(
        {
            "model_class": ["Ua"],
            "vocab_size": [len(phoneme_to_id)],
            "start_token_id": [phoneme_to_id["<SOS>"]],
            "cnn_args": [None],
            "fold_id": list(range(5)),
            "num_epochs": [100],
            "batch_size": [1024, 2048, 4096],
            "recur_type": ["LSTM"],  # ["RNN", "LSTM"],
            "hidden_size": [64, 128],  # [256, 512],  # [2**i for i in range(4, 10)],
            "num_layers": [1],
            "learning_rate": [0.005, 0.001, 0.0005],  # [10**-i for i in range(2, 5)],
            "droprate": [0.0, 0.1, 0.2],  # [0.5, 0.6, 0.7, 0.8],
            "tf_ratio": [0.0],  # [
            #     0.0,
            #     0.05,
            #     0.1,
            # ],  # [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
            "include_stress": [False],
        }
    )
    return grid


if __name__ == "__main__":
    grid = get_grid()
    create_jean_zay_train_repetition_queuer(grid=grid, bypass_datagen=True)
