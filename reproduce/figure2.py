import os
import sys
import warnings
from ast import literal_eval

import pandas as pd
import torch

converters = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
}

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import test
from swp.utils.datasets import enrich_for_plotting
from swp.utils.models import get_model
from swp.utils.setup import seed_everything, set_device
from swp.viz.test.length import plot_length_errors
from swp.viz.test.position import plot_position_errors_smooth
from swp.viz.test.regressions import regression_plots
from swp.viz.test.sonority import plot_sonority_errors

# Set random seed and device
seed_everything(42)
device = set_device()

# Suppress warnings for nested tensors
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

# Load the model
model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
weights_path = "weights/75.pth"
batch_size = 2048

model = get_model(model_name)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)

# Evaluate on the Word Feature Evaluation Dataset
wfe_df = pd.read_csv("data/wfe.csv", converters=converters, index_col=0)
wfe_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=wfe_df)
wfe_results, _ = test(
    model=model,
    device=device,
    test_df=wfe_df,
    test_loader=wfe_loader,
)
wfe_results = enrich_for_plotting(wfe_results)
# Evaluate on the Sonority Sequencing Principle Dataset
ssp_df = pd.read_csv("data/ssp.csv", converters=converters, index_col=0)
ssp_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=ssp_df)
ssp_results, _ = test(
    model=model,
    device=device,
    test_df=ssp_df,
    test_loader=ssp_loader,
)
ssp_results = enrich_for_plotting(ssp_results)

plot_length_errors(wfe_results)
regression_plots(wfe_results)
plot_position_errors_smooth(wfe_results)
plot_sonority_errors(ssp_results)
