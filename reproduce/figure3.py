import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
import torch

converters = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
}

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import test
from swp.utils.embeddings import create_embeddings_LSTM_hook
from swp.utils.models import get_model
from swp.utils.setup import seed_everything, set_device
from swp.viz.embeddings import dendro_dmatrix, mlem_importance, mlem_phonemes

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

# Create hook to extract embeddings
buffer = {"Hidden": [], "Cell": []}
hook = create_embeddings_LSTM_hook(buffer)
hook_handle = model.encoder.recurrent.register_forward_hook(hook)

# Evaluate on the Single Phonemes Dataset
phonemes_df = pd.read_csv("data/phonemes.csv", converters=converters)
phonemes_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=phonemes_df)
phonemes_results = test(
    model=model,
    device=device,
    test_df=phonemes_df,
    test_loader=phonemes_loader,
)

# Save the embeddings for the Word Feature Evaluation Dataset
embeddings = np.concat(buffer["Hidden"])
hook_handle.remove()

# TODO: clarigy inconsistency, remove hardcoded path
# I believe it has to do with taking the embeddings from before stop token
npy_path = "../results/evaluation/Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1/b1024_l0.001_fall_s42_sn_ec/75/control/phonemes_h.npy"
embeddings = np.load(npy_path)[np.arange(len(phonemes_df)), phonemes_df.Length, :]

# Create a dendrogram and distance matrix for the embeddings
dendro_dmatrix(phonemes_df, embeddings)


# Separate the phonemes into vowels and consonants
# Drop irrelevant columns for either group
v_df = phonemes_df.query("Type == 'V' and Diphthong == False")
drops = [
    "Symbol",
    "Type",
    "Place",
    "Manner",
    "Voiced",
    "No_Stress",
    "Length",
    "Prediction",
]
v_df = v_df.drop(columns=drops)
v_emb = embeddings[v_df.index]

c_df = phonemes_df.query("Type == 'C'")
drops = [
    "Symbol",
    "Type",
    "Height",
    "Backness",
    "Diphthong",
    "No_Stress",
    "Length",
    "Prediction",
]
c_df = c_df.drop(columns=drops)
c_emb = embeddings[c_df.index]

# Calculate feature importances and plot
v_results = mlem_importance(v_df, v_emb)
c_results = mlem_importance(c_df, c_emb)
mlem_phonemes(v_results, c_results)
