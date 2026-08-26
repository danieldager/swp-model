#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from intervention.state_analysis.states_extract import StateExtractor, StatesDataset
from ast import literal_eval
from sklearn.decomposition import PCA
from typing import Optional
from intervention.models.repeat_model import get_model
from intervention.paths import DATASETS_DIR, STATES_DIR, get_phoneme_to_id, resolve_weights
from intervention.state_analysis.utils import test_model_repetition
from intervention.utils import seed_everything, set_device
#%%
#  Setup
seed_everything(42)
device = set_device()
model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
weights_path = resolve_weights("resources/weights/1024_75.pth")

model = get_model(model_name)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
phoneme_to_id = get_phoneme_to_id()
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
converters = {"Word": str, "Phonemes": literal_eval, "No_Stress": literal_eval}


# Load phoneme categories
phoneme_data = pd.read_csv(DATASETS_DIR / "phonemes.csv")
vowels = phoneme_data["Phoneme"][phoneme_data["Type"] == "V"].tolist()
consonants = phoneme_data["Phoneme"][phoneme_data["Type"] == "C"].tolist()

#%%
# test_df = pd.read_csv("datasets/wfe.csv", converters=converters)
# # drop duplicated rows
# test_df = test_df.drop([604, 588])
# test_df.to_csv("datasets/wfe.csv", index=False)
#%%
def extract_states_from_df(df, path):
    extractor = StateExtractor(model, phoneme_to_id, device)
    wfe_ds = StatesDataset.from_dataframe(df, extractor, word_col="Word")

    wfe_ds.state = np.concatenate([wfe_ds.h, wfe_ds.c], axis=1)
    wfe_ds.delta_state = np.concatenate([wfe_ds.delta_h, wfe_ds.delta_c], axis=1)

    # wfe_ds.metadata = wfe_ds.metadata.rename(columns={"seq_condition": "word"})
    wfe_ds.metadata["phoneme_type"] = wfe_ds.metadata["phoneme"].apply(lambda x: "C" if x in consonants else ("V" if x in vowels else "EOS"))
    wfe_ds.save(path)
    
#%%

test_df = pd.read_csv(DATASETS_DIR / "wfe.csv", converters=converters)
extract_states_from_df(test_df, str(STATES_DIR / "test_states"))

#%%
from intervention.paths import get_train_dataset
train_df = get_train_dataset()
extract_states_from_df(train_df, str(STATES_DIR / "train_states"))

#%%
# # check repetition model prediction on wfe dataset
# wfe_new = pd.read_csv("datasets/wfe.csv", converters=converters)
# for idx, row in wfe_new.iterrows():
#     seq = row['No_Stress'] + ["<EOS>"]
#     can_repeat, predicted_seq = test_model_repetition(phon_seq=seq, model=model, phoneme_to_id=phoneme_to_id, device=device)
#     wfe_new.at[idx, "can_repeat"] = can_repeat
#     wfe_new.at[idx, "predicted_seq"] = predicted_seq
# wfe_new

# wfe_new.to_csv("datasets/wfe_with_repetition.csv", index=False)


# %%
ssp_df = pd.read_csv(DATASETS_DIR / "ssp.csv", converters=converters)
extractor = StateExtractor(model, phoneme_to_id, device)
ssp_ds = StatesDataset.from_dataframe(ssp_df, extractor, word_col="Type")
ssp_ds.state = np.concatenate([ssp_ds.h, ssp_ds.c], axis=1)
ssp_ds.delta_state = np.concatenate([ssp_ds.delta_h, ssp_ds.delta_c], axis=1)
ssp_ds.save(str(STATES_DIR / "ssp_states"))
# %%
ssp_ds
# %%
