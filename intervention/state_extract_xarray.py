import numpy as np
import pandas as pd
import xarray as xr
import torch
from ast import literal_eval
from typing import List, Dict, Tuple

class StateExtractor:
    """Extract LSTM encoder hidden and cell states"""

    def __init__(self, model: torch.nn.Module, phoneme_to_id: dict, device: torch.device):
        self.model = model
        self.phoneme_to_id = phoneme_to_id
        self.device = device
        self.model.eval()

    def _to_input_ids(self, phonemes: List[str]) -> torch.Tensor:
        ids = [self.phoneme_to_id[p] for p in phonemes]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _get_final_hidden(self, input_ids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            h, c = self.model.encoder(input_ids)
            return h[-1, 0].cpu().numpy(), c[-1, 0].cpu().numpy()

    def extract_sequential(self, phonemes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        h_list, c_list = [], []
        for i in range(1, len(phonemes) + 1):
            h, c = self._get_final_hidden(self._to_input_ids(phonemes[:i]))
            h_list.append(h)
            c_list.append(c)
        return np.stack(h_list), np.stack(c_list)


def build_xarray_dataset(
    df: pd.DataFrame, 
    extractor: StateExtractor, 
    vowels: List[str], 
    consonants: List[str],
    word_col: str = "Word",
    phoneme_col: str = "No_Stress",
    append_eos: bool = True
) -> xr.Dataset:
    """Extracts states and builds a padded xarray Dataset."""
    words, all_h, all_c = [], [], []
    phonemes_padded, types_padded = [], []
    lengths = []
    
    # Determine determining max length
    max_len = df[phoneme_col].apply(len).max() + (1 if append_eos else 0)
    
    for _, row in df.iterrows():
        phonemes = list(row[phoneme_col])
        if append_eos:
            phonemes.append("<EOS>")
        
        length = len(phonemes)
        lengths.append(length)
        words.append(row[word_col])
        
        # Get states
        h_seq, c_seq = extractor.extract_sequential(phonemes)
        
        # Pad to max_len
        pad_len = max_len - length
        if pad_len > 0:
            h_seq = np.pad(h_seq, ((0, pad_len), (0, 0)), constant_values=np.nan)
            c_seq = np.pad(c_seq, ((0, pad_len), (0, 0)), constant_values=np.nan)
            phonemes.extend([""] * pad_len)
            
        all_h.append(h_seq)
        all_c.append(c_seq)
        
        # Phoneme types
        p_types = ["V" if p in vowels else ("C" if p in consonants else ("EOS" if p == "<EOS>" else "")) for p in phonemes]
        phonemes_padded.append(phonemes)
        types_padded.append(p_types)

    # Convert to arrays
    h = np.stack(all_h)  # (num_words, max_len, hidden_size)
    c = np.stack(all_c)
    
    # Deltas
    # dh[0] = h[0], effectively doing diff with padding
    dh = np.diff(h, axis=1, prepend=np.full_like(h[:, :1, :], np.nan))
    dc = np.diff(c, axis=1, prepend=np.full_like(c[:, :1, :], np.nan))
    # Fill actual first step with base value instead of nan
    for i, length in enumerate(lengths):
        if length > 0:
            dh[i, 0] = h[i, 0]
            dc[i, 0] = c[i, 0]

    # Combine states (h+c)
    state = np.concatenate([h, c], axis=2)
    delta_state = np.concatenate([dh, dc], axis=2)

    # Build dataset
    return xr.Dataset(
        data_vars={
            "h": (("word", "step", "hidden_dim"), h),
            "c": (("word", "step", "hidden_dim"), c),
            "delta_h": (("word", "step", "hidden_dim"), dh),
            "delta_c": (("word", "step", "hidden_dim"), dc),
            "state": (("word", "step", "state_dim"), state),
            "delta_state": (("word", "step", "state_dim"), delta_state),
        },
        coords={
            "word": words,
            "step": np.arange(max_len),
            "hidden_dim": np.arange(h.shape[-1]),
            "state_dim": np.arange(state.shape[-1]),
            "word_length": ("word", lengths),
            "phoneme": (("word", "step"), phonemes_padded),
            "phoneme_type": (("word", "step"), types_padded),
        }
    )
