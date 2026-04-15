import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional

class StateExtractor:
    """Extract LSTM encoder hidden and cell states"""

    def __init__(self, model, phoneme_to_id:dict, device):
        self. model = model
        self.phoneme_to_id = phoneme_to_id
        self.device = device

    def _to_input_ids(self, phonemes: list[str]) -> torch.Tensor:
        """converts phonemes list to input ids tensor [1, seq_len]"""
        ids = [self.phoneme_to_id[p] for p in phonemes]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
    def _get_final_hidden(self, input_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """gets the final output of encoder (h, c) as numpy arrays of shape (hidden size)"""
        with torch.no_grad():
            h, c = self.model.encoder(input_ids) # [num_layers, batch, hidden_size]
            return h[-1, 0].cpu().numpy(), c[-1,0].cpu().numpy()
    
    def extract_sequential(self, phonemes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Feed phonemes one-by-one (prefix 1..N), return states at each step. 
        return: h_states, c_states: (seq_len, hidden_size)
        """
        h_list, c_list = [], []
        for i in range(1, len(phonemes)+1):
            input_ids = self._to_input_ids(phonemes[:i])
            h,c = self._get_final_hidden(input_ids)
            h_list.append(h)
            c_list.append(c)
        return np.stack(h_list), np.stack(c_list)
    
    def extract_final(self, phonemes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """get the final states of the full sequence"""
        input_ids = self._to_input_ids(phonemes)
        h,c = self._get_final_hidden(input_ids)
        return h, c
    
@dataclass
class StatesDataset:
    """Stores extracted states for an entire dataset.
    
    Keeps metadata (scalars/strings) in a DataFrame,
    and high-dimensional embeddings in separate numpy arrays.
    Rows are aligned by index.
    
    Attributes:
        metadata: DataFrame with columns [seq_id, wo, position, phoneme]
        h: np.ndarray of shape (N, hidden_size) — hidden states
        c: np.ndarray of shape (N, hidden_size) — cell states
        delta_h: np.ndarray of shape (N, hidden_size) — h deltas (first = h itself)
        delta_c: np.ndarray of shape (N, hidden_size) — c deltas (first = c itself)
    """
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    h: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    delta_h: Optional[np.ndarray] = None
    delta_c: Optional[np.ndarray] = None
    state: Optional[np.ndarray] = None
    delta_state: Optional[np.ndarray] = None

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        extractor: StateExtractor,
        phoneme_col: str = "No_Stress",
        word_col: str = "Word",
        append_eos: bool = True,
    ) -> "StatesDataset":
        """build dataset from DataFrame with phoneme sequences and conditions."""
        all_meta_rows = []
        all_h, all_c = [], []
        all_delta_h, all_delta_c = [], []

        for seq_id, row in df.iterrows():
            phonemes = list(row[phoneme_col])
            if append_eos:
                phonemes = phonemes + ["<EOS>"]
            
            # extract sequential states (shape_len, hidden_size)
            h_seq, c_seq = extractor.extract_sequential(phonemes)

            # deltas, dh[0] = 0
            dh = np.diff(h_seq, axis=0, prepend=h_seq[0:1])
            dc = np.diff(c_seq, axis=0, prepend=c_seq[0:1])
            # dh[0] = h[0]
            dh[0] = h_seq[0]
            dc[0] = c_seq[0]

            for pos, phoneme in enumerate(phonemes):
                all_meta_rows.append({
                    "seq_id": seq_id,
                    "word": row[word_col],
                    "position": pos,
                    "phoneme": phoneme,
                })
            
            all_h.append(h_seq)
            all_c.append(c_seq)
            all_delta_h.append(dh)
            all_delta_c.append(dc)

        metadata = pd.DataFrame(all_meta_rows).reset_index(drop=True)
        return StatesDataset(
            metadata=metadata,
            h=np.concatenate(all_h, axis=0),
            c=np.concatenate(all_c, axis=0),
            delta_h=np.concatenate(all_delta_h, axis=0),
            delta_c=np.concatenate(all_delta_c, axis=0),
            # state = np.concatenate([h, c], axis=1)
        )
    
    def __len__(self):
        return len(self.metadata)
    
    def get_mask(self, **filters) -> np.ndarray:
        """get a boolean mask for rows matching filters
        
        examples:
        ds.get_mask(phonemes="P")
        ds.get_mask(position=2, phoneme="AO")
        """
        mask = np.ones(len(self), dtype=bool)
        for col, val in filters.items():
            if isinstance(val,list):
                mask &= self.metadata[col].isin(val).values
            else:
                mask &= (self.metadata[col] ==val).values  
            
        return mask
    def get_embeddings(self, embed_type:str, mask:Optional[np.ndarray]=None)-> np.ndarray:
        """get embeddings by type optionally filtered"""
        arr = getattr(self, embed_type)
        if mask is not None:
           return arr[mask]
        return arr

    def copy(self) -> "StatesDataset":
        """Return a deep copy of metadata and embedding arrays."""
        return StatesDataset(
            metadata=self.metadata.copy(deep=True),
            h=self.h.copy() if self.h is not None else None,
            c=self.c.copy() if self.c is not None else None,
            delta_h=self.delta_h.copy() if self.delta_h is not None else None,
            delta_c=self.delta_c.copy() if self.delta_c is not None else None,
            state=self.state.copy() if self.state is not None else None,
            delta_state=self.delta_state.copy() if self.delta_state is not None else None,
        )

    def save(self, path: str):
        """save dataset to dist (npz + csv)"""
        arrays = {
            "h": self.h,
            "c": self.c,
            "delta_h": self.delta_h,
            "delta_c": self.delta_c,
        }
        if self.state is not None:
            arrays["state"] = self.state
        if self.delta_state is not None:
            arrays["delta_state"] = self.delta_state
        np.savez_compressed(f"{path}_embeddings.npz", **arrays)
        self.metadata.to_csv(f"{path}_metadata.csv", index=False)
    
    @staticmethod
    def load(path: str) -> "StatesDataset":
        "load dataset"
        data = np.load(f"{path}_embeddings.npz")
        metadata = pd.read_csv(f"{path}_metadata.csv")
        return StatesDataset(
            metadata=metadata,
            h=data["h"], c=data["c"],
            delta_h=data["delta_h"], delta_c=data["delta_c"],
            state=data["state"] if "state" in data.files else None,
            delta_state=data["delta_state"] if "delta_state" in data.files else None,
        )