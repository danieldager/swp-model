
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from intervention.paths import get_phoneme_to_id


def get_median_angle(ph_embs_2d, mask):
    ph_angles = np.arctan2(ph_embs_2d[:,1], ph_embs_2d[:,0])
    ph_angle_mean = stats.circmean(ph_angles)
    return ph_angle_mean

def get_angle_inverse(pca, angle):
    point_2d = np.array([[np.cos(angle), np.sin(angle)]])
    inverse = pca.inverse_transform(point_2d)
    return inverse / np.linalg.norm(inverse)


def get_pca(mask, dataset, state='state', d=2):
    embs = dataset.get_embeddings(f"delta_{state}", mask)
    pca = PCA(n_components=d)
    embs_2d = pca.fit_transform(embs)
    return embs_2d, pca

def get_encoder_hidden(model, input_ids):
        """Get final hidden state from encoder LSTM."""
        with torch.no_grad():
            (h, c) = model.encoder(input_ids)
            return h[-1] , c[-1] 
def state_to_hc(state, device):
    # split on axis 1
    h = state.reshape(-1, 256)[:, :128]
    c = state.reshape(-1, 256)[:, 128:]
    
    h = h.float().to(device)
    c = c.float().to(device)
    h = h.unsqueeze(0)  # add batch dimension
    c = c.unsqueeze(0)  # add batch dimension
    return h, c
def h_c_to_state(h, c):
    h = torch.as_tensor(h)
    c = torch.as_tensor(c)
    return torch.cat((h, c), dim=-1)
        
def decode_to_phones(model, hidden_states, phon_seq,device, teacher_forcing=False):
    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    sos_token_id = phoneme_to_id["<SOS>"]
    inp = torch.full((1, 1), sos_token_id, dtype=torch.long).to(device)
    outputs = []

    for t in range(len(phon_seq)): 
        embedded = model.decoder.embedding(inp)
        embedded = model.decoder.dropout(embedded)
        out, hidden_states = model.decoder.recurrent(embedded, hidden_states)
        logits = out @ model.decoder.embedding.weight.T
        outputs.append(logits)
        
        if teacher_forcing:
            # Convert phoneme to token ID, then to tensor
            token_id = phoneme_to_id.get(phon_seq[t])
            inp = torch.full((1, 1), token_id, dtype=torch.long).to(device)
        else:
            inp = logits.argmax(dim=-1)

    predicted_phonemes = [id_to_phoneme[logit.argmax().item()] for logit in outputs]
    # print(f"target pos {pos}: {predicted_phonemes}")
    return predicted_phonemes

def test_model_repetition(phon_seq, model, phoneme_to_id, device):
    input_ids = [phoneme_to_id[p] for p in phon_seq]
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device) 
    with torch.no_grad():
        (h, c) = model.encoder(input_ids_tensor)
    pred_phonemes = decode_to_phones(model=model, hidden_states=(h, c), phon_seq=phon_seq, device=device, teacher_forcing=False)
    if pred_phonemes == list(phon_seq):
        return True, None
    else:
        return False, pred_phonemes