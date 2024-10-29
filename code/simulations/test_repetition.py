import argparse
import sys, torch
import pandas as pd
from pathlib import Path

from Levenshtein import editops

""" PATHS """
FILE_DIR = Path(__file__).resolve()
MODELS_DIR = FILE_DIR.parent.parent / "models"
WEIGHTS_DIR = MODELS_DIR.parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
sys.path.append(str(MODELS_DIR))

from DataGen import DataGen
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from utils import seed_everything, set_device
from plots import levenshtein_bar_graph

device = set_device()

""" COMMAND LINE INTERFACE """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='The model to be tested')
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args

""" TESTING LOOP """
def test_repetition(D: DataGen, model: str) -> pd.DataFrame:
    """ LOAD DATA """
    _, _, test_dl, vocab_size, index_to_phoneme = D.dataloaders()
    
    """ PARAMETERS """
    # Unpack parameters from model name
    h, l, d = [p[1:] for p in model.split('_')[1:-1]]
    hidden_size, num_layers, dropout = int(h), int(l), float(d)

    """ INITIALIZE MODEL """
    encoder = EncoderRNN(vocab_size, hidden_size, num_layers, dropout).to(device)
    decoder = DecoderRNN(hidden_size, vocab_size, num_layers, dropout).to(device)

    """ LOAD WEIGHTS """
    e = torch.load(WEIGHTS_DIR / f'encoder_{model}.pth', map_location=device, weights_only=True)
    d = torch.load(WEIGHTS_DIR / f'decoder_{model}.pth', map_location=device, weights_only=True)
    encoder.load_state_dict(e)
    decoder.load_state_dict(d)

    """ TESTING LOOP """
    predictions = []
    deletions = []
    insertions = []
    substitutions = []
    edit_distance = []

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs = inputs.to(device)
            targets = targets.to(device)
            insertion, deletion, substitution = 0, 0, 0

            encoder_hidden = encoder(inputs)
            decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
            decoder_output = decoder(decoder_input, encoder_hidden)

            prediction = torch.argmax(decoder_output, dim=-1)
            prediction = prediction.squeeze().cpu().tolist()
            targets = targets.squeeze().cpu().tolist()

            ops = editops(prediction, targets)
            for op, _, _ in ops:
                if op == 'insert': insertion += 1
                elif op == 'delete': deletion += 1
                elif op == 'replace': substitution += 1
        
            deletions.append(deletion)
            insertions.append(insertion)
            substitutions.append(substitution)
            edit_distance.append(len(ops))
            predictions.append([index_to_phoneme[i] for i in prediction][:-1])

    D.test_data['Prediction'] = predictions
    D.test_data['Deletions'] = deletions
    D.test_data['Insertions'] = insertions
    D.test_data['Substitutions'] = substitutions
    D.test_data['Edit Distance'] = edit_distance

    levenshtein_bar_graph(D.test_data, model)
    D.test_data.drop(columns=['Category'])

    return D.test_data

if __name__ == "__main__":
    D = DataGen()
    args = parse_args()
    seed_everything()
    test_repetition(args.name, D)
    