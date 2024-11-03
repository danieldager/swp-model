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

from Phonemes import Phonemes
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from utils import seed_everything, set_device
from plots import errors_bar_chart

device = set_device()

""" ARGUMENT PARSING """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--epoch', type=int, default=None)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args

def load_model(model: str, vocab_size: int) -> tuple:    
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

    model_dict = {'n': model, 'e': encoder, 'd': decoder, 'h': hidden_size}
    return model_dict

""" TESTING LOOP """
def test_repetition(P: Phonemes, M: dict, epoch: int=None) -> pd.DataFrame:

    """ UNPACK VARIABLES """
    test_data = P.test_data
    test_dataloader = P.test_dataloader
    index_to_phone = P.index_to_phone
    model, encoder, decoder, hidden_size = M['n'], M['e'], M['d'], M['h']

    """ TESTING LOOP """
    predictions = []
    deletions = []
    insertions = []
    substitutions = []
    edit_distance = []

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
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
            predictions.append([index_to_phone[i] for i in prediction][:-1])

    test_data['Prediction'] = predictions
    test_data['Deletions'] = deletions
    test_data['Insertions'] = insertions
    test_data['Substitutions'] = substitutions
    test_data['Edit Distance'] = edit_distance

    errors_bar_chart(test_data, model, epoch)
    test_data.drop(columns=['Category'])

    return test_data

if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    model_dict = load_model(P, args.name)
    test_data = test_repetition(P, model_dict)
    