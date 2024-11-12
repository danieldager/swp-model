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
from plots import errors_bar_chart, parametric_plots

device = set_device()

""" ARGUMENT PARSING """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args


""" TESTING LOOP """
def test_repetition(P: Phonemes, model_name: str) -> list:
    # Unpack parameters from model name
    e, h, l, d = [p[1:] for p in model_name.split('_')[:-1]]
    num_epochs, hidden_size, num_layers, dropout = int(e), int(h), int(l), float(d)

    # Unpack variables from Phonemes class
    test_data = P.test_data
    vocab_size = P.vocab_size
    index_to_phone = P.index_to_phone
    test_dataloader = P.test_dataloader

    # Add checkpoints for proper iteration
    checkpoints = [f"1_{i}" for i in range(1, 11)]
    epochs = checkpoints + list(range(2, num_epochs + 1))

    dataframes = []
    for epoch in epochs:

        """ LOAD MODEL """
        MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model_name
        encoder_path = MODEL_WEIGHTS_DIR / f'encoder{epoch}.pth'
        decoder_path = MODEL_WEIGHTS_DIR / f'decoder{epoch}.pth'
        encoder = EncoderRNN(vocab_size, hidden_size, num_layers, dropout).to(device)
        decoder = DecoderRNN(hidden_size, vocab_size, num_layers, dropout).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        
        """ TESTING LOOP """
        deletions = []
        insertions = []
        substitutions = []
        edit_distance = []
        predictions = []
        test_data = test_data.copy()

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

        errors_bar_chart(test_data, model_name, epoch)
        parametric_plots(test_data, model_name, epoch)
        dataframes.append(test_data)
    
    return dataframes

if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    dfrs = test_repetition(P, args.name)
    