import os
import torch
from torch.utils.data import Dataset
import numpy as np
import re
import sys


SMILE_REG = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|
#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

SPECIAL_TOKENS = {
    'reactant': '[REACTANT]',
    'solvent': '[SOLVENT]',
    'catalyst': '[CATALYST]',
    'reagent': '[REAGENT]',
    'celcius': '[CELSIUS]',
    'gram': '[GRAM]',
    'mole': '[MOLE]',
    'pad': '<PAD>',
    'sep': '<SEP>'
}

SEQ_START = set(['[REACTANT]', '[SOLVENT]', '[CATALYST]', '<EOS>'
                  '[REAGENT]', '[CELSIUS]', '<SEP>', '<BOS>'])
PAD_POS = 0

def get_position_indices(sequence):
    position_indices = []
    current_pos = -1
    for token in sequence:
        if token in SEQ_START:
            current_pos += 1
        position_indices.append(current_pos)
    return np.array(position_indices)

def pad_position_indices(pos_indices, max_length):
    # Shift position indices to avoid conflict with PAD_POS
    pos_indices = [pos_idx + 1 for pos_idx in pos_indices]
    padded_pos_indices = pos_indices + [PAD_POS] * (max_length - len(pos_indices))
    return padded_pos_indices

# mask applied on target sequence
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Convert each token in the sequence to its corresponding index
def convert_to_indices(sequence, token_to_index, max_length):
    indices = [token_to_index[token] for token in sequence]
    if len(indices) > max_length: return None
    assert len(indices) == len(sequence), f'{len(indices)} encoded but {len(sequence)} pass in'

    # padding
    indices.extend([token_to_index['<PAD>']] * (max_length - len(indices)))
    return indices

# def token_level_accuracy(output_tokens, tgt_output):
#     mask = tgt_output != 0
#     correct = (output_tokens == tgt_output) & mask
#     return correct.sum().item(), mask.sum().item()

# def sequence_level_accuracy(output_tokens, batch_tgt, tgt_output):
#     output_tokens_seq = output_tokens.reshape(batch_tgt.size(0), -1)
#     tgt_output_seq = tgt_output.reshape(batch_tgt.size(0), -1)
#     seq_mask = (tgt_output_seq != 0)
#     seq_correct = ((output_tokens_seq == tgt_output_seq) | (~seq_mask)).all(dim=1)
#     return seq_correct.sum().item()


def train(model, train_loader, optimizer, criterion, vocab_size, device):
    model.train()
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_exact_match = 0
    total_sequences = 0
    bs = 0
    
    for batch_src, batch_tgt, batch_src_pos, batch_tgt_pos in train_loader:
        optimizer.zero_grad()
        batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
        batch_src_pos, batch_tgt_pos = batch_src_pos.to(device), batch_tgt_pos.to(device)

        tgt_input, tgt_output = batch_tgt[:, :-1], batch_tgt[:, 1:]
        src_key_padding_mask = (batch_src == PAD_POS)
        tgt_key_padding_mask = (tgt_input == PAD_POS)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        output = model(batch_src, tgt_input, 
                       batch_src_pos, 
                       batch_tgt_pos[:, :-1], 
                       tgt_mask=tgt_mask,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.reshape(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            output_tokens = output.argmax(dim=1)
            mask = (tgt_output != PAD_POS)
            correct = (output_tokens == tgt_output) & mask
            total_correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

            # Compute sequence-level accuracy
            output_sequences = output_tokens.reshape(batch_tgt.size(0), -1)
            tgt_sequences = tgt_output.reshape(batch_tgt.size(0), -1)
            seq_correct = ((output_sequences == tgt_sequences) | (tgt_sequences == PAD_POS)).all(dim=1).sum().item()
            total_exact_match += seq_correct
            total_sequences += batch_tgt.size(0)
        bs += 1
        sys.stdout.write(f'Batch [{bs}|{len(train_loader)}] Loss: {loss.item()}\r')
        sys.stdout.flush()
    avg_loss = total_loss / len(train_loader)
    train_token_accuracy = total_correct_tokens / total_tokens
    train_sequence_accuracy = total_exact_match / total_sequences
    print()
    print(f'Loss: {avg_loss:.3f}, Train Acc: {train_token_accuracy:.3f}, Train Seq Acc: {train_sequence_accuracy:.3f}')
    return avg_loss, train_token_accuracy, train_sequence_accuracy

def validation(model, test_loader, criterion, vocab_size, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_exact_match = 0
        total_sequences = 0

        for batch_src, batch_tgt, batch_src_pos, batch_tgt_pos in test_loader:
            batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
            batch_src_pos, batch_tgt_pos = batch_src_pos.to(device), batch_tgt_pos.to(device)
        
            tgt_input, tgt_output = batch_tgt[:, :-1], batch_tgt[:, 1:]
            src_key_padding_mask = (batch_src == PAD_POS)
            tgt_key_padding_mask = (tgt_input == PAD_POS)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            output = model(batch_src, tgt_input, 
                       batch_src_pos, 
                       batch_tgt_pos[:, :-1], 
                       tgt_mask=tgt_mask,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask)
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            output_tokens = output.argmax(dim=1)

            mask = (tgt_output != PAD_POS)
            correct = (output_tokens == tgt_output) & mask
            total_correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

            # Compute sequence-level accuracy
            output_sequences = output_tokens.reshape(batch_tgt.size(0), -1)
            tgt_sequences = tgt_output.reshape(batch_tgt.size(0), -1)
            seq_correct = ((output_sequences == tgt_sequences) | (tgt_sequences == PAD_POS)).all(dim=1).sum().item()
            total_exact_match += seq_correct
            total_sequences += batch_tgt.size(0)
        
        avg_val_loss = total_loss / len(test_loader)
        token_accuracy = total_correct_tokens / total_tokens
        sequence_accuracy = total_exact_match / total_sequences
        print(f'Val Loss: {avg_val_loss:.3f}, Val Acc: {token_accuracy:.3f}, Val Seq Acc: {sequence_accuracy:.3f}')
    return avg_val_loss, token_accuracy, sequence_accuracy

class ReactionSequenceDataset(Dataset):
    def __init__(self, npz_file_path):
        data = np.load(npz_file_path)
        self.x = data['x']
        self.y = data['y']
        self.x_pos = data['x_pos']
        self.y_pos = data['y_pos']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        x_pos = torch.tensor(self.x_pos[idx], dtype=torch.long)
        y_pos = torch.tensor(self.y_pos[idx], dtype=torch.long)
        return x, y, x_pos, y_pos


class SimpleSmilesTokenizer(object):
    def __init__(self):
        self.regex = re.compile(SMILE_REG)
    
    def tokenize(self, text):
        tokens = [token for token in self.regex.findall(text)]
        return tokens