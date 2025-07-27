import os
import pandas as pd
from collections import Counter
import numpy as np
from utils import convert_to_indices, pad_position_indices
import pickle

MAX_SEQ_LENGTH = 300
MAX_TGT_LENGTH = 150
data_root = './data/USPTO_480K'
event_csv_path = os.path.join(data_root, 'smile_events.parquet')

dictionary_path = os.path.join(data_root, './dictionary.pkl')
word_path = os.path.join(data_root,'all_data.npz')

print('Loading events')
processed_df = pd.read_parquet(event_csv_path)
print(len(processed_df), 'rows in total')
kept = processed_df[(processed_df['x_position'].str.len() <= MAX_SEQ_LENGTH) & 
                    (processed_df['y_position'].str.len() <= MAX_TGT_LENGTH)]
print(f'{100*len(kept)/len(processed_df)}% data preserved')
# exit()

# Extract tokens from both input and product sequences and count their occurrences
print(f'Creating dictionary, n_reactions: {len(kept)}')
all_tokens = []
for _, row in kept.iterrows():
    all_tokens.extend(row['x'])  # Split input sequence into tokens
    all_tokens.extend(row['y'])  # Split product sequence into tokens

# Count occurrences and sort by frequency
token_counts = Counter(all_tokens)
unique_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)

# Create a dictionary mapping each token to a unique index and reserve 0 for padding token
token2index = {token: idx for idx, token in enumerate(unique_tokens, start=1)}
token2index['<PAD>'] = 0
index2token = {v: k for k, v in token2index.items()}

# Save dictionary
print('Dictionary Size', len(token2index))
pickle.dump((token2index, index2token), open(dictionary_path, 'wb'))

# Apply conversion to the dataset
print('Building dataset')
X = []
Y = []
x_pos = []
y_pos = []
for idx, row in kept.iterrows():
    input_idx = convert_to_indices(row['x'], token2index, MAX_SEQ_LENGTH)
    product_idx = convert_to_indices(row['y'], token2index, MAX_TGT_LENGTH)

    if input_idx is not None and product_idx is not None:
        padded_x_pos = pad_position_indices(row['x_position'], MAX_SEQ_LENGTH)
        padded_y_pos = pad_position_indices(row['y_position'], MAX_TGT_LENGTH)
        if max(padded_x_pos) <= 1 or max(padded_y_pos) < 1:
            print(idx, 'has too few compounds. Discard')
            continue
        X.append(input_idx)
        Y.append(product_idx)
        x_pos.append(padded_x_pos)
        y_pos.append(padded_y_pos)
        
    if (idx+1) % 100000 == 0:
        print('[Processing data]', idx+1)

X = np.array(X)
Y = np.array(Y)
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
print('Saving data')
np.savez_compressed(word_path, X=X, Y=Y, x_pos=x_pos, y_pos=y_pos)